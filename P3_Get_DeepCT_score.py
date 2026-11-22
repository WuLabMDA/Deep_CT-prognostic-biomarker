
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

import deepct_utils as du

SUBNET_COLUMN = {1: "Sub1_risk", 2: "Sub2_risk", 3: "Sub3_risk", 4: "Sub4_subtype"}


def parse_args():
    p = argparse.ArgumentParser(description="Deep-CT late fusion")
    p.add_argument("--config",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
                   help="Path to config.yaml (default: config.yaml next to this script)")
    p.add_argument("--subnets", nargs="+", type=int, help="Override fusion.subnets, e.g. 1 2 3")
    p.add_argument("--heads", nargs="+", help="Override fusion.heads, e.g. cox_ph mean")
    p.add_argument("--results-dir", help="Folder with OS_risk.csv / PFS_risk.csv (default: paths.output_dir)")
    p.add_argument("--weights-dir")
    p.add_argument("--list", action="store_true", help="List the available subnet/head combinations and exit")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = du.load_config(args.config if os.path.isfile(args.config) else None)
    P, I, W, R, O = (cfg[s] for s in du.CONFIG_SECTIONS)
    F = cfg.get("fusion") or {}

    weights_dir = du.pick(args.weights_dir, P.get("weights_dir"), "Weights")
    results_dir = du.pick(args.results_dir, P.get("output_dir"), "Results")
    fusion_path = os.path.join(weights_dir, W.get("fusion") or du.FUSION_DEFAULT_FILE)
    tree_path = os.path.join(weights_dir, W.get("fusion_trees") or du.FUSION_DEFAULT_TREE_FILE)
    art = du.load_fusion(fusion_path)
    trees = du.load_fusion_trees(tree_path) if os.path.isfile(tree_path) else None

    if args.list:
        print(du.available_fusions(art, trees).to_string(index=False))
        return

    subnets = sorted(set(int(k) for k in du.pick(args.subnets, F.get("subnets"), [1, 2, 3])))
    heads = du.pick(args.heads, F.get("heads"), ["cox_ph"])
    heads = [heads] if isinstance(heads, str) else list(heads)
    bad = [h for h in heads if h not in du.FUSION_ALL_HEADS]
    if bad or not subnets or not set(subnets) <= {1, 2, 3, 4}:
        sys.exit(f"Invalid choice. subnets must be from 1-4; heads from: {', '.join(du.FUSION_ALL_HEADS)}")

    print("Deep-CT late fusion")
    print(f"  subnets : {subnets}")
    print(f"  heads   : {heads}")
    print(f"  results : {results_dir}")

    log = {"timestamp": datetime.now().isoformat(timespec="seconds"), "config_file": cfg.get("_config_file"),
           "subnets": subnets, "heads": heads, "fusion_sha256": du.sha256_file(fusion_path),
           "fusion_trees_sha256": du.sha256_file(tree_path) if trees else None, "outputs": {}}

    done = 0
    for endpoint in ("OS", "PFS"):
        in_path = os.path.join(results_dir, f"{endpoint}_risk.csv")
        if not os.path.isfile(in_path):
            print(f"  {endpoint}: {in_path} not found — skipped (run extract_all.py first).")
            continue
        tab = pd.read_csv(in_path)
        tab["ID"] = tab["ID"].astype(str).str.strip()
        missing = [SUBNET_COLUMN[k] for k in subnets if SUBNET_COLUMN[k] not in tab.columns]
        if missing:
            sys.exit(f"{in_path} has no {missing} — run extract_all.py with those subnets included.")

        # rename to the column names the frozen fusion expects: Subnet_k_<endpoint>_risk
        inputs = tab[["ID"] + [SUBNET_COLUMN[k] for k in subnets]].rename(
            columns={SUBNET_COLUMN[k]: f"Subnet_{k}_{endpoint}_risk" for k in subnets})
        incomplete = inputs.isna().any(axis=1)
        if incomplete.any():
            print(f"  {endpoint}: {int(incomplete.sum())} patient(s) with a missing subnet value — left blank.")

        out = pd.DataFrame({"ID": tab["ID"]})
        for head in heads:
            col = pd.Series(float("nan"), index=tab.index)
            try:
                col[~incomplete] = du.predict_fusion(art, inputs[~incomplete], endpoint, subnets, head, trees)
            except KeyError as exc:
                sys.exit(f"{exc.args[0]}  Run with --list to see what is available.")
            out[f"DeepCT_{endpoint}_{head.replace('_', '')}_head"] = col.values
        for c in (f"{endpoint}_time", f"{endpoint}_event"):
            if c in tab.columns:
                out[c] = tab[c].values

        out_path = os.path.join(results_dir, f"DeepCT_{endpoint}_risk.csv")
        out.to_csv(out_path, index=False)
        log["outputs"][endpoint] = {"file": out_path, "n": len(out)}
        print(f"Saved {out_path}  ({len(out)} patients)")
        done += 1

    if not done:
        sys.exit("Nothing fused: neither OS_risk.csv nor PFS_risk.csv was found.")
    du.write_run_log(os.path.join(results_dir, "DeepCT_run_info.json"), log)


if __name__ == "__main__":
    main()