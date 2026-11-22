import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import torch

import deepct_utils as du


def parse_args():
    p = argparse.ArgumentParser(description="Deep-CT: run all subnets, write OS_risk.csv and PFS_risk.csv")
    p.add_argument("--config",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
                   help="Path to config.yaml (default: config.yaml next to this script)")
    p.add_argument("--subnets", nargs="+", type=int, default=[1, 2, 3, 4], help="Subnets to run (default: 1 2 3 4)")
    p.add_argument("--output-dir")
    p.add_argument("--outcomes-file")
    p.add_argument("--device")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="Also save Sub1/Sub2/Sub3 score CSVs, Sub3 features and Sub4 subtypes")
    return p.parse_args()


def free(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    cfg = du.load_config(args.config if os.path.isfile(args.config) else None)
    P, I, W, R, O = (cfg[s] for s in du.CONFIG_SECTIONS)

    subnets = sorted(set(args.subnets))
    if not set(subnets) <= {1, 2, 3, 4}:
        sys.exit("--subnets must be from 1, 2, 3, 4")
    if 4 in subnets and 3 not in subnets:
        subnets = sorted(subnets + [3])   # Sub4 uses the Sub3 bottleneck features
        print("Note: Sub4 needs Sub3, so Sub3 is run as well.")

    weights_dir = du.pick(None, P.get("weights_dir"), "Weights")
    output_dir = du.pick(args.output_dir, P.get("output_dir"), "Results")
    dca_dir = du.pick(None, P.get("dca_dir"), "DCA_parameters")
    outcomes_file = du.pick(args.outcomes_file, O.get("file"))
    cohort = O.get("cohort")
    device = du.get_device(du.pick(args.device, R.get("device"), "auto"))
    batch_size = int(R.get("batch_size") or 2)
    os.makedirs(output_dir, exist_ok=True)

    print("Deep-CT — all subnets")
    print(f"  config   : {cfg.get('_config_file')}")
    print(f"  subnets  : {subnets}")
    print(f"  weights  : {weights_dir}")
    print(f"  output   : {output_dir}")
    print(f"  outcomes : {outcomes_file or 'none'}")
    print(f"  device   : {device}")

    # ---- inputs (shared by all subnets) ----
    manifest = I.get("manifest")
    if manifest:
        cases, unmatched = du.collect_cases_from_manifest(manifest), []
    else:
        dirs = [I.get("original_dir"), I.get("mediastinal_dir"), I.get("lung_dir")]
        if not all(dirs):
            sys.exit("No inputs: set inputs in config.yaml (three window folders or a manifest).")
        cases, unmatched = du.collect_cases_from_dirs(*dirs, I.get("pattern") or "*.nii.gz")
    if unmatched:
        print(f"WARNING: {len(unmatched)} file(s) not in all three folders, skipped: {unmatched[:5]}")
    if not cases:
        sys.exit("No complete cases found — check the input folders and the file pattern.")
    print(f"Found {len(cases)} patient(s).\n")

    os_tab = pd.DataFrame({"ID": [c["ID"] for c in cases]})
    pfs_tab = os_tab.copy()
    log = {"timestamp": datetime.now().isoformat(timespec="seconds"), "config_file": cfg.get("_config_file"),
           "subnets": subnets, "n_patients": len(cases), "skipped_unmatched_files": unmatched, "weights_sha256": {}}

    def add(k, df, os_col, pfs_col, name):
        nonlocal os_tab, pfs_tab
        os_tab = os_tab.merge(df[["ID", os_col]].rename(columns={os_col: name}), on="ID", how="left")
        pfs_tab = pfs_tab.merge(df[["ID", pfs_col]].rename(columns={pfs_col: name}), on="ID", how="left")
        if args.keep_intermediate:
            df.to_csv(os.path.join(output_dir, f"Sub{k}_{'subtypes' if k == 4 else 'risk_scores'}.csv"), index=False)

    # ---- Sub1 ----
    if 1 in subnets:
        print("[Sub1] 3D DenseNet121")
        m_os, m_pfs, ck = du.load_sub1_models(weights_dir, W.get("sub1_os") or du.SUB1_DEFAULT_OS_WEIGHTS,
                                              W.get("sub1_pfs") or du.SUB1_DEFAULT_PFS_WEIGHTS, device)
        s1 = du.predict_sub1(m_os, m_pfs, cases, device, batch_size=batch_size)
        del m_os, m_pfs; free(device)
        add(1, s1, "OS_log_risk_score", "PFS_log_risk_score", "Sub1_risk")
        log["weights_sha256"].update(ck)

    # ---- Sub2 ----
    if 2 in subnets:
        print("[Sub2] U-Net -> inner fusion -> survival head")
        unet_path = os.path.join(weights_dir, W.get("sub2") or du.SUB2_DEFAULT_UNET)
        unet = du.load_sub2_unet(unet_path, device, use_skip=True)
        feats = du.extract_sub2_unet_features(unet, cases, device)
        del unet; free(device)
        surv_path = os.path.join(weights_dir, W.get("sub2_survival") or du.SUB2_DEFAULT_SURVIVAL)
        sc = du.predict_sub2_survival(du.load_sub2_survival(surv_path),
                                      du.fuse_sub2(feats, du.load_dca_params(dca_dir)))
        s2 = pd.DataFrame({"ID": [c["ID"] for c in cases], "OS_log_risk_score": sc["OS"],
                           "PFS_log_risk_score": sc["PFS"]})
        add(2, s2, "OS_log_risk_score", "PFS_log_risk_score", "Sub2_risk")
        log["weights_sha256"].update({os.path.basename(p): du.sha256_file(p) for p in (unet_path, surv_path)})

    # ---- Sub3 (+ Sub4) ----
    if 3 in subnets:
        print("[Sub3] 3D autoencoder")
        m_os, m_pfs, ck = du.load_sub3_models(weights_dir, W.get("sub3_os") or du.SUB3_DEFAULT_OS_WEIGHTS,
                                              W.get("sub3_pfs") or du.SUB3_DEFAULT_PFS_WEIGHTS, device)
        s3, s3_feats = du.predict_sub3(m_os, m_pfs, cases, device, batch_size=batch_size)
        del m_os, m_pfs; free(device)
        add(3, s3, "OS_log_risk_score", "PFS_log_risk_score", "Sub3_risk")
        log["weights_sha256"].update(ck)
        if args.keep_intermediate:
            s3_feats.to_csv(os.path.join(output_dir, "Sub3_bottleneck_features.csv"), index=False)
        if 4 in subnets:
            print("[Sub4] DEC subtypes")
            sub4_path = os.path.join(weights_dir, W.get("sub4") or du.SUB4_DEFAULT_ARTIFACT)
            s4 = du.predict_sub4(du.load_sub4(sub4_path), s3_feats, device)
            add(4, s4, "OS_subtype", "PFS_subtype", "Sub4_subtype")
            log["weights_sha256"][os.path.basename(sub4_path)] = du.sha256_file(sub4_path)

    # ---- outcomes (optional) + save ----
    if outcomes_file:
        cols_in_file = set(du.read_table(outcomes_file).columns)
        for name, cols in (("OS", ("OS_time", "OS_event")), ("PFS", ("PFS_time", "PFS_event"))):
            if not any(c in cols_in_file for c in cols):
                print(f"  Note: the outcome file has no {cols[0]}/{cols[1]} columns; {name}_risk.csv has scores only.")
            elif name == "OS":
                os_tab = du.attach_outcomes(os_tab, outcomes_file, cohort=cohort, outcome_cols=cols)
            else:
                pfs_tab = du.attach_outcomes(pfs_tab, outcomes_file, cohort=cohort, outcome_cols=cols)

    os_path, pfs_path = os.path.join(output_dir, "OS_risk.csv"), os.path.join(output_dir, "PFS_risk.csv")
    os_tab.to_csv(os_path, index=False)
    pfs_tab.to_csv(pfs_path, index=False)
    du.write_run_log(os.path.join(output_dir, "extract_all_run_info.json"), log)
    print(f"\nSaved {os_path}")
    print(f"Saved {pfs_path}")


if __name__ == "__main__":
    main()