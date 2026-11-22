
import argparse
import csv
import sys
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import exposure
from skimage.transform import resize
import os
import pathlib

VERSION = "deepct-preproc 2.0"

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
FIXED_WINDOWS = {
    "lung_window": (-1350.0, 150.0),
    "media_window": (-125.0, 225.0),
}
ORIG_PERCENTILES = (0.5, 99.5)
VARIANTS = ("lung_window", "media_window", "original_window")

OUT_SIZE = 128            # output cube edge (voxels)
OUT_SPACING = 3.0         # output voxel size (mm)  -> 384 mm field of view
WORK_FACTOR = 2           # working grid = 256^3 at 1.5 mm
CLAHE_KERNEL_MM = 48.0    # CLAHE tile size in mm (= 32 voxels on the 1.5 mm working grid)
CLAHE_CLIP = 0.01         # skimage default
AIR_HU = -1024.0          # fill value outside the scanned volume

LUNG_HU_RANGE = (-1350.0, 5000.0)   # allowed HU inside the lung mask
NOT_HU_P1 = -500.0                  # 1st percentile of lung HU above this -> data is not in HU
THICK_SLICE_MM = 3.0                # warn above this slice thickness


class CaseError(Exception):
    """Problem with one case; it is logged and the batch continues."""


# ----------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------
def strip_nii(name):
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def find_pairs(ct_dir, mask_dir, ct_prefix, mask_prefix):
    """Return [(case_id, ct_path, mask_path or None)] for every CT in ct_dir."""
    cts = sorted(p for p in ct_dir.iterdir() if p.name.endswith((".nii.gz", ".nii")))
    pairs = []
    for ct_path in cts:
        stem = strip_nii(ct_path.name)
        core = stem[len(ct_prefix):] if ct_prefix and stem.startswith(ct_prefix) else stem
        candidates = [mask_dir / f"{mask_prefix}{core}{ext}" for ext in (".nii.gz", ".nii")]
        candidates.append(mask_dir / ct_path.name)
        mask_path = next((c for c in candidates if c.exists()), None)
        pairs.append((stem, ct_path, mask_path))
    return pairs


def load_case(ct_path, mask_path, reorient):
    ct_nii = nib.load(str(ct_path))
    mask_nii = nib.load(str(mask_path))
    if len(ct_nii.shape) != 3 or len(mask_nii.shape) != 3:
        raise CaseError(f"Expected 3D volumes, got CT {ct_nii.shape} and mask {mask_nii.shape}")
    if ct_nii.shape != mask_nii.shape:
        raise CaseError(f"Dimension mismatch: CT {ct_nii.shape} vs mask {mask_nii.shape}")
    if not np.allclose(ct_nii.affine, mask_nii.affine, rtol=0, atol=1e-4):
        raise CaseError("CT and lung mask have different spatial transforms (affines)")
    orientation_in = "".join(nib.aff2axcodes(ct_nii.affine))
    if reorient:
        ct_nii = nib.as_closest_canonical(ct_nii)
        mask_nii = nib.as_closest_canonical(mask_nii)
    ct = ct_nii.get_fdata(dtype=np.float32)
    mask = (np.asanyarray(mask_nii.dataobj) > 0).astype(np.uint8)
    return ct_nii, ct, mask, orientation_in


def validate(ct, mask, ct_nii, warns):
    """Raise CaseError on unusable input; append soft issues to `warns`. Returns QC info."""
    if not np.all(np.isfinite(ct)):
        raise CaseError("CT contains NaN/Inf values")
    lung = ct[mask == 1]
    if lung.size == 0:
        raise CaseError("Lung mask is empty")

    p1, median = np.percentile(lung, [1, 50])
    if p1 > NOT_HU_P1:
        raise CaseError(f"Lung intensities do not look like HU (1st percentile = {p1:.0f}); "
                        "check rescale slope/intercept")
    lmin, lmax = float(lung.min()), float(lung.max())
    if lmin < LUNG_HU_RANGE[0] or lmax > LUNG_HU_RANGE[1]:
        raise CaseError(f"Unexpected HU range inside lung mask: [{lmin:.0f}, {lmax:.0f}]")

    zooms = np.sqrt((ct_nii.affine[:3, :3] ** 2).sum(0))
    if zooms.max() > THICK_SLICE_MM:
        warns.append(f"thick slices ({zooms.max():.1f} mm): fine detail is limited")

    return {
        "shape": "x".join(map(str, ct.shape)),
        "spacing_mm": "x".join(f"{z:.3g}" for z in zooms),
        "lung_voxels": int(lung.size),
        "lung_hu_min": round(lmin, 1),
        "lung_hu_median": round(float(median), 1),
        "lung_hu_max": round(lmax, 1),
    }


# ----------------------------------------------------------------------------
# CLAHE
# ----------------------------------------------------------------------------
def clahe(volume, mask, kernel, three_d):
    """CLAHE on a [0, 1] volume; background reset to 0 afterwards.

    2D: each axial slice (axis 2). Slices without contrast are skipped, because skimage turns
    a constant image into all 1.0. 3D: one volumetric CLAHE with cubic tiles."""
    if volume.min() < 0 or volume.max() > 1:
        raise CaseError("CLAHE input must be in [0, 1]")
    if three_d:
        out = (exposure.equalize_adapthist(volume, kernel_size=kernel, clip_limit=CLAHE_CLIP)
               if volume.max() > volume.min() else np.zeros_like(volume))
        out = out.astype(np.float32)
    else:
        out = np.zeros_like(volume, dtype=np.float32)
        for k in range(volume.shape[2]):
            sl = volume[:, :, k]
            if sl.max() > sl.min():
                out[:, :, k] = exposure.equalize_adapthist(sl, kernel_size=kernel, clip_limit=CLAHE_CLIP)
    if not np.all(np.isfinite(out)) or out.min() < 0 or out.max() > 1:
        raise CaseError("CLAHE output invalid (non-finite or outside [0, 1])")
    out[mask == 0] = 0
    return out


# ----------------------------------------------------------------------------
# Physical pipeline
# ----------------------------------------------------------------------------
def lung_centred_grid(ct_affine, mask, spacing, size, warns):
    """RAS-aligned affine for a size^3 grid of `spacing` mm voxels centred on the lung bounding box."""
    idx = [np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
    lo = np.array([i[0] for i in idx], float)
    hi = np.array([i[-1] for i in idx], float)
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world = (ct_affine @ corners.T).T[:, :3]
    centre = (world.min(0) + world.max(0)) / 2
    extent = world.max(0) - world.min(0)
    fov = spacing * size
    if np.any(extent > fov):
        warns.append(f"lung extent {np.round(extent).astype(int).tolist()} mm exceeds the "
                     f"{fov:.0f} mm field of view; edges are cut")
    aff = np.diag([spacing, spacing, spacing, 1.0])
    aff[:3, 3] = centre - spacing * (size - 1) / 2
    return aff, extent


def resample(volume, in_aff, out_aff, shape, order, antialias, cval):
    """Resample `volume` (voxel->world in_aff) onto the grid out_aff with the given shape."""
    vol = volume.astype(np.float32, copy=False)
    if antialias:
        src = np.sqrt((in_aff[:3, :3] ** 2).sum(0))
        dst = np.sqrt((out_aff[:3, :3] ** 2).sum(0))
        sigma = np.maximum(0, (dst / src - 1) / 2)     # same rule as skimage.transform.resize
        if np.any(sigma > 0):
            vol = ndimage.gaussian_filter(vol, sigma, mode="nearest")
    m = np.linalg.inv(in_aff) @ out_aff
    return ndimage.affine_transform(vol, m[:3, :3], offset=m[:3, 3], output_shape=shape,
                                    order=order, mode="constant", cval=cval, prefilter=False)


def block_mean_masked(volume, mask, f):
    """Average lung voxels inside each f x f x f block (background does not darken edges)."""
    s = tuple(n // f for n in volume.shape)
    v = (volume * mask).reshape(s[0], f, s[1], f, s[2], f).sum(axis=(1, 3, 5))
    n = mask.astype(np.float32).reshape(s[0], f, s[1], f, s[2], f).sum(axis=(1, 3, 5))
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0).astype(np.float32)


def run_physical(ct_nii, ct, mask, clahe_3d, warns, info):
    work_size, work_spacing = OUT_SIZE * WORK_FACTOR, OUT_SPACING / WORK_FACTOR
    work_aff, extent = lung_centred_grid(ct_nii.affine, mask, work_spacing, work_size, warns)
    shape = (work_size,) * 3

    ct_w = resample(ct, ct_nii.affine, work_aff, shape, order=1, antialias=True, cval=AIR_HU)
    mask_w = (resample(mask, ct_nii.affine, work_aff, shape, order=1, antialias=False, cval=0) >= 0.5)
    mask_w = mask_w.astype(np.uint8)
    if mask_w.sum() == 0:
        raise CaseError("Lung mask is empty after resampling")

    lung = ct_w[mask_w == 1]
    lo, hi = np.percentile(lung, ORIG_PERCENTILES)
    if hi <= lo:
        raise CaseError("Invalid lung HU range for original_window")
    ranges = dict(FIXED_WINDOWS, original_window=(float(lo), float(hi)))
    info["original_window_hu"] = f"{lo:.0f} to {hi:.0f}"
    info["lung_extent_mm"] = "x".join(f"{e:.0f}" for e in extent)

    kernel = int(round(CLAHE_KERNEL_MM / work_spacing))
    outputs = {}
    for name in VARIANTS:
        a, b = ranges[name]
        win = (np.clip(ct_w, a, b) - a) / (b - a)
        win = np.clip(win, 0.0, 1.0)   # float32 rounding with non-integer limits can give 1.0000001
        win[mask_w == 0] = 0
        enhanced = clahe(win.astype(np.float32), mask_w, kernel, clahe_3d)
        outputs[name] = block_mean_masked(enhanced, mask_w, WORK_FACTOR)
    del ct_w

    frac =mask_w.reshape(OUT_SIZE, 2, OUT_SIZE, 2, OUT_SIZE, 2).mean(axis=(1, 3, 5))
    mask_out = (frac >= 0.5).astype(np.uint8)
    for name in VARIANTS:
        outputs[name] *= mask_out

    out_aff = work_aff @ np.diag([WORK_FACTOR] * 3 + [1.0])
    out_aff[:3, 3] = work_aff[:3, 3] + work_spacing * (WORK_FACTOR - 1) / 2   # block centre
    return outputs, mask_out, out_aff


# ----------------------------------------------------------------------------
# v1 pipeline (reproduces Main.py)
# ----------------------------------------------------------------------------
def run_v1(ct_nii, ct, mask, info):
    lung_ct = np.where(mask == 1, ct, 0).astype(np.float32)
    wins = {}
    for name, (a, b) in FIXED_WINDOWS.items():
        wins[name] = ((np.clip(lung_ct, a, b) - a) / (b - a)).astype(np.float32)
    lo, hi = round(float(lung_ct.min())), round(float(lung_ct.max()))
    if hi <= lo:
        raise CaseError("Invalid HU range")
    wins["original_window"] = ((lung_ct - lo) / (hi - lo)).astype(np.float32)
    info["original_window_hu"] = f"{lo} to {hi}"

    outputs = {}
    mask_out = resize(mask, (OUT_SIZE,) * 3, order=0, preserve_range=True,
                      anti_aliasing=False).astype(np.uint8)
    for name in VARIANTS:
        vol = wins.pop(name)
        enhanced = np.empty_like(vol)
        for k in range(vol.shape[2]):
            enhanced[:, :, k] = exposure.equalize_adapthist(vol[:, :, k])
        if not np.all(np.isfinite(enhanced)) or enhanced.min() < 0 or enhanced.max() > 1:
            raise CaseError(f"{name}: CLAHE output invalid")
        enhanced[mask == 0] = 0
        out = resize(enhanced, (OUT_SIZE,) * 3, order=1, preserve_range=True).astype(np.float32)
        outputs[name] = out * mask_out

    scale = np.array(ct_nii.shape[:3], float) / OUT_SIZE
    out_aff = ct_nii.affine @ np.block([[np.diag(scale), ((scale - 1) / 2)[:, None]],
                                        [np.zeros((1, 3)), np.ones((1, 1))]])
    return outputs, mask_out, out_aff


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------
def save_nifti(volume, affine, ref_header, path, descrip):
    is_mask = volume.dtype == np.uint8
    img = nib.Nifti1Image(volume, affine)            # fresh header: no stale HU scaling/display range
    img.header.set_xyzt_units(*ref_header.get_xyzt_units())
    img.header["cal_min"], img.header["cal_max"] = 0.0, 1.0
    img.header["descrip"] = descrip.encode()[:79]
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    if not is_mask:
        img.set_data_dtype(np.float32)
    nib.save(img, str(path))


# ----------------------------------------------------------------------------
# One case
# ----------------------------------------------------------------------------
def process_case(ct_path, mask_path, out_paths, args, warns):
    physical = args.geometry == "physical"
    ct_nii, ct, mask, orientation_in = load_case(ct_path, mask_path, reorient=physical)
    info = validate(ct, mask, ct_nii, warns)
    info["orientation_in"] = orientation_in

    if physical:
        outputs, mask_out, out_aff = run_physical(ct_nii, ct, mask, args.clahe_3d, warns, info)
        descrip = f"{VERSION} physical {'clahe3d' if args.clahe_3d else 'clahe2d'}"
    else:
        outputs, mask_out, out_aff = run_v1(ct_nii, ct, mask, info)
        descrip = f"{VERSION} v1"

    for name in VARIANTS:
        save_nifti(outputs[name].astype(np.float32), out_aff, ct_nii.header, out_paths[name], f"{descrip} {name}")
    save_nifti(mask_out.astype(np.uint8), out_aff, ct_nii.header, out_paths["lung_mask"], f"{descrip} lung_mask")
    info["output_spacing_mm"] = "x".join(f"{z:.3g}" for z in np.sqrt((out_aff[:3, :3] ** 2).sum(0)))
    info["lung_voxels_128"] = int(mask_out.sum())
    return info


# ----------------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------------
LOG_FIELDS = ["case_id", "status", "ct_file", "mask_file", "shape", "spacing_mm", "orientation_in",
              "lung_extent_mm", "output_spacing_mm", "lung_voxels", "lung_voxels_128",
              "lung_hu_min", "lung_hu_median", "lung_hu_max", "original_window_hu", "warnings", "error"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deep-CT preprocessing: CT + lung mask -> 128^3 CLAHE inputs")
    ap.add_argument("--ct-dir", required=True, type=Path)
    ap.add_argument("--mask-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--ct-prefix", default="CT_", help="prefix stripped from CT names to get the case id (default CT_)")
    ap.add_argument("--mask-prefix", default="LM_", help="prefix added to the case id to find the mask (default LM_)")
    ap.add_argument("--geometry", choices=["physical", "v1"], default="physical",
                    help="physical = 3 mm isotropic, lung-centred (default); v1 = reproduce Main.py")
    ap.add_argument("--clahe-3d", action="store_true", help="volumetric CLAHE instead of per-slice (physical only)")
    ap.add_argument("--skip-existing", action="store_true", help="skip cases whose outputs already exist")
    args = ap.parse_args(argv)

    names = VARIANTS + ("lung_mask",)
    out_dirs = {name: args.out_dir / name for name in names}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.ct_dir, args.mask_dir, args.ct_prefix, args.mask_prefix)
    if not pairs:
        sys.exit(f"No .nii/.nii.gz files found in {args.ct_dir}")

    print(f"{VERSION} | geometry={args.geometry}" + (" | 3D CLAHE" if args.clahe_3d else ""))
    log_path = args.out_dir / "preprocessing_log.csv"
    n_ok = n_fail = 0
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for i, (case_id, ct_path, mask_path) in enumerate(pairs, 1):
            row = {"case_id": case_id, "ct_file": ct_path.name, "mask_file": mask_path.name if mask_path else ""}
            out_paths = {name: out_dirs[name] / f"{case_id}.nii.gz" for name in names}
            warns = []
            print(f"[{i:3d}/{len(pairs)}] {case_id} ... ", end="", flush=True)

            if args.skip_existing and all(p.exists() for p in out_paths.values()):
                row["status"] = "skipped"
                print("skipped (outputs exist)")
            elif mask_path is None:
                row.update(status="failed", error=f"No lung mask found for {ct_path.name}")
                n_fail += 1
                print("FAILED: no mask")
            else:
                try:
                    row.update(process_case(ct_path, mask_path, out_paths, args, warns))
                    row["status"] = "ok"
                    n_ok += 1
                    print("ok" + (f"  [warning: {'; '.join(warns)}]" if warns else ""))
                except CaseError as e:
                    row.update(status="failed", error=str(e))
                    n_fail += 1
                    print(f"FAILED: {e}")
                except Exception as e:  # unexpected: keep the batch going, record the traceback
                    row.update(status="failed", error=f"{type(e).__name__}: {e}")
                    n_fail += 1
                    print(f"FAILED (unexpected): {e}")
                    traceback.print_exc()
            row["warnings"] = "; ".join(warns)
            writer.writerow(row)
            fh.flush()

    print(f"\nDone: {n_ok} ok, {n_fail} failed. Log: {log_path}")

root = pathlib.Path.cwd()

if __name__ == "__main__":
    if len(sys.argv) > 1:          # user passed --ct-dir / --mask-dir / --out-dir
        main()
    else:                          # default: the Hugo example
        main(["--ct-dir", os.path.join(root, "Hugo_data", "CT"),
              "--mask-dir", os.path.join(root, "Hugo_data", "lung_mask"),
              "--out-dir", os.path.join(root, "Hugo_data", "Hugo_3D")])