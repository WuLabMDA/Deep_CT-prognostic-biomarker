"""
deepct_utils.py — shared utilities for Deep-CT inference.

Sections
--------
0. Configuration      (read config.yaml: where data, weights and outputs live)
1. General helpers   (device, file names, checksums, environment info)
2. Checkpoint I/O     (safe state_dict loading)
3. Input collection   (folders or manifest CSV -> list of cases)
4. Sub1               (3D DenseNet121 multitask survival network)
5. Sub3               (3D residual autoencoder + survival head; 256-d bottleneck features)
6. Sub4               (DEC subtyping on Sub3 features: frozen inference + one-time fitting)
7. Sub2               (U-Net features -> EWS + DCA inner fusion -> frozen survival SVM)
8. Late fusion        (combine Sub1-4 into final OS / PFS scores; user picks subnets + head)
9. Outcomes & checks  (optional outcome join, reference comparison, run log)

IMPORTANT: the model classes must keep their attribute names exactly
(backbone, fc1..fc4, ...) because the saved weights are loaded with
strict=True and the state_dict keys depend on those names.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

# =============================================================================
# 0. Configuration
# =============================================================================
# All locations are set in one config.yaml. Relative paths are resolved like this:
#   paths.data_root / weights_dir / output_dir / dca_dir -> relative to the config file's folder
#   inputs.original_dir / mediastinal_dir / lung_dir / manifest -> relative to data_root
#   outcomes.file                               -> relative to the config file's folder
# So if config.yaml sits in the package folder, the whole folder can be moved or
# shared without editing any path.

CONFIG_SECTIONS = ("paths", "inputs", "weights", "runtime", "outcomes")


def _resolve(p, root):
    if p is None or str(p).strip() == "":
        return None
    p = os.path.expanduser(str(p).strip())
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(root, p))


def load_config(path: Optional[str]) -> Dict[str, dict]:
    """Read a YAML config and resolve all paths to absolute ones.

    With path=None, returns an empty config (everything must then come from
    command-line arguments).
    """
    if path is None:
        return {s: {} for s in CONFIG_SECTIONS}
    import yaml  # pip install pyyaml

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for s in CONFIG_SECTIONS:
        cfg[s] = cfg.get(s) or {}

    cfg_dir = os.path.dirname(os.path.abspath(path))
    for key in ("data_root", "weights_dir", "output_dir", "dca_dir"):
        cfg["paths"][key] = _resolve(cfg["paths"].get(key), cfg_dir)
    data_root = cfg["paths"]["data_root"] or cfg_dir
    for key in ("original_dir", "mediastinal_dir", "lung_dir", "manifest"):
        cfg["inputs"][key] = _resolve(cfg["inputs"].get(key), data_root)
    cfg["outcomes"]["file"] = _resolve(cfg["outcomes"].get("file"), cfg_dir)
    cfg["_config_file"] = os.path.abspath(path)
    return cfg


def pick(cli_value, config_value, default=None):
    """Command-line value wins, then the config value, then the default."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


# =============================================================================
# 1. General helpers
# =============================================================================

NIFTI_EXTS = (".nii.gz", ".nii")


def get_device(name: str = "auto") -> torch.device:
    """'auto' -> CUDA if available, else CPU. Otherwise e.g. 'cpu', 'cuda:0'."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def strip_nifti_ext(filename: str) -> str:
    """'Patient_001.nii.gz' -> 'Patient_001' (also handles .nii)."""
    base = os.path.basename(filename)
    for ext in NIFTI_EXTS:
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 checksum of a file (used to record exactly which weights ran)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def environment_info() -> Dict[str, str]:
    """Library versions, recorded in the run log for reproducibility."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "pandas": pd.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    try:
        import monai
        info["monai"] = monai.__version__
    except ImportError:
        info["monai"] = "not installed"
    return info


# =============================================================================
# 2. Checkpoint I/O
# =============================================================================

def load_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return a plain state_dict.

    - Uses weights_only=True where supported (safe: tensors only, no pickled code).
    - Unwraps {'state_dict': ...}-style containers.
    - Strips a 'module.' prefix left by DataParallel/DDP.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Weights file not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # very old torch without the weights_only argument
        ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint at {path} is not a state_dict-like dictionary.")
    if ckpt and all(k.startswith("module.") for k in ckpt):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}
    return ckpt


def load_model_weights(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Strictly load weights into `model`, move it to `device`, set eval mode."""
    model.load_state_dict(load_state_dict(path, device), strict=True)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# 3. Input collection
# =============================================================================
# A "case" is a dict: {"ID": str, "original": path, "mediastinal": path, "lung": path}

def collect_cases_from_dirs(
    original_dir: str,
    mediastinal_dir: str,
    lung_dir: str,
    pattern: str = "*.nii.gz",
) -> Tuple[List[dict], List[str]]:
    """Pair the three window images by identical file name across three folders.

    Returns (cases, unmatched_filenames). A file is unmatched if it is missing
    from either of the other two folders.
    """
    for d in (original_dir, mediastinal_dir, lung_dir):
        if not os.path.isdir(d):
            raise NotADirectoryError(f"Input folder not found: {d}")

    cases, unmatched = [], []
    for orig_path in sorted(glob.glob(os.path.join(original_dir, pattern))):
        fname = os.path.basename(orig_path)
        med_path = os.path.join(mediastinal_dir, fname)
        lung_path = os.path.join(lung_dir, fname)
        if os.path.isfile(med_path) and os.path.isfile(lung_path):
            cases.append({"ID": strip_nifti_ext(fname), "original": orig_path,
                          "mediastinal": med_path, "lung": lung_path})
        else:
            unmatched.append(fname)
    _check_unique_ids(cases)
    return cases, unmatched


def collect_cases_from_manifest(manifest_csv: str) -> List[dict]:
    """Read cases from a CSV with columns: ID, original, mediastinal, lung.

    Relative paths are resolved relative to the manifest's own folder.
    """
    df = pd.read_csv(manifest_csv, dtype=str)
    required = ["ID", "original", "mediastinal", "lung"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest is missing columns: {missing_cols}")

    base = os.path.dirname(os.path.abspath(manifest_csv))
    cases, missing_files = [], []
    for _, row in df.iterrows():
        case = {"ID": str(row["ID"]).strip()}
        for key in ("original", "mediastinal", "lung"):
            p = str(row[key]).strip()
            p = p if os.path.isabs(p) else os.path.join(base, p)
            if not os.path.isfile(p):
                missing_files.append(p)
            case[key] = p
        cases.append(case)
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} file(s) in the manifest do not exist, e.g. {missing_files[:3]}")
    _check_unique_ids(cases)
    return cases


def _check_unique_ids(cases: Sequence[dict]) -> None:
    ids = [c["ID"] for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"Duplicate patient IDs: {dupes[:10]}")


# =============================================================================
# 4. Sub1 — 3D DenseNet121 multitask survival network
# =============================================================================
# Input contract (must match how the published scores were produced):
#   - three co-registered 3D NIfTI volumes per patient: original, mediastinal
#     and lung window, identical shape and affine;
#   - stacked as channels in the order [original, mediastinal, lung];
#   - resized to 128^3 with trilinear interpolation (align_corners=False);
#   - NO intensity normalisation inside the model: the windowing/cropping
#     upstream must reproduce the training preprocessing exactly.
# Output: log-risk scores (higher = higher risk), uncalibrated.

SUB1_INPUT_SIZE = (128, 128, 128)
SUB1_CHANNEL_ORDER = ("original", "mediastinal", "lung")
SUB1_DEFAULT_OS_WEIGHTS = "sub1_multitask_10_29_model1_OS.pth"
SUB1_DEFAULT_PFS_WEIGHTS = "sub1_multitask_10_29_model1_PFS.pth"


class Identity(nn.Module):
    def forward(self, x):
        return x


class Sub1DenseNet(nn.Module):
    """3D DenseNet121 backbone (1024-d features) + four heads.

    fc1: OS log-risk, fc2: PFS log-risk, fc3: age (auxiliary), fc4: 9 clinical
    labels (auxiliary). The auxiliary heads are unused at inference but must
    exist so the checkpoint loads with strict=True.
    """

    def __init__(self, drop_rate: float = 0.5):
        super().__init__()
        from monai.networks.nets import DenseNet121  # imported here so other subnets don't need MONAI

        self.backbone = DenseNet121(spatial_dims=3, in_channels=3, out_channels=1,
                                    dropout_prob=drop_rate)
        self.backbone.class_layers.out = Identity()
        self.fc1 = nn.Linear(1024, 1)  # OS
        self.fc2 = nn.Linear(1024, 1)  # PFS
        self.fc3 = nn.Linear(1024, 1)  # age (aux)
        self.fc4 = nn.Linear(1024, 9)  # clinical labels (aux)

    def forward(self, x):
        encoded = self.backbone(x)
        return self.fc1(encoded), self.fc2(encoded), torch.sigmoid(self.fc3(encoded)), self.fc4(encoded)


Dense_model = Sub1DenseNet  # backward-compatible name used in the original scripts


def load_ct_input(original_path: str, mediastinal_path: str, lung_path: str) -> torch.Tensor:
    """Load the three window volumes -> tensor of shape (1, 3, 128, 128, 128)."""
    imgs = [nib.load(p) for p in (original_path, mediastinal_path, lung_path)]  # = SUB1_CHANNEL_ORDER
    shapes = [img.shape for img in imgs]
    if any(len(s) != 3 for s in shapes):
        raise ValueError(f"Expected three 3D volumes, got shapes {shapes} ({original_path})")
    if not (shapes[0] == shapes[1] == shapes[2]):
        raise ValueError(f"The three volumes have different shapes {shapes} ({original_path})")
    affines = [img.affine for img in imgs]
    if not (np.allclose(affines[0], affines[1], atol=1e-4) and np.allclose(affines[0], affines[2], atol=1e-4)):
        raise ValueError(f"The three volumes have different affines/orientations ({original_path})")

    arr = np.stack([np.asarray(img.dataobj, dtype=np.float32) for img in imgs], axis=0)  # (3, X, Y, Z)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite voxel values (NaN/Inf) in {original_path}")

    x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)  # (1, 3, X, Y, Z)
    x = F.interpolate(x, size=SUB1_INPUT_SIZE, mode="trilinear", align_corners=False)
    return x


load_sub1_input = load_ct_input  # Sub1 and Sub3 share the same input preparation


def load_sub1_models(weights_dir: str, os_weights: str, pfs_weights: str,
                     device: torch.device) -> Tuple[nn.Module, nn.Module, Dict[str, str]]:
    """Build and load the OS-selected and PFS-selected Sub1 checkpoints.

    Returns (os_model, pfs_model, {weights file -> sha256}).
    """
    paths = {"OS": os.path.join(weights_dir, os_weights), "PFS": os.path.join(weights_dir, pfs_weights)}
    os_model = load_model_weights(Sub1DenseNet(), paths["OS"], device)
    pfs_model = load_model_weights(Sub1DenseNet(), paths["PFS"], device)
    checksums = {os.path.basename(p): sha256_file(p) for p in paths.values()}
    return os_model, pfs_model, checksums


@torch.inference_mode()
def predict_sub1(os_model: nn.Module, pfs_model: nn.Module, cases: Sequence[dict],
                 device: torch.device, batch_size: int = 2, verbose: bool = True) -> pd.DataFrame:
    """Score every case. OS score comes from fc1 of the OS checkpoint, PFS score
    from fc2 of the PFS checkpoint. Each image is loaded once and fed to both.

    Returns a DataFrame: ID, OS_log_risk_score, PFS_log_risk_score.
    """
    os_model.eval()
    pfs_model.eval()
    ids, os_scores, pfs_scores = [], [], []
    n = len(cases)
    for start in range(0, n, batch_size):
        chunk = cases[start:start + batch_size]
        x = torch.cat([load_ct_input(c["original"], c["mediastinal"], c["lung"]) for c in chunk]).to(device)
        os_out = os_model(x)[0]   # fc1
        pfs_out = pfs_model(x)[1]  # fc2
        ids.extend(c["ID"] for c in chunk)
        os_scores.extend(os_out.float().cpu().numpy().reshape(-1).tolist())
        pfs_scores.extend(pfs_out.float().cpu().numpy().reshape(-1).tolist())
        if verbose:
            print(f"  Sub1 progress: {min(start + batch_size, n)}/{n}")
    return pd.DataFrame({"ID": ids, "OS_log_risk_score": os_scores, "PFS_log_risk_score": pfs_scores})


# =============================================================================
# 5. Sub3 — 3D residual autoencoder with survival head (+ bottleneck features)
# =============================================================================
# Same input contract as Sub1 (load_ct_input: [original, mediastinal, lung], 128^3).
# Outputs:
#   OS score   = fc1 of the OS-selected checkpoint
#   PFS score  = fc2 of the PFS-selected checkpoint
#   Age_output (auxiliary) and the 256-d bottleneck features = PFS checkpoint
#   (bottleneck = AvgPool3d(8) over the 256x8x8x8 encoder output). These 256
#   features are the input to Sub4.
#
# AutoEncoder_New below is the lab's modification of MONAI's AutoEncoder
# (MONAI Consortium, Apache License 2.0). Keep it unchanged: attribute and
# module names define the state_dict keys of the saved weights.

SUB3_DEFAULT_OS_WEIGHTS = "sub3_AE_multitask_11_1_model1-rep1_OS.pth"
SUB3_DEFAULT_PFS_WEIGHTS = "sub3_AE_multitask_11_1_model1-rep1_PFS.pth"
SUB3_N_FEATURES = 256
SUB3_FEATURE_COLS = [f"feature_{i}" for i in range(1, SUB3_N_FEATURES + 1)]
SUB3_ARCH = dict(dimensions=3, in_channels=3, out_channels=3, num_res_units=2,
                 channels=(32, 64, 128, 256), strides=(2, 2, 2, 2),
                 inter_channels=(256, 256), inter_dilations=(2, 2), dropout=0.5)


class AutoEncoder_New(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, channels, strides,
                 kernel_size=3, up_kernel_size=3, num_res_units=0, inter_channels=None,
                 inter_dilations=None, num_inter_units=2, act=Act.PRELU, norm=Norm.INSTANCE,
                 dropout=None, bias=True):
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]
        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)
        self.intermediate, self.encoded_channels = self._get_intermediate_module(self.encoded_channels, num_inter_units)
        self.decode, _ = self._get_decode_module(self.encoded_channels, decode_channel_list, strides[::-1] or [1])
        self.survivalnet = self._get_survival_module(self.encoded_channels)
        self.fc1 = nn.Linear(1024, 1)  # OS
        self.fc2 = nn.Linear(1024, 1)  # PFS
        self.fc3 = nn.Linear(1024, 1)  # age (aux)
        self.fc4 = nn.Linear(1024, 9)  # clinical labels (aux)
        self.bn = nn.Sequential(nn.AvgPool3d(8), nn.Flatten())  # bottleneck features

    def _get_survival_module(self, in_channels):
        survNet = nn.Sequential()
        tmp = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm3d(256), nn.PReLU(), nn.Dropout3d(self.dropout),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=2, stride=2),
            nn.BatchNorm3d(512), nn.PReLU(), nn.Dropout3d(self.dropout),
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=2, stride=1),
            nn.BatchNorm3d(1024), nn.PReLU(), nn.Dropout3d(self.dropout),
            nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm3d(1024), nn.PReLU(), nn.Dropout3d(self.dropout),
            nn.Flatten(),
        )
        survNet.add_module("surv_FCN", tmp)
        return survNet

    def _get_encode_module(self, in_channels, channels, strides):
        encode = nn.Sequential()
        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            encode.add_module("encode_%i" % i, self._get_encode_layer(layer_channels, c, s, False))
            layer_channels = c
        return encode, layer_channels

    def _get_intermediate_module(self, in_channels, num_inter_units):
        intermediate = nn.Identity()
        layer_channels = in_channels
        if self.inter_channels:
            intermediate = nn.Sequential()
            for i, (dc, di) in enumerate(zip(self.inter_channels, self.inter_dilations)):
                if self.num_inter_units > 0:
                    unit = ResidualUnit(spatial_dims=self.dimensions, in_channels=layer_channels, out_channels=dc,
                                        strides=1, kernel_size=self.kernel_size, subunits=self.num_inter_units,
                                        act=self.act, norm=self.norm, dropout=self.dropout, dilation=di, bias=self.bias)
                else:
                    unit = Convolution(spatial_dims=self.dimensions, in_channels=layer_channels, out_channels=dc,
                                       strides=1, kernel_size=self.kernel_size, act=self.act, norm=self.norm,
                                       dropout=self.dropout, dilation=di, bias=self.bias)
                intermediate.add_module("inter_%i" % i, unit)
                layer_channels = dc
        return intermediate, layer_channels

    def _get_decode_module(self, in_channels, channels, strides):
        decode = nn.Sequential()
        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            decode.add_module("decode_%i" % i, self._get_decode_layer(layer_channels, c, s, i == (len(strides) - 1)))
            layer_channels = c
        return decode, layer_channels

    def _get_encode_layer(self, in_channels, out_channels, strides, is_last):
        if self.num_res_units > 0:
            return ResidualUnit(spatial_dims=self.dimensions, in_channels=in_channels, out_channels=out_channels,
                                strides=strides, kernel_size=self.kernel_size, subunits=self.num_res_units,
                                act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                last_conv_only=is_last)
        return Convolution(spatial_dims=self.dimensions, in_channels=in_channels, out_channels=out_channels,
                           strides=strides, kernel_size=self.kernel_size, act=self.act, norm=self.norm,
                           dropout=self.dropout, bias=self.bias, conv_only=is_last)

    def _get_decode_layer(self, in_channels, out_channels, strides, is_last):
        decode = nn.Sequential()
        conv = Convolution(spatial_dims=self.dimensions, in_channels=in_channels, out_channels=out_channels,
                           strides=strides, kernel_size=self.up_kernel_size, act=self.act, norm=self.norm,
                           dropout=self.dropout, bias=self.bias, conv_only=is_last and self.num_res_units == 0,
                           is_transposed=True)
        decode.add_module("conv", conv)
        if self.num_res_units > 0:
            ru = ResidualUnit(spatial_dims=self.dimensions, in_channels=out_channels, out_channels=out_channels,
                              strides=1, kernel_size=self.kernel_size, subunits=1, act=self.act, norm=self.norm,
                              dropout=self.dropout, bias=self.bias, last_conv_only=is_last)
            decode.add_module("resunit", ru)
        return decode

    def forward(self, x):
        x = self.encode(x)
        encoded = self.intermediate(x)
        x = self.decode(encoded)
        x1 = self.survivalnet(encoded)
        os_ = self.fc1(x1)
        pfs = self.fc2(x1)
        age = torch.sigmoid(self.fc3(x1))
        label = self.fc4(x1)
        f = self.bn(encoded)
        return os_, pfs, age, label, x, f


def build_sub3_model() -> AutoEncoder_New:
    return AutoEncoder_New(**SUB3_ARCH)


def load_sub3_models(weights_dir: str, os_weights: str, pfs_weights: str,
                     device: torch.device) -> Tuple[nn.Module, nn.Module, Dict[str, str]]:
    """Build and load the OS-selected and PFS-selected Sub3 checkpoints."""
    paths = {"OS": os.path.join(weights_dir, os_weights), "PFS": os.path.join(weights_dir, pfs_weights)}
    os_model = load_model_weights(build_sub3_model(), paths["OS"], device)
    pfs_model = load_model_weights(build_sub3_model(), paths["PFS"], device)
    checksums = {os.path.basename(p): sha256_file(p) for p in paths.values()}
    return os_model, pfs_model, checksums


@torch.inference_mode()
def predict_sub3(os_model: nn.Module, pfs_model: nn.Module, cases: Sequence[dict],
                 device: torch.device, batch_size: int = 2,
                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every case with Sub3. Each image is loaded once and fed to both models.

    Returns (scores, features):
      scores   : ID, OS_log_risk_score, PFS_log_risk_score, Age_output
      features : ID, feature_1 .. feature_256  (PFS checkpoint; input to Sub4)
    """
    os_model.eval()
    pfs_model.eval()
    ids, os_s, pfs_s, age_s, feats = [], [], [], [], []
    n = len(cases)
    for start in range(0, n, batch_size):
        chunk = cases[start:start + batch_size]
        x = torch.cat([load_ct_input(c["original"], c["mediastinal"], c["lung"]) for c in chunk]).to(device)
        os_out = os_model(x)[0]                       # fc1 (OS checkpoint)
        _, pfs_out, age_out, _, _, f = pfs_model(x)   # fc2, age, bottleneck (PFS checkpoint)
        ids.extend(c["ID"] for c in chunk)
        os_s.extend(os_out.float().cpu().numpy().reshape(-1).tolist())
        pfs_s.extend(pfs_out.float().cpu().numpy().reshape(-1).tolist())
        age_s.extend(age_out.float().cpu().numpy().reshape(-1).tolist())
        feats.append(f.float().cpu().numpy())
        if verbose:
            print(f"  Sub3 progress: {min(start + batch_size, n)}/{n}")
    scores = pd.DataFrame({"ID": ids, "OS_log_risk_score": os_s,
                           "PFS_log_risk_score": pfs_s, "Age_output": age_s})
    features = pd.DataFrame(np.concatenate(feats, axis=0), columns=SUB3_FEATURE_COLS)
    features.insert(0, "ID", ids)
    return scores, features


# =============================================================================
# 6. Sub4 — DEC subtyping on the Sub3 bottleneck features
# =============================================================================
# Pipeline (all parts fitted ONCE on the Train cohort, then frozen in one file):
#   256 features -> StandardScaler -> drop constant columns -> PCA (99% var)
#   -> DEC encoder (8-d latent) -> nearest cluster centre -> S1/S2
# S1 = higher-risk subtype: the cluster with the shorter median Train OS time
# (for OS_subtype) or Train PFS time (for PFS_subtype), oriented separately.
#
# Users only need predict_sub4() + the frozen file (no outcomes, no Train data).
# fit_sub4() reproduces Subnet_4_DEC_Final.py and is run once by the lab
# (see freeze_sub4.py).

SUB4_DEFAULT_ARTIFACT = "sub4_DEC_frozen.pth"
SUB4_FORMAT = "deepct-sub4-dec-v1"
SUB4_FIT_DEFAULTS = dict(pca_level=0.99, n_clusters=2, seed=123, latent_dim=8, hidden_dim=128,
                         batch_size=64, pretrain_epochs=100, finetune_epochs=100, lr=1e-3)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Sub4Autoencoder(nn.Module):
    """Same layers and names as 'Autoencoder' in Subnet_4_DEC_Final.py."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def _target_distribution(q):
    weight = q ** 2 / (q.sum(axis=0, keepdims=True) + 1e-10)
    return weight / (weight.sum(axis=1, keepdims=True) + 1e-10)


def _student_t_assign(z, centers):
    dist_sq = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)
    q = 1.0 / (1.0 + dist_sq)
    return q / q.sum(dim=1, keepdim=True)


def _fit_dec(x_train: np.ndarray, device: torch.device, p: dict):
    """DEC fitting, step for step as in Subnet_4_DEC_Final.py (same RNG order)."""
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.cluster import KMeans

    model = Sub4Autoencoder(x_train.shape[1], p["hidden_dim"], p["latent_dim"]).to(device)
    tensor = torch.tensor(x_train, dtype=torch.float32)
    pretrain_loader = DataLoader(TensorDataset(tensor), batch_size=p["batch_size"], shuffle=True)
    finetune_loader = DataLoader(TensorDataset(tensor), batch_size=p["batch_size"], shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])

    model.train()  # stage 1: autoencoder pretraining
    for _ in range(p["pretrain_epochs"]):
        for (xb,) in pretrain_loader:
            xb = xb.to(device)
            reconstruction, _ = model(xb)
            loss = nn.functional.mse_loss(reconstruction, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()  # k-means initialisation of the cluster centres
    xt = tensor.to(device)
    with torch.no_grad():
        z0 = model.encoder(xt).cpu().numpy()
    kmeans = KMeans(n_clusters=p["n_clusters"], n_init=30, random_state=p["seed"])
    kmeans.fit(z0)
    centers = nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [centers], lr=p["lr"])

    model.train()  # stage 2: DEC fine-tuning (KL divergence)
    for epoch in range(p["finetune_epochs"]):
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                q_all = _student_t_assign(model.encoder(xt), centers).cpu().numpy()
                p_all = torch.tensor(_target_distribution(q_all), dtype=torch.float32, device=device)
                model.train()
        offset = 0
        for (xb,) in finetune_loader:
            xb = xb.to(device)
            batch_n = len(xb)
            q = _student_t_assign(model.encoder(xb), centers)
            pt = p_all[offset:offset + batch_n]
            offset += batch_n
            loss = torch.sum(pt * torch.log((pt + 1e-10) / (q + 1e-10))) / batch_n
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, centers.detach().cpu().numpy()


def _high_risk_cluster(clusters: np.ndarray, times: pd.Series, n_clusters: int) -> Tuple[int, Dict[int, float]]:
    """Cluster with the shortest median time in Train = S1 (higher risk)."""
    df = pd.DataFrame({"c": clusters, "t": np.asarray(times)})
    medians = df.groupby("c")["t"].median()
    if len(medians) != n_clusters:
        raise RuntimeError("DEC produced fewer non-empty Train clusters than expected.")
    return int(medians.idxmin()), {int(k): float(v) for k, v in medians.items()}


def fit_sub4(train_features: pd.DataFrame, train_os_time: pd.Series, train_pfs_time: pd.Series,
             device: torch.device, **overrides) -> dict:
    """One-time fit on the Train cohort -> frozen artifact dict (see save_sub4)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    p = {**SUB4_FIT_DEFAULTS, **overrides}
    x_raw = train_features[SUB3_FEATURE_COLS].to_numpy(dtype=float)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    keep = np.std(x_scaled, axis=0) > 1e-12
    x_scaled = x_scaled[:, keep]

    set_seed(p["seed"])
    pca = PCA(n_components=p["pca_level"], svd_solver="full")
    x_pca = pca.fit_transform(x_scaled)
    model, centers = _fit_dec(x_pca, device, p)

    art = {
        "format": SUB4_FORMAT,
        "feature_cols": list(SUB3_FEATURE_COLS),
        "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float64),
        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float64),
        "keep_mask": torch.tensor(keep, dtype=torch.bool),
        "pca_mean": torch.tensor(pca.mean_, dtype=torch.float64),
        "pca_components": torch.tensor(pca.components_, dtype=torch.float64),
        "input_dim": int(x_pca.shape[1]),
        "hidden_dim": int(p["hidden_dim"]),
        "latent_dim": int(p["latent_dim"]),
        "ae_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "centers": torch.tensor(centers, dtype=torch.float32),
        "fit_params": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in p.items()},
        "n_train": int(len(train_features)),
    }
    train_clusters = _dec_clusters(art, x_raw, device)
    art["os_high_risk_cluster"], art["train_os_cluster_medians"] = _high_risk_cluster(
        train_clusters, train_os_time, p["n_clusters"])
    art["pfs_high_risk_cluster"], art["train_pfs_cluster_medians"] = _high_risk_cluster(
        train_clusters, train_pfs_time, p["n_clusters"])
    return art


def _dec_clusters(art: dict, x_raw: np.ndarray, device: torch.device) -> np.ndarray:
    """Frozen transform: raw 256 features -> DEC cluster index."""
    x = (x_raw - art["scaler_mean"].numpy()) / art["scaler_scale"].numpy()
    x = x[:, art["keep_mask"].numpy()]
    x = (x - art["pca_mean"].numpy()) @ art["pca_components"].numpy().T
    model = Sub4Autoencoder(art["input_dim"], art["hidden_dim"], art["latent_dim"])
    model.load_state_dict(art["ae_state_dict"], strict=True)
    model.to(device).eval()
    with torch.no_grad():
        z = model.encoder(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    centers = art["centers"].numpy()
    d = np.sqrt(((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    return np.argmin(d, axis=1)


def save_sub4(art: dict, path: str) -> None:
    torch.save(art, path)


def load_sub4(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sub4 frozen model not found: {path}")
    try:
        art = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        art = torch.load(path, map_location="cpu")
    if not isinstance(art, dict) or art.get("format") != SUB4_FORMAT:
        raise ValueError(f"{path} is not a frozen Deep-CT Sub4 file (expected format '{SUB4_FORMAT}').")
    return art


def predict_sub4(art: dict, features: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Assign subtypes from Sub3 bottleneck features (ID + feature_1..256).

    Returns: ID, DEC_cluster, OS_subtype, PFS_subtype  (S1 = higher risk).
    """
    missing = [c for c in art["feature_cols"] if c not in features.columns]
    if missing:
        raise ValueError(f"Features table is missing {len(missing)} columns, e.g. {missing[:3]}")
    x_raw = features[art["feature_cols"]].to_numpy(dtype=float)
    if not np.isfinite(x_raw).all():
        raise ValueError("Non-finite values in the Sub3 features.")
    clusters = _dec_clusters(art, x_raw, device)
    return pd.DataFrame({
        "ID": features["ID"].values,
        "DEC_cluster": clusters,
        "OS_subtype": np.where(clusters == art["os_high_risk_cluster"], "S1", "S2"),
        "PFS_subtype": np.where(clusters == art["pfs_high_risk_cluster"], "S1", "S2"),
    })


# =============================================================================
# 7. Sub2 — U-Net features -> inner fusion (EWS + DCA) -> survival SVM
# =============================================================================
# Per patient, each CT window (lung, media, original) goes through the same U-Net
# encoder separately, giving features at 4 levels: 32, 64, 128, 256 dims.
# Inner fusion, per level (same order as Step_4_Inner_Fusion_module.py):
#   EWS = lung * media * original                       (element-wise product)
#   DCA = for pairs LM, LO, MO: [X @ Ax.T, Y @ Ay.T]    (Ax, Ay fitted on Train; .npz files)
# Fused vector = [EWS level 0..3 | DCA level 0..3]  ->  480 + 24 = 504 dims
# Survival head (fitted on Train, frozen in one file by Step_5_Survival_module.py):
#   StandardScaler -> PCA(32) -> per endpoint: StandardScaler -> linear FastSurvivalSVM.
#   Score = sign * (x @ coef + intercept); higher = higher risk (same as model.predict).

SUB2_LEVEL_DIMS = {0: 32, 1: 64, 2: 128, 3: 256}
SUB2_CHANNELS = ("lung", "media", "original")
SUB2_PAIRS = (("LM", "lung", "media"), ("LO", "lung", "original"), ("MO", "media", "original"))
SUB2_FUSED_DIM = 504
SUB2_DEFAULT_SURVIVAL = "sub2_survival_frozen.pth"
SUB2_SURV_FORMAT = "deepct-sub2-survival-v1"


def load_dca_params(dca_dir: str) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """Read level_{0..3}_{LM,LO,MO}.npz (arrays Ax, Ay) from dca_dir."""
    params: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {name: {} for name, _, _ in SUB2_PAIRS}
    for level, dim in SUB2_LEVEL_DIMS.items():
        for pair_name, _, _ in SUB2_PAIRS:
            path = os.path.join(dca_dir, f"level_{level}_{pair_name}.npz")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"DCA parameter file not found: {path}")
            with np.load(path, allow_pickle=False) as saved:
                Ax = np.asarray(saved["Ax"], dtype=np.float64)
                Ay = np.asarray(saved["Ay"], dtype=np.float64)
            if Ax.shape[1] != dim or Ay.shape[1] != dim:
                raise ValueError(f"{path}: expected projections with {dim} columns, got {Ax.shape}, {Ay.shape}")
            params[pair_name][level] = {"Ax": Ax, "Ay": Ay}
    return params


def fuse_sub2(level_features: Dict[int, Dict[str, np.ndarray]],
              dca_params: Dict[str, Dict[int, Dict[str, np.ndarray]]]) -> np.ndarray:
    """level_features[level][channel] = (N x dim) array  ->  fused (N x 504) array."""
    ews_levels, dca_levels = [], []
    n_rows = None
    for level, dim in SUB2_LEVEL_DIMS.items():
        f = {}
        for ch in SUB2_CHANNELS:
            arr = np.asarray(level_features[level][ch], dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != dim:
                raise ValueError(f"Level {level} '{ch}' features have shape {arr.shape}; expected N x {dim}")
            if not np.isfinite(arr).all():
                raise ValueError(f"Level {level} '{ch}' features contain NaN/Inf")
            n_rows = arr.shape[0] if n_rows is None else n_rows
            if arr.shape[0] != n_rows:
                raise ValueError("Row-count mismatch across windows/levels")
            f[ch] = arr
        ews_levels.append(f["lung"] * f["media"] * f["original"])
        pair_feats = []
        for pair_name, ch_x, ch_y in SUB2_PAIRS:
            p = dca_params[pair_name][level]
            pair_feats.append(np.concatenate((f[ch_x] @ p["Ax"].T, f[ch_y] @ p["Ay"].T), axis=1))
        dca_levels.append(np.concatenate(pair_feats, axis=1))
    fused = np.concatenate((np.concatenate(ews_levels, axis=1), np.concatenate(dca_levels, axis=1)), axis=1)
    if fused.shape[1] != SUB2_FUSED_DIM:
        raise ValueError(f"Fused vector has {fused.shape[1]} dims; expected {SUB2_FUSED_DIM}")
    return fused


def load_sub2_feature_dir(feature_dir: str, cohort: str) -> Tuple[List[str], Dict[int, Dict[str, np.ndarray]]]:
    """Read Step-3 outputs: {channel}_{cohort}_{level}.npy and lung_{cohort}_case_names.npy."""
    ids_path = os.path.join(feature_dir, f"lung_{cohort}_case_names.npy")
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f"Case-name file not found: {ids_path}")
    ids = [strip_nifti_ext(str(x).strip()) for x in np.load(ids_path, allow_pickle=False)]
    feats: Dict[int, Dict[str, np.ndarray]] = {}
    for level in SUB2_LEVEL_DIMS:
        feats[level] = {}
        for ch in SUB2_CHANNELS:
            path = os.path.join(feature_dir, f"{ch}_{cohort}_{level}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Feature file not found: {path}")
            feats[level][ch] = np.load(path, allow_pickle=False)
    if feats[0]["lung"].shape[0] != len(ids):
        raise ValueError(f"{len(ids)} case names but {feats[0]['lung'].shape[0]} feature rows for {cohort}")
    return ids, feats


# ---- Sub2 U-Net (from the lab's fn/fn_unet.py; keep attribute names: they define the
# ---- state_dict keys of sub2_skip_model_0.000.pt) ----------------------------------

SUB2_DEFAULT_UNET = "sub2_skip_model_0.000.pt"
SUB2_UNET_ARCH = dict(in_channels=1, out_channels=1, n_blocks=4, start_filters=32, activation="relu",
                      normalization="batch", conv_mode="same", dim=3)
SUB2_WINDOW_KEYS = {"lung": "lung", "media": "mediastinal", "original": "original"}  # channel -> case key


def _u_conv(dim):
    return nn.Conv3d if dim == 3 else nn.Conv2d


def _u_get_conv(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True, dim=2):
    return _u_conv(dim)(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def _u_get_up(in_c, out_c, kernel_size=2, stride=2, dim=3, up_mode="transposed"):
    if up_mode == "transposed":
        return (nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d)(in_c, out_c, kernel_size=kernel_size, stride=stride)
    return nn.Upsample(scale_factor=2.0, mode=up_mode)


def _u_get_pool(kernel_size=2, stride=2, padding=0, dim=2):
    return (nn.MaxPool3d if dim == 3 else nn.MaxPool2d)(kernel_size=kernel_size, stride=stride, padding=padding)


def _u_get_act(activation):
    return {"relu": nn.ReLU(), "leaky": nn.LeakyReLU(negative_slope=0.1), "elu": nn.ELU()}[activation]


def _u_get_norm(normalization, num_channels, dim):
    if normalization == "batch":
        return nn.BatchNorm3d(num_channels) if dim == 3 else nn.BatchNorm2d(num_channels)
    if normalization == "instance":
        return nn.InstanceNorm3d(num_channels) if dim == 3 else nn.InstanceNorm2d(num_channels)
    if "group" in normalization:
        return nn.GroupNorm(num_groups=int(normalization.partition("group")[-1]), num_channels=num_channels)


def _u_autocrop(encoder_layer, decoder_layer):
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds, es = encoder_layer.shape[2:], decoder_layer.shape[2:]
        sl = [slice((d - e) // 2, (d + e) // 2) for d, e in zip(ds, es)]
        encoder_layer = encoder_layer[(slice(None), slice(None), *sl)]
    return encoder_layer, decoder_layer


class _UDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True, activation="relu", normalization=None,
                 dim=2, conv_mode="same"):
        super().__init__()
        self.pooling, self.normalization = pooling, normalization
        pad = 1 if conv_mode == "same" else 0
        self.conv1 = _u_get_conv(in_channels, out_channels, 3, 1, pad, True, dim)
        self.conv2 = _u_get_conv(out_channels, out_channels, 3, 1, pad, True, dim)
        if pooling:
            self.pool = _u_get_pool(2, 2, 0, dim)
        self.act1, self.act2 = _u_get_act(activation), _u_get_act(activation)
        if normalization:
            self.norm1 = _u_get_norm(normalization, out_channels, dim)
            self.norm2 = _u_get_norm(normalization, out_channels, dim)

    def forward(self, x):
        y = self.act1(self.conv1(x))
        if self.normalization:
            y = self.norm1(y)
        y = self.act2(self.conv2(y))
        if self.normalization:
            y = self.norm2(y)
        before_pooling = y
        if self.pooling:
            y = self.pool(y)
        return y, before_pooling


class _UUpBlock(nn.Module):
    """UpBlock_with_Skip (skip=True) or UpBlock_no_skip (skip=False) from fn_unet.py."""

    def __init__(self, in_channels, out_channels, skip, activation="relu", normalization=None, dim=3,
                 conv_mode="same", up_mode="transposed"):
        super().__init__()
        self.skip, self.normalization, self.up_mode = skip, normalization, up_mode
        pad = 1 if conv_mode == "same" else 0
        self.up = _u_get_up(in_channels, out_channels, 2, 2, dim, up_mode)
        self.conv0 = _u_get_conv(in_channels, out_channels, 1, 1, 0, True, dim)
        self.conv1 = _u_get_conv((2 if skip else 1) * out_channels, out_channels, 3, 1, pad, True, dim)
        self.conv2 = _u_get_conv(out_channels, out_channels, 3, 1, pad, True, dim)
        self.act0, self.act1, self.act2 = (_u_get_act(activation) for _ in range(3))
        if normalization:
            self.norm0 = _u_get_norm(normalization, out_channels, dim)
            self.norm1 = _u_get_norm(normalization, out_channels, dim)
            self.norm2 = _u_get_norm(normalization, out_channels, dim)

    def forward(self, encoder_layer, decoder_layer):
        up = self.up(decoder_layer)
        if self.up_mode != "transposed":
            up = self.conv0(up)
        up = self.act0(up)
        if self.normalization:
            up = self.norm0(up)
        if self.skip:
            enc, _ = _u_autocrop(encoder_layer, up)
            up = torch.cat((up, enc), 1)
        y = self.act1(self.conv1(up))
        if self.normalization:
            y = self.norm1(y)
        y = self.act2(self.conv2(y))
        if self.normalization:
            y = self.norm2(y)
        return y


class Sub2UNet(nn.Module):
    """Same structure and parameter names as fn_unet.UNet."""

    def __init__(self, in_channels=1, out_channels=2, n_blocks=4, start_filters=32, activation="relu",
                 normalization="batch", conv_mode="same", dim=2, up_mode="transposed", use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        down, up = [], []
        for i in range(n_blocks):
            f_in = in_channels if i == 0 else f_out
            f_out = start_filters * (2 ** i)
            down.append(_UDownBlock(f_in, f_out, i < n_blocks - 1, activation, normalization, dim, conv_mode))
        for _ in range(n_blocks - 1):
            f_in, f_out = f_out, f_out // 2
            up.append(_UUpBlock(f_in, f_out, use_skip, activation, normalization, dim, conv_mode, up_mode))
        self.conv_final = _u_get_conv(f_out, out_channels, 1, 1, 0, True, dim)
        self.down_blocks = nn.ModuleList(down)
        self.up_blocks = nn.ModuleList(up)

    def encode(self, x):
        """Encoder only: returns the 4 'before_pooling' maps (= norm2 outputs)."""
        feats = []
        for block in self.down_blocks:
            x, before_pooling = block(x)
            feats.append(before_pooling)
        return feats

    def forward(self, x):
        enc = []
        for block in self.down_blocks:
            x, before_pooling = block(x)
            enc.append(before_pooling)
        for i, block in enumerate(self.up_blocks):
            x = block(enc[-(i + 2)] if self.use_skip else None, x)
        return torch.sigmoid(self.conv_final(x))


def load_sub2_unet(path: str, device: torch.device, use_skip: bool = True) -> Sub2UNet:
    return load_model_weights(Sub2UNet(**SUB2_UNET_ARCH, use_skip=use_skip), path, device)


@torch.inference_mode()
def extract_sub2_unet_features(model: Sub2UNet, cases: Sequence[dict], device: torch.device,
                               verbose: bool = True) -> Dict[int, Dict[str, np.ndarray]]:
    """Each window image (native size, no resizing) -> encoder -> spatial mean of each
    level's norm2 output, exactly as fn_feature_extraction.FeatureExtraction.

    Returns level_features[level][channel] = (N x dim) float32, rows in case order.
    """
    model.eval()
    out = {lv: {ch: [] for ch in SUB2_CHANNELS} for lv in SUB2_LEVEL_DIMS}
    n = len(cases)
    for i, c in enumerate(cases, start=1):
        for ch in SUB2_CHANNELS:
            arr = nib.load(c[SUB2_WINDOW_KEYS[ch]]).get_fdata(dtype=np.float32)
            if arr.ndim != 3:
                raise ValueError(f"Expected a 3D volume, got shape {arr.shape}: {c[SUB2_WINDOW_KEYS[ch]]}")
            x = torch.from_numpy(np.asarray(arr, dtype=np.float32)[None, None, ...]).to(device)
            for lv, fmap in enumerate(model.encode(x)):
                out[lv][ch].append(fmap.mean(dim=(2, 3, 4)).squeeze(0).float().cpu().numpy())
        if verbose:
            print(f"  Sub2 U-Net progress: {i}/{n}")
    return {lv: {ch: np.asarray(v, dtype=np.float32) for ch, v in d.items()} for lv, d in out.items()}


def load_sub2_survival(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sub2 frozen survival model not found: {path}")
    try:
        art = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        art = torch.load(path, map_location="cpu")
    if not isinstance(art, dict) or art.get("format") != SUB2_SURV_FORMAT:
        raise ValueError(f"{path} is not a frozen Deep-CT Sub2 survival file (expected '{SUB2_SURV_FORMAT}').")
    return art


def predict_sub2_survival(art: dict, fused: np.ndarray) -> Dict[str, np.ndarray]:
    """Fused (N x 504) -> {'OS': scores, 'PFS': scores}; higher = higher risk."""
    x = np.asarray(fused, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != SUB2_FUSED_DIM:
        raise ValueError(f"Expected N x {SUB2_FUSED_DIM} fused features, got {x.shape}")
    x = (x - art["scaler_mean"].numpy()) / art["scaler_scale"].numpy()
    x = (x - art["pca_mean"].numpy()) @ art["pca_components"].numpy().T
    out = {}
    for ep in ("OS", "PFS"):
        h = art["heads"][ep]
        z = (x - h["scaler_mean"].numpy()) / h["scaler_scale"].numpy()
        out[ep] = float(h["sign"]) * (z @ h["coef"].numpy() + float(h["intercept"]))
    return out


# =============================================================================
# 8. Late fusion — combine Sub1-4 outputs into final OS / PFS risk scores
# =============================================================================
# Input columns (per endpoint E = OS / PFS):
#   Subnet_1_E_risk = Sub1 E_log_risk_score     Subnet_2_E_risk = Sub2 E_log_risk_score
#   Subnet_3_E_risk = Sub3 E_log_risk_score     Subnet_4_E_risk = Sub4 E_subtype (S1 -> 1, S2 -> 0)
# Every subset of subnets has its own frozen fusion, fitted on Train by the lab's
# late-fusion script: Train StandardScaler, then one of the heads below.
#   direct : mean, max, rank_average (Train percentiles)         -> fusion_frozen.pth
#   linear : cox_ph, ridge_cox, survival_svm                      -> fusion_frozen.pth
#   trees  : rsf, gbs, extra_trees (only subsets the lab saved)   -> fusion_trees_frozen.joblib
# All heads: higher score = higher risk.

FUSION_FORMAT = "deepct-late-fusion-v1"
FUSION_DEFAULT_FILE = "fusion_frozen.pth"
FUSION_DEFAULT_TREE_FILE = "fusion_trees_frozen.joblib"
FUSION_DIRECT_HEADS = ("mean", "max", "rank_average")
FUSION_LINEAR_HEADS = ("cox_ph", "ridge_cox", "survival_svm")
FUSION_TREE_HEADS = ("rsf", "gbs", "extra_trees")
FUSION_ALL_HEADS = FUSION_DIRECT_HEADS + FUSION_LINEAR_HEADS + FUSION_TREE_HEADS


def fusion_key(endpoint: str, subnets: Sequence[int]) -> str:
    return f"{endpoint}|{'+'.join(str(k) for k in sorted(set(int(s) for s in subnets)))}"


def build_fusion_inputs(sub1: Optional[pd.DataFrame] = None, sub2: Optional[pd.DataFrame] = None,
                        sub3: Optional[pd.DataFrame] = None, sub4: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge subnet outputs by ID into Subnet_k_{OS,PFS}_risk columns (only patients present in all given)."""
    parts = []
    for k, df in ((1, sub1), (2, sub2), (3, sub3)):
        if df is not None:
            parts.append(pd.DataFrame({"ID": df["ID"].astype(str).str.strip(),
                                       f"Subnet_{k}_OS_risk": df["OS_log_risk_score"].values,
                                       f"Subnet_{k}_PFS_risk": df["PFS_log_risk_score"].values}))
    if sub4 is not None:
        parts.append(pd.DataFrame({"ID": sub4["ID"].astype(str).str.strip(),
                                   "Subnet_4_OS_risk": sub4["OS_subtype"].values,
                                   "Subnet_4_PFS_risk": sub4["PFS_subtype"].values}))
    if not parts:
        raise ValueError("No subnet outputs given.")
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="ID", how="inner", validate="one_to_one")
    n_max = max(len(p) for p in parts)
    if len(merged) < n_max:
        print(f"  Note: {n_max - len(merged)} patient(s) are missing from at least one subnet output and were dropped.")
    return merged


def _fusion_raw_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Same conversion as the lab script: S1/S2 -> 1/0, otherwise numeric."""
    out = []
    for c in cols:
        v = df[c]
        s = v.astype(str).str.strip().str.upper()
        if s.isin(["S1", "S2"]).all():
            out.append(s.map({"S1": 1.0, "S2": 0.0}).to_numpy(dtype=float))
        else:
            num = pd.to_numeric(v, errors="raise").to_numpy(dtype=float)
            if not np.isfinite(num).all():
                raise ValueError(f"Non-finite values in {c}")
            out.append(num)
    return np.column_stack(out)


def load_fusion(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Frozen fusion file not found: {path}")
    try:
        art = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        art = torch.load(path, map_location="cpu")
    if not isinstance(art, dict) or art.get("format") != FUSION_FORMAT:
        raise ValueError(f"{path} is not a frozen Deep-CT fusion file (expected '{FUSION_FORMAT}').")
    return art


def load_fusion_trees(path: str) -> dict:
    """Tree heads are scikit-survival objects saved with joblib (pickle): load only files from the lab,
    with the same scikit-survival / scikit-learn versions used to save them."""
    import joblib
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Frozen tree-head file not found: {path}")
    trees = joblib.load(path)
    if not isinstance(trees, dict) or trees.get("format") != FUSION_FORMAT:
        raise ValueError(f"{path} is not a frozen Deep-CT fusion tree file.")
    return trees


def available_fusions(art: dict, trees: Optional[dict] = None) -> pd.DataFrame:
    rows = []
    for key, e in art["entries"].items():
        heads = list(FUSION_DIRECT_HEADS) + list(e["linear"].keys())
        if trees:
            heads += [h for h in FUSION_TREE_HEADS if f"{key}|{h}" in trees["entries"]]
        rows.append({"endpoint": e["endpoint"], "subnets": "+".join(map(str, e["subnets"])), "heads": ", ".join(heads)})
    return pd.DataFrame(rows)


def predict_fusion(art: dict, inputs: pd.DataFrame, endpoint: str, subnets: Sequence[int], head: str,
                   trees: Optional[dict] = None) -> np.ndarray:
    key = fusion_key(endpoint, subnets)
    if key not in art["entries"]:
        raise KeyError(f"No frozen fusion for {endpoint} with subnets {sorted(set(subnets))}.")
    e = art["entries"][key]
    missing = [c for c in e["columns"] if c not in inputs.columns]
    if missing:
        raise ValueError(f"Fusion inputs are missing {missing}")
    x = (_fusion_raw_matrix(inputs, e["columns"]) - e["scaler_mean"].numpy()) / e["scaler_scale"].numpy()

    if head == "mean":
        return np.mean(x, axis=1)
    if head == "max":
        return np.max(x, axis=1)
    if head == "rank_average":
        sorted_train = e["train_sorted"].numpy()  # (k, n_train)
        n_train, k = sorted_train.shape[1], x.shape[1]
        score = np.zeros(x.shape[0], dtype=float)
        for j in range(k):
            score += ((np.searchsorted(sorted_train[j], x[:, j], side="right") - 0.5) / n_train) / k
        return score
    if head in FUSION_LINEAR_HEADS:
        if head not in e["linear"]:
            raise KeyError(f"Head '{head}' was not frozen for {key} (fit failed in the lab run).")
        h = e["linear"][head]
        return float(h["sign"]) * (x @ h["coef"].numpy() + float(h["intercept"]))
    if head in FUSION_TREE_HEADS:
        tkey = f"{key}|{head}"
        if trees is None or tkey not in trees["entries"]:
            raise KeyError(f"Tree head '{head}' was not saved for subnets {key.split('|')[1]} ({endpoint}).")
        return np.asarray(trees["entries"][tkey].predict(x), dtype=float).ravel()
    raise ValueError(f"Unknown fusion head '{head}'. Choose from: {', '.join(FUSION_ALL_HEADS)}")


# =============================================================================
# 9. Outcomes (optional), reference comparison, run log
# =============================================================================

DEFAULT_OUTCOME_COLUMNS = ("OS_time", "OS_event", "PFS_time", "PFS_event")


def read_table(path: str) -> pd.DataFrame:
    """Read .xlsx/.xls or .csv."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def attach_outcomes(scores: pd.DataFrame, outcomes_path: str, cohort: Optional[str] = None,
                    id_col: str = "ID", cohort_col: str = "Cohort",
                    outcome_cols: Sequence[str] = DEFAULT_OUTCOME_COLUMNS) -> pd.DataFrame:
    """Left-join outcome columns onto the scores by ID. Never drops a scored patient;
    patients without an outcome row simply get empty outcome columns.

    If `cohort` is given, the outcomes table is first filtered to that cohort
    (case-insensitive), as in the original multi-cohort Excel file.
    """
    if not os.path.isfile(outcomes_path):
        raise FileNotFoundError(f"Outcomes file not found: {outcomes_path}")
    out = read_table(outcomes_path)
    if id_col not in out.columns:
        if "file" in out.columns:          # e.g. 'CT_Main_0001.nii.gz' -> 'CT_Main_0001'
            out[id_col] = out["file"]
        else:
            raise ValueError(f"Outcomes file needs an '{id_col}' or 'file' column; "
                             f"found {list(out.columns)}")
    out[id_col] = out[id_col].astype(str).str.strip().map(strip_nifti_ext)
    if cohort is not None:
        if cohort_col not in out.columns:
            raise ValueError(f"--cohort given but outcomes file has no '{cohort_col}' column.")
        out = out[out[cohort_col].astype(str).str.strip().str.lower() == cohort.strip().lower()]
    if out[id_col].duplicated().any():
        raise ValueError("Duplicate IDs in the outcomes file (use --cohort if IDs repeat across cohorts).")

    keep = [c for c in outcome_cols if c in out.columns]
    merged = scores.merge(out[[id_col] + keep].rename(columns={id_col: "ID"}), on="ID", how="left")
    n_missing = merged[keep].isna().all(axis=1).sum() if keep else len(merged)
    if n_missing:
        print(f"  Note: {n_missing}/{len(merged)} scored patients have no outcome row (kept, outcomes left blank).")
    return merged


def compare_with_reference(scores: pd.DataFrame, reference_csv: str,
                           score_cols: Sequence[str] = ("OS_log_risk_score", "PFS_log_risk_score"),
                           atol: float = 1e-4) -> Dict[str, object]:
    """Compare new scores with an earlier output (e.g. JCO_repro_*.csv) by patient ID.

    The reference may have an 'ID' column or a 'file' column (file name).
    """
    ref = pd.read_csv(reference_csv)
    if "ID" not in ref.columns:
        if "file" not in ref.columns:
            raise ValueError("Reference CSV needs an 'ID' or 'file' column.")
        ref["ID"] = ref["file"].astype(str).map(strip_nifti_ext)
    ref["ID"] = ref["ID"].astype(str).str.strip()

    both = scores.merge(ref[["ID"] + list(score_cols)], on="ID", suffixes=("", "_ref"))
    report: Dict[str, object] = {"n_new": len(scores), "n_reference": len(ref), "n_matched": len(both)}
    ok = len(both) > 0
    for col in score_cols:
        diff = (both[col] - both[f"{col}_ref"]).abs()
        report[f"{col}_max_abs_diff"] = float(diff.max()) if len(diff) else float("nan")
        ok = ok and bool((diff <= atol).all())
    report["all_within_tolerance"] = ok
    report["tolerance"] = atol
    return report


def write_run_log(path: str, info: Dict[str, object]) -> None:
    """Write a JSON run log (inputs, weights + checksums, versions, results)."""
    payload = dict(info)
    payload["environment"] = environment_info()
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
