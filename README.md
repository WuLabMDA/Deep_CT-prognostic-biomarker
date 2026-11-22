# Deep-CT-prognostic-biomarker

**Deep-CT predicts benefit from ICI therapy in NSCLC patients**

> **This is an updated version of Deep-CT (v1.0, 2026).** Deep-CT was originally published in
> *The Lancet Digital Health* (2023; doi: [10.1016/S2589-7500(23)00082-1](https://doi.org/10.1016/S2589-7500(23)00082-1)).
> This release packages the published model so it can be run directly on new data. See
> [What's new in this version](#whats-new-in-this-version) below.

The goal is to develop a deep learning-powered prognostic framework applied to baseline CT images to automate the quantification of patients' risk of progression or death on ICI. Given the clinical observation that the metastatic patterns as manifested on CT can vary significantly across stage IV lung cancer patients, we focused on lung parenchymal and tumor lesions. To mitigate the uncertainty inevitable in one particular type of network model, we adopted an ensemble learning strategy to integrate fundamentally different but potentially complementary convolutional neural network (CNN) architectures to increase the model's generalizability. The ensemble framework consisted of four 3D-CNN models (figure below), including a supervised learning network (sub-network 1), two hybrid networks that merge supervised and unsupervised learning differently (sub-networks 2 and 3), and an unsupervised learning network (sub-network 4).

![Main_Fig](https://github.com/WuLabMDA/Deep_CT-prognostic-biomarker/assets/77283272/d79b7e36-e80e-4347-80ee-c92fd173d536)

Please find the work in detail here: https://authors.elsevier.com/sd/article/S2589-7500(23)00082-1

> **Pretrained weights:** [10.5281/zenodo.24447497](https://doi.org/10.5281/zenodo.24447497)
> **Example data:** [10.5281/zenodo.24447797](https://doi.org/10.5281/zenodo.24447797)

## What's new in this version

Compared with the original code released with the 2023 paper:

- **Ready-to-use pretrained model.** All trained components (the four sub-networks, the Sub-network 2 fusion
  parameters, the Sub-network 4 subtyping model and the final fusion head) are provided as weights on Zenodo.
  Nothing has to be retrained, and no training data or outcomes are needed to score new patients.
- **One pipeline, three scripts.** Preprocessing (P1), all four sub-networks in one run (P2) and the final Deep-CT
  score (P3), instead of running each sub-network separately and combining their outputs by hand.
- **Single configuration file.** All paths and options are set in `config.yaml`.
- **Choice of sub-networks and fusion head.** Users can combine any subset of the four sub-networks with one of
  several survival heads (see [Fusion heads](#fusion-heads)).
- **Outcomes are optional.** Scores are computed from images alone; outcome data, if given, are only attached to
  the output for evaluation.
- **Built-in checks and logs.** Input validation, preprocessing QC log, and a record of weight checksums and
  library versions for every run.

The original 2023 code (separate sub-network folders) is archived and available on request.

## Data preparation

Deep-CT requires baseline CT in three different contrasts as input: (1) lung window, (2) mediastinal window, and (3) the original/raw one. The figure below shows the overall data-preparation pipeline. The final input size is 128×128×128. In this package, these inputs are created by `P1_Preprocess_Data.py` from a CT and a lung mask.

![image](https://user-images.githubusercontent.com/77283272/214706511-6b2f1c80-cee7-4513-8ad7-2773e11263bc.png)

## Results

![Results](https://github.com/WuLabMDA/Deep_CT-prognostic-biomarker/assets/77283272/2b8a69b1-b736-4705-ba66-761c7efb0144)

## Hardware

Each sub-network was trained independently on an NVIDIA A100 (40 GB) GPU. Running the pretrained model (inference) does not need a large GPU: it runs on a standard GPU or, more slowly, on CPU.

---

## How to run Deep-CT

### Pipeline

```
CT + lung mask
   │
   ▼  P1_Preprocess_Data.py      → three 128³ CT windows per patient (original, mediastinal, lung)
   │
   ▼  P2_Extract_all_subnet.py   → OS_risk.csv, PFS_risk.csv
   │      Sub1  3D DenseNet121 survival network          → Sub1_risk
   │      Sub2  3D U-Net features + DCA fusion + SVM      → Sub2_risk
   │      Sub3  3D autoencoder survival network           → Sub3_risk
   │      Sub4  DEC subtyping of Sub3 features            → Sub4_subtype (S1 = higher risk, S2)
   │
   ▼  P3_Get_DeepCT_score.py     → DeepCT_OS_risk.csv, DeepCT_PFS_risk.csv
          late fusion of the chosen subnetworks with the chosen survival head
```

For all scores, a higher value means a higher risk.

### Repository layout

```
Deep_CT-prognostic-biomarker/
├── P1_Preprocess_Data.py
├── P2_Extract_all_subnet.py
├── P3_Get_DeepCT_score.py
├── deepct_utils.py          # model definitions and shared functions
├── config.yaml              # the only file you need to edit
├── requirements.txt
├── Hugo_data/               # ← example data, from Zenodo (see "Example data")
├── Weights/                 # ← pretrained weights, from Zenodo
└── DCA_parameters/          # ← pretrained weights, from Zenodo
```

### Installation

Requires Python [version] (tested with PyTorch [version]). A GPU is optional; everything also runs on CPU, more slowly.

```bash
# 1. Get the code
git clone https://github.com/WuLabMDA/Deep_CT-prognostic-biomarker.git
cd Deep_CT-prognostic-biomarker

# 2. Create an environment
conda create -n deepct python=[version]
conda activate deepct

# 3. Install PyTorch for your machine (CUDA or CPU): pick the command at
#    https://pytorch.org/get-started/locally/   e.g. CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install the remaining packages
pip install -r requirements.txt
```

### Pretrained weights

Download `DeepCT_weights_v1.0.zip` from Zenodo: **https://doi.org/10.5281/zenodo.24447497**

Unzip it so that `Weights/` and `DCA_parameters/` sit inside the repository folder, next to `config.yaml`.
The download includes `SHA256SUMS.txt` to check that the files are intact.

### Example data

Download `Hugo_data.zip` from Zenodo: **https://doi.org/10.5281/zenodo.24447797**

Unzip it into the repository folder, so that `Hugo_data/` sits next to `config.yaml`:

```
Hugo_data/
├── CT/                  CT_<ID>.nii.gz      (CT in Hounsfield units)
├── lung_mask/           LM_<ID>.nii.gz      (binary lung mask, same grid as the CT)
└── Hugo_outcome.xlsx    PatientID, OS_time, OS_event   (optional)
```

These [n] cases are from [dataset name] ([source DOI / link]), redistributed under [license].
Please cite the original dataset if you use them.

### Quick start

After installing, unzip the weights and the example data into the repository folder, then:

```bash
python P1_Preprocess_Data.py       # Hugo_data/CT + lung_mask → Hugo_data/Hugo_3D
python P2_Extract_all_subnet.py    # → Results/OS_risk.csv, Results/PFS_risk.csv
python P3_Get_DeepCT_score.py      # → Results/DeepCT_OS_risk.csv, Results/DeepCT_PFS_risk.csv
```

### Using your own data

1. **Prepare inputs.** Each patient needs a CT in Hounsfield units and a binary lung mask with the same size
   and orientation, both as NIfTI (`.nii.gz`). By default CTs are named `CT_<ID>.nii.gz` and masks
   `LM_<ID>.nii.gz` (prefixes can be changed in P1).
2. **Preprocess** with `P1_Preprocess_Data.py`. Cases that fail checks (e.g. CT not in HU, empty mask) are listed
   in `preprocessing_log.csv` and skipped.
3. **Edit `config.yaml`**: set `paths.data_root` to the preprocessed folder and, optionally,
   `outcomes.file` to your outcome table.
4. Run **P2**, then **P3**.

#### Outcome file (optional)

Excel or CSV with an ID column (`ID`, `PatientID`, `PID` or `file`) and any of
`OS_time`, `OS_event`, `PFS_time`, `PFS_event`. IDs must equal the image file names without `.nii.gz`.
Outcomes are only copied next to the scores; they are never used to compute them.

### Configuration (`config.yaml`)

Relative paths are read from the folder that contains `config.yaml`.

| Section | Key | Meaning |
|---|---|---|
| `paths` | `data_root` | Folder with the three preprocessed window folders |
| | `weights_dir`, `dca_dir` | Pretrained files (from Zenodo) |
| | `output_dir` | Where results are written (created if needed) |
| `inputs` | `original_dir`, `mediastinal_dir`, `lung_dir` | Window folder names inside `data_root` |
| `outcomes` | `file` | Optional outcome table (`null` = scores only) |
| `runtime` | `device` | `auto`, `cpu` or `cuda` |
| `fusion` | `subnets` | Any combination of `1, 2, 3, 4` |
| | `heads` | One or more fusion heads (see below) |

#### Fusion heads

| Type | Heads | Output column (OS example) |
|---|---|---|
| Direct | `mean`, `max`, `rank_average` | `DeepCT_OS_mean_head`, … |
| Linear | `cox_ph`, `ridge_cox`, `survival_svm` | `DeepCT_OS_coxph_head`, … |
| Tree | `rsf`, `gbs`, `extra_trees` (only subnet combination 1+2+3) | `DeepCT_OS_rsf_head`, … |

Run `python P3_Get_DeepCT_score.py --list` to see every available subnet/head combination.
Options can also be given on the command line, e.g. `--subnets 1 2 --heads cox_ph mean`.

### Outputs

| File | Columns |
|---|---|
| `OS_risk.csv` / `PFS_risk.csv` | `ID`, `Sub1_risk`, `Sub2_risk`, `Sub3_risk`, `Sub4_subtype`, [`OS_time`, `OS_event`] |
| `DeepCT_OS_risk.csv` / `DeepCT_PFS_risk.csv` | `ID`, `DeepCT_<endpoint>_<head>_head` …, [outcome columns] |
| `*_run_info.json` | Settings, weight checksums and library versions of the run |

Scores are relative risk scores, not calibrated probabilities.

### Validation vs. refitting

- **Deep-CT validation (default):** all subnetworks and the fusion head are used exactly as published.
  Report this as external validation of Deep-CT.
- **Refitting on your own population:** you may keep the four subnetworks fixed and refit only the fusion
  head on *your own training split*, evaluating on held-out data. Report this as "Deep-CT subnetworks with a
  locally refitted fusion", not as Deep-CT.
- Re-deriving a risk-group cutoff on your data while keeping the published model counts as validation with
  local recalibration.

### Notes

- Tree fusion heads are saved with joblib and need the scikit-learn / scikit-survival versions in
  `requirements.txt`. Only load weight files from the official Zenodo record.
- Deep-CT is a research tool and is not intended for clinical decision-making.

## Citation

```bibtex
@article{saad2023deepct,
  title   = {Predicting benefit from immune checkpoint inhibitors in patients with non-small-cell lung cancer
             by CT-based ensemble deep learning: a retrospective study},
  author  = {Saad, Maliazurina B. and others},
  journal = {The Lancet Digital Health},
  year    = {2023},
  doi     = {10.1016/S2589-7500(23)00082-1}
}

```

## License

Code: BSD-3-Clause (see `LICENSE`). Weights and example data: see the license on each Zenodo record.


