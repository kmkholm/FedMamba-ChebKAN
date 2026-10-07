# FedMamba-ChebKAN

Code for **“FedMamba-ChebKAN: Federated Mamba with Chebyshev-KAN for Heterogeneous IIoT Fault Detection”**
(Alsmadi, Tawfik, Fathi; under review at PLOS ONE).

A Mamba selective state-space encoder with a Chebyshev-KAN head, trained federatedly with **SpectralFedAvg**:
per-edge tail-energy truncation of Chebyshev coefficients, consensus-degree averaging on the shared basis, and an
optional order-scaled Gaussian mechanism with matched clipping. Baselines: Local-only, Centralised (pooled data),
FedAvg, FedProx, SCAFFOLD, FedBN, all on the same architecture, plus a 1-D CNN FedAvg reference.

## Layout
```
fedmamba_chebkan/
  data.py      leakage-free client partitions for CWRU, Paderborn, MIMII (fan), C-MAPSS, and the Dirichlet study
  models.py    Mamba encoder (pure-PyTorch chunked selective scan), Chebyshev-KAN layer, 1-D CNN, task heads
  fl.py        federated loop: FedAvg / FedProx / SCAFFOLD / FedBN / SpectralFedAvg, 8-bit uploads, DP (matched clipping)
  metrics.py   accuracy, macro-F1, macro-AUROC; MIMII AUROC/AUPRC/FAR@95%TPR per machine ID; C-MAPSS RMSE and PHM08 per engine
  run.py       one run -> results/<exp>/<name>.json (per-seed metrics, curves, payload, timing)
  grid.py      the full experiment grid of the paper
  analysis.py  tables, statistics (exact Wilcoxon, rank-biserial, Benjamini-Hochberg), figures, per-seed CSV
partitions/    client-to-recording/engine assignment manifests for every dataset
results/       per-seed result files used in the paper
```

## Data (public, not redistributed)
| Dataset | Source | Expected location (override with env vars) |
|---|---|---|
| CWRU 12 kHz drive-end + normal baseline | https://engineering.case.edu/bearingdatacenter | `$FMCK_DATA/CWRU/{12k_drive_end,normal_baseline}/*.mat` |
| Paderborn (KAt) | https://groups.uni-paderborn.de/kat/BearingDataCenter/ | `$FMCK_DATA/Paderborn/<bearing>/*.mat` |
| NASA C-MAPSS | https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip | `$FMCK_DATA/C-MAPSS/CMAPSSData/` |
| MIMII fan (−6/0/+6 dB) | https://zenodo.org/records/3384388 | `$FMCK_MIMII/{-6_dB,0_dB,6_dB}/fan/id_XX/` |

`python scripts/download_datasets.py small` (CWRU + C-MAPSS), `... paderborn` and `... mimii` download all four from these URLs (resumable; MIMII md5-checked).

## Partitions (no window crosses a split)
* **CWRU**: 12 clients = 4 motor loads × 3 sensor positions (DE, FE, BA); 4 classes (normal, ball, inner race, outer race @6:00; 7/14/21-mil faults pooled). Every recording is cut into contiguous 60/20/20 % segments (train/val/test) *before* windowing (L = 1024, 50 % overlap inside a segment). The normal-baseline files have no BA channel, so BA clients hold no normal windows.
* **Paderborn**: 12 clients = 2 operating settings (N15_M07_F10, N15_M01_F10) × 6 bearing triples (one healthy, one outer-race, one inner-race real-damage bearing from K001/K002, KA04/KA15, KI04/KI14). The 20 recordings of each bearing and setting are dealt to its three clients without overlap and split 4/1/1 by recording.
* **MIMII**: 12 clients = machine IDs 00/02/04/06 × SNR; split by clip; training and validation on normal clips only; 64 × 313 log-Mel features.
* **C-MAPSS**: 20 clients = 5 groups of training engines per FD subset; validation engines held out per client; test = the official test engines, scored on the last window against the official RUL (one prediction per engine).
* **Dirichlet study**: pooled CWRU drive-end windows re-partitioned over 12 clients with Dir(α) class proportions.

Normalisation statistics are fitted on each client's training windows only.

## Reproduce
```bash
pip install -r requirements.txt
python fedmamba_chebkan/data.py cwru paderborn cmapss mimii      # builds and caches partitions, writes manifests
python fedmamba_chebkan/run.py --dataset cwru --method spectral --seed 0
python fedmamba_chebkan/grid.py                                    # full grid of the paper
python fedmamba_chebkan/analysis.py                                # tables, statistics, figures
```
Tested with Python 3.13, PyTorch 2.12 (CUDA 12.6), one NVIDIA RTX A4000 Laptop GPU (8 GB).

## Privacy note
The order-scaled Gaussian perturbation carries an (ε, δ) guarantee only with **matched clipping** (each coefficient order
rescaled by 1/β_n before joint clipping, Poisson client sampling, fixed normaliser), as implemented in `fl.py` (`--dp_sigma`).
The guarantee assumes a trusted coordinator and does not cover the transmitted truncation degrees.

## License
MIT (see LICENSE).
