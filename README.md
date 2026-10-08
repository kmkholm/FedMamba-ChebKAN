# FedMamba-ChebKAN

Code for **“FedMamba-ChebKAN: Federated Mamba with Chebyshev-KAN for Heterogeneous IIoT Fault Detection”**
(Alsmadi, Tawfik, Fathi; under review at PLOS ONE).

A Mamba selective state-space encoder with a Chebyshev-KAN head, trained federatedly with **SpectralFedAvg**:
per-edge tail-energy truncation of Chebyshev coefficients and consensus-degree averaging on the shared basis;
SpectralFedAvg-BN additionally keeps BatchNorm on the clients. Baselines: Local-only, pooled-data reference, FedAvg,
FedProx, SCAFFOLD and FedBN, all on the same architecture, partitions, seeds and training budget.

## Layout
```
fedmamba_chebkan/
  data.py      leakage-free client partitions for CWRU, Paderborn, MIMII (fan) and C-MAPSS
  models.py    Mamba encoder (pure-PyTorch chunked selective scan), Chebyshev-KAN layer, task heads
  fl.py        federated loop: FedAvg / FedProx / SCAFFOLD / FedBN / SpectralFedAvg / SpectralFedAvg-BN, client-side DP-SGD
  metrics.py   accuracy, macro-F1, macro-AUROC; MIMII AUROC/AUPRC/FAR@95%TPR per machine ID; C-MAPSS RMSE and PHM08 per engine
  run.py       one run -> results/<exp>/<name>.json (per-seed metrics, curves, payload, timing)
  grid.py      the full experiment grid of the paper
  analysis.py  tables, statistics (exact Wilcoxon, rank-biserial, Benjamini-Hochberg), figures, per-seed CSV
partitions/    client-to-recording/engine assignment manifests for every dataset
results/main/   per-seed result file of every run reported in the paper (+ trained SpectralFedAvg models, seed 0)
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

## Results
`results/main/` holds one JSON file per run (4 datasets × 8 methods × 8 seeds = 256 runs): final metrics, the selected
round, validation/test curves, measured upload per client and round, and run time. `analysis.py` recomputes every
table, the Wilcoxon/Benjamini–Hochberg statistics and Figures 2–7 of the paper from these files.

## Privacy study (example-level DP-SGD on every client)
```bash
python fedmamba_chebkan/pretrain_public.py      # encoder + first KAN layer on PUBLIC Paderborn data -> results/public/
P=results/public/paderborn_pretrained.pt
python fedmamba_chebkan/run.py --dataset cwru --method spectral --seed 1 --norm group --win_norm --select_last        --full_train --init $P --exp dp_full --lr 5e-3 --batch 4096 --local_epochs 4 --dpsgd_sigma 8 --tag _full_dpsgd8
```
Every client clips per-example gradients (C = 1) and adds Gaussian noise before anything leaves the device, so the guarantee
holds against the coordinator and covers the uploaded coefficients and truncation degrees (post-processing). GroupNorm replaces
BatchNorm, windows are standardised individually, and the last-round model is reported. `run.py` stores the per-client
(epsilon, delta = 1e-5) from the realised number of noisy steps (Google `dp_accounting` RDP accountant).
`--full_train` gives every client all non-overlapping windows of its training segments (639-924 instead of 195-260).
`results/dp_full/` holds the runs of Table 8 (sigma 4 / 8 / 15 and the no-noise reference, seeds 1-3; `dp_full_grid.py`);
`results/dp/` holds the same study with the 195-260 sampled windows of the main comparison (lr 3e-3), quoted in the text.

## License
MIT (see LICENSE).
