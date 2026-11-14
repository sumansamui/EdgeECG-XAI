# ECG-CaRE

Code for *ECG-CaRE: An Interpretable and Edge-Deployable Framework for Arrhythmia
Classification Competitive with Deep Models* (A. Mukherjee, N. Biswas, S. Samui).

ECG-CaRE classifies heartbeats into the five AAMI EC57 classes (N, S, V, F, Q) from 24
causal morphological features with a Random Forest. It adds per-beat early stopping, a
five-objective Bayesian optimisation of the forest (qEHVI), a composite DART score for
picking one model from the Pareto front, and SHAP explanations.

## Repository layout

| Path | Content |
|---|---|
| `ECG_CaRE_reproduction.ipynb` | The whole pipeline, with comments and saved outputs |
| `data/` | Put `mitbih_23_morph_rr_features.xlsx` here |
| `results/` | Tables written by the notebook (CSV / JSON) |
| `figures/` | Figures written by the notebook (PNG) |
| `requirements.txt` | Python packages |

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook ECG_CaRE_reproduction.ipynb     # then Run All
```

or without opening it:

```bash
jupyter nbconvert --to notebook --execute ECG_CaRE_reproduction.ipynb \
        --output ECG_CaRE_reproduction.ipynb --ExecutePreprocessor.timeout=7200
```

The full run (20 Sobol + 30 qEHVI evaluations) took about 12 minutes on a 2-core CPU.
Set `FAST_MODE = True` in the configuration cell for a 3–4 minute smoke test.
No GPU is needed.

## Data

`mitbih_23_morph_rr_features.xlsx`: 109,398 beats from the MIT-BIH Arrhythmia Database,
one row per beat, with 23 morphological features, `RR_prev`, `RR_next`, the AAMI label
(`label`, 0–4) and the record id.

Split (AAMI EC57 inter-patient protocol, paced records 102/104/107/217 excluded):

| Set | Records | Beats | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|
| DS1 (train) | 22 | 50,977 | 45,824 | 943 | 3,788 | 414 | 8 |
| DS2 (test) | 22 | 49,668 | 44,218 | 1,836 | 3,219 | 388 | 7 |

The file holds features only, not waveforms.

## What the notebook does

| Section | Step | Output |
|---|---|---|
| 1 | Load data, DS1/DS2 split | `table1_class_counts.csv` |
| 2 | IQR Winsorization + z-score, fitted on DS1 only | — |
| 4 | Gini feature ranking on DS1 | `fig3_feature_importance.png`, `gini_ranking.csv` |
| 5 | RF_ALL, RF_CAUSAL, RF_TOP10, ECG-CaRE-ES | `table5_diagnostic_ds2.csv` |
| 6 | Five-objective qEHVI MOBO (BoTorch) | `mobo_all_trials.csv`, `fig2_mobo.png` |
| 7 | DART score, top-5 models, DS2 test | `pareto_dart.csv`, `table6_dart_top5.csv`, `best_params.json` |
| 8 | Early-stopping sweep on the top-1 model | `table7_early_stopping.csv` |
| 9 | Size, latency, RAM, FLOPs | `table8_deployability.csv` |
| 10 | SHAP: global importance, beeswarm, waterfalls | `fig4_*.png`, `fig5_*.png`, `shap_*.csv`, `table4_*.csv` |
| 11 | Manuscript numbers next to this run's numbers | `paper_vs_reproduced_table5.csv` |

## Method details as implemented

**Search space (15 dimensions).** `n_estimators`, `max_depth`, `min_samples_leaf`,
`min_samples_split`, `max_features`, `n_features`, four class weights `w_S, w_V, w_F, w_Q`,
five decision thresholds `tau_N, tau_S, tau_V, tau_F, tau_Q`.

- `n_features = k` selects the **top-k features by DS1 Gini rank**.
- The `tau_c` are **decision thresholds**: predicted class = `argmax_c p_c / tau_c`. They
  are applied inside the cross-validation, so they change the objectives. They are not
  the early-stopping threshold of Eq. 2.

**Objectives (all minimised).** `−BalAcc` and `FNR_abn` from pooled out-of-fold predictions
of a 3-fold GroupKFold on DS1 (groups = record id); latency, RAM and flash from a forest
refitted on all of DS1. DS2 is not used during the search or the DART selection.

**Surrogate and acquisition.** One GP per objective with an ARD Matérn-5/2 kernel;
qLogEHVI with `q = 1`; reference point set from the worst Sobol observation per objective.

**Early stopping (Eq. 1–2).** Trees are sorted by their individual balanced accuracy on a
15 % record subset of DS1. Inference stops once the running mean probability of some
class reaches `tau`, but not before `K_MIN = 5` trees. Without `K_MIN`, Eq. 2 stops at the
first tree whenever that tree ends in a pure leaf. `tau = 1.00` switches early stopping off.

**DART.** `DART = 0.50·D + 0.30·E + 0.20·Ed`, on the Pareto set, from DS1-CV objectives.
`D` = mean of normalised BalAcc and normalised `1 − FNR_abn`; `E` = mean of normalised
latency, RAM, flash; `Ed` = share of two limits met (latency ≤ 10 ms, flash ≤ 1024 KB).
All terms are min-max normalised within the Pareto set, so DART values are relative to
that set and cannot be compared across different runs.

**Hardware numbers.** Latency is a single-beat, tree-by-tree Python loop on a desktop
CPU. RAM is the `tracemalloc` peak of one `predict_proba` call. Flash is the pickle size of
the forest. FLOPs is the mean number of threshold comparisons per beat. These are
relative proxies measured the same way for every model; none is a microcontroller
measurement.

## Results of the saved run

Seed 42, 20 + 30 evaluations, test set DS2. Numbers below are the ones stored in
`results/` and in the notebook outputs.

**Diagnostic performance (Table V, RF rows)**

| Model | Acc % | BalAcc % | MCC | MaF1 | FNR-V % | FNR-S % | FNR-F % | FNR-Q % |
|---|---|---|---|---|---|---|---|---|
| RF_ALL (with `RR_next`) | 84.95 | 41.88 | 0.447 | 0.366 | 12.27 | 90.47 | 76.29 | 100 |
| RF_CAUSAL | 84.20 | 37.58 | 0.419 | 0.339 | 12.95 | 95.37 | 91.75 | 100 |
| RF_TOP10 | 81.07 | 36.33 | 0.375 | 0.334 | 14.14 | 93.90 | 94.85 | 100 |
| ECG-CaRE-ES (τ = 0.85) | 84.11 | 37.39 | 0.415 | 0.338 | 13.92 | 95.32 | 91.75 | 100 |
| ECG-CaRE+MOBO (DART #1) | 70.07 | 42.37 | 0.257 | 0.345 | 47.59 | 77.07 | 36.86 | 100 |

**DART top-5 on DS2 (Table VI)**

| Rank | DART | BalAcc % | MCC | Acc % | Lat ms | RAM KB | Flash KB | Trees | Depth | Features |
|---|---|---|---|---|---|---|---|---|---|---|
| #1 | 0.901 | 42.37 | 0.257 | 70.07 | 0.043 | 13.4 | 464 | 10 | 5 | 5 |
| #2 | 0.872 | 45.69 | 0.352 | 79.98 | 0.071 | 13.3 | 1442 | 10 | 25 | 5 |
| #3 | 0.828 | 46.26 | 0.373 | 80.75 | 0.060 | 13.4 | 464 | 10 | 5 | 5 |
| #4 | 0.797 | 29.94 | 0.168 | 36.81 | 0.050 | 13.3 | 868 | 10 | 25 | 24 |
| #5 | 0.775 | 34.84 | 0.200 | 51.58 | 0.058 | 13.3 | 463 | 10 | 5 | 15 |

**Early stopping on the top-1 model, T = 10 trees (Table VII)**

| τ | Trees % | N | V | S | Latency ms | ΔBalAcc pp |
|---|---|---|---|---|---|---|
| 1.00 | 100 | 100 | 100 | 100 | 0.039 ± 0.029 | 0.00 |
| 0.90 | 64.2 | 63.6 | 71.8 | 62.2 | 0.027 ± 0.013 | −0.01 |
| 0.85 | 63.4 | 62.7 | 71.4 | 62.2 | 0.026 ± 0.012 | −0.02 |
| 0.70 | 60.0 | 59.5 | 65.6 | 59.9 | 0.027 ± 0.013 | −0.04 |

On the 100-tree RF_CAUSAL forest, τ = 0.85 used 34 % of the trees for a 0.19 pp BalAcc loss.

**Deployability proxies (Table VIII, RF rows)**

| Model | Size KB | Nodes | Lat ms | RAM KB | FLOPs |
|---|---|---|---|---|---|
| RF_CAUSAL | 7814 | 72,670 | 0.750 | 14.2 | 1131 |
| RF_TOP10 | 8634 | 80,746 | 0.499 | 14.0 | 1176 |
| ECG-CaRE-ES | 7814 | 72,670 | 0.242 | 14.1 | 384 |
| ECG-CaRE+MOBO | 464 | 602 | 0.039 | 13.4 | 50 |

**MOBO.** Pareto set: 14 of 50 trials. Hypervolume gain: 143 in the Sobol phase, 267 in
the qEHVI phase. Best DS1-CV balanced accuracy over all trials: 39.4 %.

**SHAP (top-1 model, 487 DS2 beats).** Mean |SHAP| summed over classes: `RR_prev` 0.70,
`A_R` 0.35, `A_T` 0.18, `slope_RS` 0.14, `A_Q` 0.10. The top-1 model uses only these five
features.



## Citation

```bibtex
@INPROCEEDINGS{11622981,
  author={Mukherjee, Atin and Biswas, Nilanjan and Samui, Suman},
  booktitle={2026 International Conference on Signal Processing and Communications (SPCOM)}, 
  title={ECG-CaRE: An Interpretable and Edge-Deployable Framework for Arrhythmia Classification Competitive with Deep Models}, 
  year={2026},
  doi={10.1109/SPCOM71105.2026.11622981}}

```

Dataset: G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database,"
*IEEE Eng. Med. Biol. Mag.*, 20(3), 45–50, 2001.
