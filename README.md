# Personalizing Heart-Rate Prediction from Recent Workout History
### A User-Disjoint FitRec Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23334136.svg)](https://doi.org/10.5281/zenodo.23334136)

---

This repository contains the minimal reproducibility files for a study of
future-workout heart-rate prediction from recent personal workout history.
The primary benchmark uses a user-disjoint chronological test, one fixed target
workout per user, and user-level paired uncertainty.

## Repository Structure

```text
fitrec-personalization/
├── data/
│   └── README.md
├── scripts/
│   ├── fit_baseline_partial_pooling.py
│   ├── fit_lightgbm_personalization.py
│   ├── run_feature_matched_ablation.py
│   ├── run_adapted_fitrec_lstm.py
│   ├── analyze_fixed_cohort_history_depth.py
│   └── validate_release.py
├── results/
│   ├── primary_k5_summary.csv
│   ├── fixed_cohort_history_depth/
│   ├── feature_matched_ablation/
│   ├── adapted_fitrec_lstm/
│   ├── reference/
│   └── validation.json
├── tests/
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Main Results

At the prespecified K=5 history depth, LightGBM personalization achieved mean
user-level MAE 9.46 bpm versus 10.63 bpm for hierarchical-linear
personalization. The paired improvement was 1.16 bpm (95% user-bootstrap CI
0.75–1.61; n=107). Residual TCN achieved 9.41 bpm and did not reliably improve
on LightGBM.

The feature-matched ablation retained 107 identical users and targets. Extended
features improved hierarchical linear by 0.14 bpm (95% CI 0.06–0.23) and
LightGBM by 0.27 bpm (95% CI 0.01–0.55). The adapted FitRec-style LSTM achieved
MAE 9.58 bpm (95% CI 8.45–10.79); its paired differences from LightGBM and
residual TCN included zero.

These are prediction results for recorded consumer-platform heart rate, not a
clinical validation, causal effect, or training intervention.

---

## Reproducing the Analyses

1. Create an environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Supply the lawful preprocessed development and locked-test sequence
   directories described in `data/README.md`.

3. Run the feature-matched ablation:

   ```bash
   python scripts/run_feature_matched_ablation.py \
     --development-dir /path/to/preprocessed_devval_speedcap55_v1 \
     --test-dir /path/to/locked_test/preprocessed \
     --output-dir outputs/feature_matched_ablation \
     --bootstrap-replicates 10000
   ```

4. Run the adapted FitRec-style LSTM:

   ```bash
   python scripts/run_adapted_fitrec_lstm.py \
     --development-dir /path/to/preprocessed_devval_speedcap55_v1 \
     --test-dir /path/to/locked_test/preprocessed \
     --output-dir outputs/adapted_fitrec_lstm \
     --epochs 20 \
     --bootstrap-replicates 10000 \
     --reference-lightgbm results/reference/sequence_models_k5_user_metrics.csv \
     --reference-hierarchical results/reference/hierarchical_k5_user_metrics.csv \
     --reference-tcn results/reference/sequence_models_k5_user_metrics.csv
   ```

5. Validate the included release outputs:

   ```bash
   python scripts/validate_release.py
   pytest -q
   ```

The exact run parameters, package versions, seeds, and input-index SHA-256
values are recorded in the two `experiment_manifest.json` files.

---

## Data Availability

Raw Endomondo/FitRec records and workout-level HR/GPS sequences are not
redistributed. The repository includes only pseudonymous user-level error
tables, aggregate results, reference metrics, and analysis code. See
`data/README.md` for the expected local input format and official source link.

---

## Citation

> Salaj, Š. (2026). *Personalizing Heart-Rate Prediction from Recent Workout
> History: A User-Disjoint FitRec Benchmark* (v1.0.0) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.23334136

---

## License

Code is released under the [MIT License](LICENSE). Data remain subject to their
original terms and are not relicensed by this repository.
