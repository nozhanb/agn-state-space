# Plot Index — AGN SBI Project
# NGC 1365 XMM-Newton Inference + Simulation Studies
#
# All paths are relative to:
#   /Users/home/Documents/Claude/project/agn_sbi/
#
# To find a plot: search this file by name or keyword.
# Last updated: 2026-06-03
#
# PROVENANCE RULE: every plot and table in the paper must have an entry here.
# For each entry record: output file, generating script, and the exact NPZ/data
# file(s) read at generation time. This file is the single source of truth for
# reproducing any number in the paper.

---

## RUN 1 — Flat Γ prior  (T=21, 5 ks bins, full covering)
Inference script : code_backup/ngc_1365/count_inference_hmc_ngc1365_xmm_5ks.py
Plot script      : code_backup/ngc_1365/make_diagnostic_plots.py
Results          : code_backup/ngc_1365/results/
Posterior NPZ    : code_backup/ngc_1365/results/inference_ngc1365_xmm_5ks_T21.npz
Key result       : Γ → 1.00 (boundary), NH ~ 10^19.8–10^21.3  (~1.4 dex below literature)

| File | Path | Description |
|------|------|-------------|
| 01_light_curve.pdf | code_backup/ngc_1365/results/plots/ | NGC 1365 net count light curve, 5 ks bins |
| 02_posterior_predictive.pdf | code_backup/ngc_1365/results/plots/ | Posterior predictive check vs observed counts |
| 03_nh_timeseries.pdf | code_backup/ngc_1365/results/plots/ | NH(t) posterior median + 90% CI |
| 04_phi_timeseries.pdf | code_backup/ngc_1365/results/plots/ | φ(t) normalisation posterior |
| 05_nh_vs_countrate.pdf | code_backup/ngc_1365/results/plots/ | NH vs count rate scatter |
| 06_marginal_posteriors.pdf | code_backup/ngc_1365/results/plots/ | Marginal posteriors: Γ, NH mean, τ, σ, φ mean |
| 07_spectral_snapshots.pdf | code_backup/ngc_1365/results/plots/ | Count spectra at 4 selected time bins |
| 08_trace_plots.pdf | code_backup/ngc_1365/results/plots/ | MCMC trace plots for all global parameters |
| 09_joint_nh_phi.pdf | code_backup/ngc_1365/results/plots/ | Joint panel: light curve + NH(t) + φ(t) |

---

## RUN 2 — Gaussian Γ prior N(1.75, 0.20²)  (T=21, 5 ks bins, full covering)
Inference script : code_backup/ngc_1365/count_inference_hmc_ngc1365_xmm_5ks_gamma_prior.py
Plot script      : code_backup/ngc_1365/make_diagnostic_plots_gamma_prior.py
Results          : code_backup/ngc_1365/results_gamma_prior/
Posterior NPZ    : code_backup/ngc_1365/results_gamma_prior/inference_ngc1365_xmm_5ks_T21_gamma_prior.npz
Key result       : Γ → 0.835 (further from 1.75 than flat prior); likelihood overwhelms prior by ~4.5σ

| File | Path | Description |
|------|------|-------------|
| 01_light_curve.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | Light curve (same data as Run 1) |
| 02_posterior_predictive.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | Posterior predictive check |
| 03_nh_timeseries.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | NH(t) with Braito 2014 reference band |
| 04_phi_timeseries.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | φ(t) normalisation |
| 05_nh_vs_countrate.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | NH vs count rate |
| 06_marginal_posteriors.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | Marginals including prior overlay on Γ panel |
| 07_spectral_snapshots.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | Count spectra at 4 bins |
| 08_trace_plots.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | MCMC traces |
| 09_joint_nh_phi.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | Joint panel |
| 10_comparison_flat_vs_gamma_prior.pdf | code_backup/ngc_1365/results_gamma_prior/plots/ | Side-by-side: flat prior vs Gaussian prior (Γ, NH, PP) |

---

## RUN 3 — Fixed Γ = 1.75  (T=21, 5 ks bins, full covering)
Inference script : code_backup/ngc_1365/count_inference_hmc_ngc1365_xmm_5ks_fixed_gamma.py
Plot script      : (none yet — diagnostic plots not generated for this run)
Results          : code_backup/ngc_1365/results_fixed_gamma/
Posterior NPZ    : code_backup/ngc_1365/results_fixed_gamma/inference_ngc1365_xmm_5ks_T21_fixed_gamma.npz
Key result       : NH recovers to 10^22–10^23 cm^-2 — confirms Γ–NH degeneracy was root cause

---

## RUN 4 — Partial covering, f free  (T=21, 5 ks bins)
Inference script : code_backup/ngc_1365/count_inference_hmc_ngc1365_xmm_5ks_partial_covering.py
Plot script      : code_backup/ngc_1365/make_diagnostic_plots_partial_covering.py
Results          : code_backup/ngc_1365/results_partial_covering/
Posterior NPZ    : code_backup/ngc_1365/results_partial_covering/inference_ngc1365_xmm_5ks_T21_partial_covering.npz
Key result       : Γ = 1.50, f_cover = 0.87, NH ~ 10^21.9–10^23.6; physically consistent

| File | Path | Description |
|------|------|-------------|
| 01_light_curve.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | Light curve, 5 ks bins |
| 02_posterior_predictive.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | Posterior predictive check |
| 03_nh_timeseries.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | NH(t) — now in Braito 2014 range |
| 04_phi_timeseries.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | φ(t) normalisation |
| 05_nh_vs_countrate.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | NH vs count rate |
| 06_marginal_posteriors.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | Marginals: Γ, f_cover, NH, τ, σ, φ (8 panels) |
| 07_spectral_snapshots.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | Count spectra at 4 bins |
| 08_trace_plots.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | MCMC traces incl. f_cover |
| 09_joint_nh_phi.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | Joint panel |
| 10_three_way_comparison.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | Three-way: flat / fixed Γ / partial covering (Γ, NH, PP) |
| 11_fcover_gamma_joint.pdf | code_backup/ngc_1365/results_partial_covering/plots/ | f_cover marginal + Γ–f_cover joint scatter |

---

## RUN 5 — Partial covering, f free  (T=53, 2 ks bins)
Inference script : code_backup/ngc_1365/count_inference_hmc_ngc1365_xmm_2ks_T53_partial_covering.py
Plot script      : code_backup/ngc_1365/make_diagnostic_plots_partial_covering_T53.py
Results          : code_backup/ngc_1365/results_partial_covering_2ks_T53/
Posterior NPZ    : code_backup/ngc_1365/results_partial_covering_2ks_T53/inference_ngc1365_xmm_2ks_T53_partial_covering.npz
Key result       : Γ = 1.50, f_cover = 0.874, NH ~ 10^21.9–10^23.6; τ_NH = 60 bins, T/τ = 0.89

| File | Path | Description |
|------|------|-------------|
| 01_light_curve.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | Light curve, 2 ks bins |
| 02_posterior_predictive.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | Posterior predictive check |
| 03_nh_timeseries.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | NH(t) — higher time resolution |
| 04_phi_timeseries.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | φ(t) normalisation |
| 05_nh_vs_countrate.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | NH vs count rate |
| 06_marginal_posteriors.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | Marginals: Γ, f_cover, NH, τ, σ, φ (8 panels) |
| 07_spectral_snapshots.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | Count spectra at min/25th/75th/max flux bins |
| 08_trace_plots.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | MCMC traces |
| 09_joint_nh_phi.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | Joint panel |
| 10_T21_vs_T53_comparison.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | T=21 vs T=53: Γ posterior, NH(t), τ_NH posterior |
| 11_fcover_gamma_joint.pdf | code_backup/ngc_1365/results_partial_covering_2ks_T53/plots/ | f_cover marginal + Γ–f_cover joint scatter |

---

## STANDALONE PLOTS — NH(t) Power Spectral Density
Script   : plots/nh_psd_t53_partial_covering/make_nh_psd_t53.py
Data NPZ : code_backup/ngc_1365/results_partial_covering_2ks_T53/inference_ngc1365_xmm_2ks_T53_partial_covering.npz
README   : plots/nh_psd_t53_partial_covering/README.md

| File | Path | Description |
|------|------|-------------|
| nh_psd_t53_partial_covering.pdf | plots/nh_psd_t53_partial_covering/ | NH(t) time series + PSD: Vaughan bins, power-law fit β=1.50±0.10, AR(1) Lorentzian overlay |

---

## STANDALONE PLOTS — Simulation Studies (earlier work)
All in: plots/

| File | Path | Description |
|------|------|-------------|
| nh_extreme_comparison_Nh21_Tau2_Phi-3.pdf | plots/extreme_nh_comparison_3idx/ | Simulated NH(t) trajectory comparison at extreme parameters |
| nh_extreme_psd_linear_Nh21_Tau2_Phi-3.pdf | plots/extreme_nh_psd_linear/ | PSD of simulated NH(t) in linear scale |
| nh_extreme_sample_diagnostic_Nh21_Tau2_Phi-3.pdf | plots/extreme_nh_sample_diagnostic_idx382/ | Sample diagnostic for extreme NH simulation |
| psd_beta_recovery_Nh21_Tau2_Phi-3_methodB_paper.pdf | plots/psd_beta_recovery_tau2/ | PSD slope β recovery from simulation (Vaughan binning, Method B) |
| tau_posterior_Nh22_Tau38_phi-4_paper.pdf | plots/tau_posterior_tau38/ | τ posterior recovery from simulation (NH=22, τ=38 bins) |

---

## DOCUMENT
| File | Path |
|------|------|
| feasibility_assessment.tex | doc/feasibility_assessment.tex |
| feasibility_assessment.pdf | doc/feasibility_assessment.pdf |

---

## CANONICAL INFERENCE RUNS — NGC 1365, 4-observation campaign (2012–2013)
## Model: partial covering + frozen full-covering absorber (Rivers et al. 2015)
## Bin size: 2 ks. Prior: Γ ~ Normal(1.75, 0.20).
## These NPZ files are the primary data source for all results-section plots and tables.

### Obs 1 — ObsID 0692840201 (2012-07-25)
Inference script name     : inference_ngc1365_obs1.py
Inference script location : code_backup/ngc_1365/inference_ngc1365_obs1.py
Posterior NPZ name        : inference_ngc1365_xmm_2ks_obs1_partial_covering_fc.npz
Posterior NPZ location    : code_backup/ngc_1365/results_partial_covering_2ks_obs1_T59_v2/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc.npz
Summary CSV location      : code_backup/ngc_1365/results_partial_covering_2ks_obs1_T59_v2/summary_ngc1365_xmm_2ks_obs1_partial_covering_fc.csv
Notes                     : T=59 bins, NH_FC=1.0e22 cm^-2, soft wall present but never activated (max log NH=23.76)

### Obs 2 — ObsID 0692840301 (2012-12-24)
Inference script name     : inference_ngc1365_obs2.py
Inference script location : code_backup/ngc_1365/inference_ngc1365_obs2.py
Posterior NPZ name        : inference_ngc1365_xmm_2ks_obs2_partial_covering_fc.npz
Posterior NPZ location    : code_backup/ngc_1365/results_partial_covering_2ks_obs2_T60/inference_ngc1365_xmm_2ks_obs2_partial_covering_fc.npz
Summary CSV location      : code_backup/ngc_1365/results_partial_covering_2ks_obs2_T60/summary_ngc1365_xmm_2ks_obs2_partial_covering_fc.csv
Notes                     : T=60 bins, NH_FC=1.4e22 cm^-2, no soft wall

### Obs 3 — ObsID 0692840401 (2013-01-23)
Inference script name     : inference_ngc1365_obs3.py
Inference script location : code_backup/ngc_1365/inference_ngc1365_obs3.py
Posterior NPZ name        : inference_ngc1365_xmm_2ks_obs3_partial_covering_fc.npz
Posterior NPZ location    : code_backup/ngc_1365/results_partial_covering_2ks_obs3_T50_v2/inference_ngc1365_xmm_2ks_obs3_partial_covering_fc.npz
Summary CSV location      : code_backup/ngc_1365/results_partial_covering_2ks_obs3_T50_v2/summary_ngc1365_xmm_2ks_obs3_partial_covering_fc.csv
Notes                     : T=50 bins, NH_FC=1.1e22 cm^-2, soft wall active (required to suppress degenerate N_H→inf mode)

### Obs 4 — ObsID 0692840501 (2013-02-12)
Inference script name     : inference_ngc1365_obs4.py
Inference script location : code_backup/ngc_1365/inference_ngc1365_obs4.py
Posterior NPZ name        : inference_ngc1365_xmm_2ks_obs4_partial_covering_fc.npz
Posterior NPZ location    : code_backup/ngc_1365/results_partial_covering_2ks_obs4_T57/inference_ngc1365_xmm_2ks_obs4_partial_covering_fc.npz
Summary CSV location      : code_backup/ngc_1365/results_partial_covering_2ks_obs4_T57/summary_ngc1365_xmm_2ks_obs4_partial_covering_fc.csv
Notes                     : T=57 bins, NH_FC=1.0e22 cm^-2, no soft wall

---

## SENSITIVITY PLOTS — Gamma prior sensitivity, Obs 1

### PLOT: sensitivity_gamma_prior_comparison_all5
Name             : sensitivity_gamma_prior_comparison_all5.pdf / sensitivity_gamma_prior_comparison_all5.png
Output location  : code_backup/ngc_1365/sensitivity_gamma_prior_obs1_plots/sensitivity_gamma_prior_comparison_all5.pdf
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_plots/sensitivity_gamma_prior_comparison_all5.png
Script name      : NOT TRACKED — generating script was not saved to the repository.
                   Closest related script (3-run version only):
                   plot_sensitivity_comparison.py
Script location  : code_backup/ngc_1365/sensitivity_gamma_prior_obs1_178_sigma030/plot_sensitivity_comparison.py
Data file 1      : inference_ngc1365_xmm_2ks_obs1_partial_covering_fc.npz
                   code_backup/ngc_1365/results_partial_covering_2ks_obs1_T59_v2/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc.npz
                   (Run: Main — Normal(1.75, 0.20))
Data file 2      : inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma178.npz
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_175_vs_178/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma178.npz
                   (Run: Centre — Normal(1.78, 0.20))
Data file 3      : inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma178_sigma010.npz
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_178_sigma010/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma178_sigma010.npz
                   (Run: Narrow — Normal(1.78, 0.10))
Data file 4      : inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma178_sigma030.npz
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_178_sigma030/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma178_sigma030.npz
                   (Run: Broad — Normal(1.78, 0.30))
Data file 5      : inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_uniform/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
                   (Run: Uniform(1.0, 3.0))
Purpose          : Justification that the choice of Gamma prior (Normal vs Uniform, and
                   choice of sigma = 0.10 / 0.20 / 0.30) has no meaningful effect on either
                   the Gamma or the N_H posterior for Obs 1. All five posteriors are
                   indistinguishable. Supports the paper claim that N_H is robust to the
                   prior on Gamma.
WARNING          : The generating script is missing from the repository. To reproduce this
                   plot, extend plot_sensitivity_comparison.py (3-run version) to include
                   the Centre and Uniform runs listed as data files 2 and 5 above.

---

## RESULTS-SECTION PLOTS (paper figures)
## Format for every plot entry:
##   Name             — filename of the output
##   Output location  — full relative path to the output file
##   Script name      — filename of the script that generated it
##   Script location  — full relative path to that script
##   Data file(s)     — name and full relative path of every input data file read by the script

### PLOT: ngc1365_nh_timeseries
Name             : ngc1365_nh_timeseries.pdf / ngc1365_nh_timeseries.png
Output location  : plots/ngc1365_results_section/ngc1365_nh_timeseries.pdf
                   plots/ngc1365_results_section/ngc1365_nh_timeseries.png
Script name      : plot_nh_timeseries.py
Script location  : plots/ngc1365_results_section/plot_nh_timeseries.py
Data file (Obs 1): inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_uniform/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
Data file (Obs 2): inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs2/results/inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.npz
Data file (Obs 3): inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall_5k3k.npz
                   code_backup/ngc_1365/uniform_obs3_withwall_5k3k/results/inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall_5k3k.npz
Data file (Obs 4): inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs4/results/inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.npz
Prior            : Uniform(1.0, 3.0) on Gamma for all four observations
Note (Obs 3)     : Uses 5000 warmup + 3000 samples re-run for proper convergence.

### PLOT: ngc1365_obs3_results_4panel
Name             : ngc1365_obs3_results_4panel.pdf / ngc1365_obs3_results_4panel.png
Output location  : plots/ngc1365_results_section/ngc1365_obs3_results_4panel.pdf
                   plots/ngc1365_results_section/ngc1365_obs3_results_4panel.png
Script name      : plot_obs3_results_4panel.py
Script location  : plots/ngc1365_results_section/plot_obs3_results_4panel.py
Data file (Obs 3): inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall_5k3k.npz
                   code_backup/ngc_1365/uniform_obs3_withwall_5k3k/results/inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall_5k3k.npz
Note (Obs 3)     : Uses 5000 warmup + 3000 samples re-run for proper convergence.
Contents         : 4-panel figure (shared time axis, units: ks from observation start):
                     Panel 1 — N_H(t): median + 68% CI + 90% CI
                     Panel 2 — K(t) flux normalisation: median + 68% CI + 90% CI
                     Panel 3 — Counts per bin: observed (points) + predicted median + predicted 90% CI
                     Panel 4 — Residuals: (N_pred_median - N_obs) / sqrt(N_obs)
Notes            : Posterior predictive quantiles computed over 2000 samples from
                   posterior_predictive_total (channel-integrated, per 2 ks bin).
                   Residual denominator is sqrt(N_obs) — Poisson uncertainty on the data,
                   not the model. Prior: Uniform(1.0, 3.0) on Gamma, soft wall on N_H.

### PLOT: ngc1365_nh_all_obs_comparison
Name             : ngc1365_nh_all_obs_comparison.pdf / ngc1365_nh_all_obs_comparison.png
Output location  : plots/ngc1365_results_section/ngc1365_nh_all_obs_comparison.pdf
                   plots/ngc1365_results_section/ngc1365_nh_all_obs_comparison.png
Script name      : make_nh_comparison_all_obs.py
Script location  : plots/ngc1365_literature_comparison/make_nh_comparison_all_obs.py
Data file (Obs 1): inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_uniform/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
Data file (Obs 2): inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs2/results/inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.npz
Data file (Obs 3): inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall_5k3k.npz
                   code_backup/ngc_1365/uniform_obs3_withwall_5k3k/results/inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall_5k3k.npz
Data file (Obs 4): inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs4/results/inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.npz
Prior            : Uniform(1.0, 3.0) on Gamma for all four observations
Note (Obs 3)     : Uses 5000 warmup + 3000 samples re-run for proper convergence.

### PLOT: ngc1365_prior_sensitivity_uniform_vs_normal178
Name             : ngc1365_prior_sensitivity_uniform_vs_normal178.pdf / ngc1365_prior_sensitivity_uniform_vs_normal178.png
Output location  : plots/ngc1365_results_section/ngc1365_prior_sensitivity_uniform_vs_normal178.pdf
                   plots/ngc1365_results_section/ngc1365_prior_sensitivity_uniform_vs_normal178.png
Script name      : plot_prior_sensitivity_uniform_vs_normal178.py
Script location  : plots/ngc1365_results_section/plot_prior_sensitivity_uniform_vs_normal178.py
Data file (Obs 1, Uniform) : inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
                             code_backup/ngc_1365/sensitivity_gamma_prior_obs1_uniform/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
Data file (Obs 1, Normal)  : inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma178.npz
                             code_backup/ngc_1365/sensitivity_gamma_prior_obs1_175_vs_178/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma178.npz
Data file (Obs 2, Uniform) : inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.npz
                             code_backup/ngc_1365/sensitivity_gamma_uniform_obs2/results/inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.npz
Data file (Obs 2, Normal)  : inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma178.npz
                             code_backup/ngc_1365/results_partial_covering_2ks_obs2_T60_gamma178/inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma178.npz
Data file (Obs 3, Uniform) : inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall.npz
                             code_backup/ngc_1365/sensitivity_gamma_uniform_obs3_withwall/results/inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall.npz
Data file (Obs 3, Normal)  : inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma178.npz
                             code_backup/ngc_1365/results_partial_covering_2ks_obs3_T50_v2_gamma178/inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma178.npz
Data file (Obs 4, Uniform) : inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.npz
                             code_backup/ngc_1365/sensitivity_gamma_uniform_obs4/results/inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.npz
Data file (Obs 4, Normal)  : inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma178.npz
                             code_backup/ngc_1365/results_partial_covering_2ks_obs4_T57_gamma178/inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma178.npz
Contents         : 2×4 figure. Top row: Γ posterior KDE for each observation (Uniform=blue,
                   Normal(1.78,0.20)=red). Bottom row: μ_NH posterior KDE for each observation.
Purpose          : Demonstrates that the choice of Gamma prior has negligible effect on the
                   inferred N_H for Obs 1, 2, 4. Obs 3 shows a larger Γ shift (Δ = −0.218:
                   Uniform median 2.339 vs Normal(1.78) median 2.122) reflecting that the
                   Obs 3 likelihood is less constraining on Γ (soft spectrum + soft wall);
                   μ_NH remains consistent (Δ = +0.085 dex) across both priors for Obs 3.
Median shifts    : Obs 1: ΔΓ = −0.023, Δμ_NH = −0.004
                   Obs 2: ΔΓ = −0.023, Δμ_NH = −0.008
                   Obs 3: ΔΓ = −0.218, Δμ_NH = +0.085   ← Γ prior matters for Obs 3
                   Obs 4: ΔΓ = −0.017, Δμ_NH = +0.014
NPZ keys used    : gamma_values → Γ posterior samples; nH_shift → μ_NH posterior samples

### PLOT: appendix_a (Fig. A.1 + Fig. A.2)
Name             : app_A_light_curves.pdf / .png  (Fig. A.1)
                   app_A_fig_beta_4_v2.pdf / .png  (Fig. A.2)
Output location  : plots/appendix_a/app_A_light_curves.pdf
                   plots/appendix_a/app_A_fig_beta_4_v2.pdf
Script name      : plot_light_curves.py (Fig. A.1), plot_psd_beta4_recovery.py (Fig. A.2)
Script location  : plots/appendix_a/plot_light_curves.py
                   plots/appendix_a/plot_psd_beta4_recovery.py
Data file(s)     : Appendix_A_..._beta_{1.7,3.0,4.0}_and_1000_time_step_and_7_unit_shift_v2scan.npz
                   plots/appendix_a/data/ (copied from
                   code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/T1000_v2_scan_data/)
Contents         : Fig. A.1 -- three raw (pre-AR(1), pre-Poisson) colorednoise light
                   curves, beta=1.7/3/4, T=1000, mean-subtracted, one panel.
                   Fig. A.2 -- single-draw PSD recovery example for beta=4: unbinned
                   periodogram (black), 9-bin average with error bars (orange),
                   power-law+constant fit (red dashed), true input colorednoise PSD
                   (blue), log-log axes.
Purpose          : Regenerated 2026-07-23 -- both figures previously referenced by the
                   paper were not resolving in tex/ (missing files), and separately
                   needed to reflect the AR(1) initialization bugfix
                   (ar1_hmc_v2.py replacing simple_HMC.py, see
                   CORRECTION_LOG_2026-07-21.md) rather than older/stale data.
Key numbers      : Fig. A.2 single-draw (draw index 0) fitted beta = 3.82, close to
                   Table A.1's current published median for beta=4 (3.83) -- a
                   reassuring but incidental consistency check between this new
                   figure and the table (built from a different, separate script and,
                   for the table's own provenance, possibly different underlying
                   data -- see psd_beta_recovery_per_draw_binned_fit.py's docstring).
Note             : Fig. A.2's binning/fit methodology is copied verbatim from
                   psd_beta_recovery_per_draw_binned_fit.py for consistency, including
                   its known cosmetic quirk of very large error bars on 1-2 binned
                   points (linearized log-space error propagation, exaggerated when
                   within-bin periodogram scatter is large relative to the bin mean).

### TABLE: tab:appendix_params (Table A.1) -- regenerated 2026-07-23
Name             : tab:appendix_params, "Parameter estimates and true values" (Appendix A)
Table location   : tex/july_16_2026.tex, \label{tab:appendix_params}
Script name      : generate_table_A1.py (beta=3.0, beta=4.0)
                   generate_table_A1_beta1.7_longchain.py (beta=1.7, long chain + thin=50)
Script location  : plots/appendix_a/generate_table_A1.py
                   plots/appendix_a/generate_table_A1_beta1.7_longchain.py
Data file(s)     : plots/appendix_a/data/Appendix_A_..._beta_{1.7,3.0,4.0}_..._v2scan.npz
                   plots/appendix_a/data/Appendix_A_..._beta_1.7_..._v2scan_longchain.npz
Purpose          : The previously published table was traced to data generated by the
                   confirmed-buggy simple_HMC.py AR(1) model (see
                   code_backup/Claude_Beta_Recovery/CORRECTION_LOG_2026-07-23.md for full
                   provenance trace). Regenerated from the corrected ar1_hmc_v2.py model's
                   T=1000 posteriors -- same data as plots/appendix_a's Fig. A.1/A.2.
Key numbers      : beta_1: 1.59/1.59/1.59 (true 1.70); beta_2: 2.44/2.48/2.51 (true 3.00);
                   beta_3: 3.73/3.86/3.92 (true 4.00) -- none of the three now bracket the
                   true value (old table had 2 of 3 bracketing); required updating two
                   narrative sentences elsewhere in the paper that stated the old claim,
                   see CORRECTION_LOG_2026-07-23.md.

### PLOT: beta3_4_snr_trend
Name             : beta3_4_snr_trend.pdf / beta3_4_snr_trend.png
Output location  : plots/beta3_4_snr_trend/beta3_4_snr_trend.pdf
                   plots/beta3_4_snr_trend/beta3_4_snr_trend.png
Script name      : plot_beta3_4_snr_trend.py
Script location  : plots/beta3_4_snr_trend/plot_beta3_4_snr_trend.py
Data file(s)     : beta_3.0_shift_{1.0,4.0,7.0}.npz, beta_4.0_shift_{1.0,4.0,7.0}.npz
                   plots/beta3_4_snr_trend/data/ (copied from
                   code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/beta3_4_snr_sweep_v2_data/)
Contents         : Single-panel figure. tau posterior mean +/- std vs shift_term (SNR
                   proxy), for beta=3 and beta=4, with the prior's upper edge (80) marked.
Purpose          : Appendix A colorednoise validation (app:recovery) follow-up -- tests
                   whether higher signal-to-noise relaxes the tau-pinning-at-prior-edge
                   behaviour seen for beta>2. Finding: it does not -- tau moves CLOSER to
                   the edge as SNR increases, supporting a structural (short-memory AR(1)
                   vs long-memory colorednoise) explanation rather than an information
                   deficit. shift=10,13 excluded -- could not be reliably sampled (frozen-
                   chain pathology, see SCRIPT_LOG.md in the reproduction_2026-07-20 dir).
Key numbers      : distance from edge shrinks 4.32 -> 1.55 -> 0.92 (beta=3) and
                   2.64 -> 0.74 -> 0.28 (beta=4) across shift=1,4,7.

---

## TABLES (paper)
## Format for every table entry:
##   Name             — LaTeX label and human-readable description
##   Table location   — tex file and \label{} where the table appears
##   Data file(s)     — name and full relative path of every input data file the values come from

### TABLE: tab:convergence — HMC convergence diagnostics, all 4 observations
Name           : tab:convergence (R-hat and ESS for 8 global scalar parameters)
Table location : tex/  (paste ngc1365_convergence_table.tex into paper)
Script name    : make_convergence_table.py
Script location: plots/ngc1365_results_section/make_convergence_table.py
Output (LaTeX) : plots/ngc1365_results_section/ngc1365_convergence_table.tex
Output (text)  : plots/ngc1365_results_section/ngc1365_convergence_table.txt
Data file (Obs 1): summary_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.csv
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_uniform/results/
Data file (Obs 2): summary_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.csv
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs2/results/
Data file (Obs 3): summary_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall.csv
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs3_withwall/results/
Data file (Obs 4): summary_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.csv
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs4/results/
Parameters     : Gamma, f_cov, mu_NH, ln_tau_NH, sigma_NH, mu_K, ln_tau_K, sigma_K
Flags          : R-hat > 1.01 → bold; ESS < 200 → italic
WARNING        : Obs 3 shows severe non-convergence — mu_K R-hat=1.208 ESS=7,
                 mu_NH R-hat=1.093 ESS=37. See DEFERRED.md.

### TABLE: tab:ngc1365_results — Posterior scalar parameters, all 4 observations
Name           : tab:ngc1365_results (posterior medians and 90% CI)
Table location : tex/may_12_2026.tex, \label{tab:ngc1365_results}
Data file (Obs 1): inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_prior_obs1_uniform/results/inference_ngc1365_xmm_2ks_obs1_partial_covering_fc_gamma_uniform.npz
Data file (Obs 2): inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs2/results/inference_ngc1365_xmm_2ks_obs2_partial_covering_fc_gamma_uniform.npz
Data file (Obs 3): inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall.npz
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs3_withwall/results/inference_ngc1365_xmm_2ks_obs3_partial_covering_fc_gamma_uniform_withwall.npz
Data file (Obs 4): inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.npz
                   code_backup/ngc_1365/sensitivity_gamma_uniform_obs4/results/inference_ngc1365_xmm_2ks_obs4_partial_covering_fc_gamma_uniform.npz
NPZ keys used  : gamma_values → Gamma; f_cover → f_cov; nH_shift → mu_NH;
                 nh_tau → tau_NH (days) = exp(nh_tau)*2000/86400; nh_sigma → sigma_NH
CI level       : 90% (p5–p95)
Prior          : Uniform(1.0, 3.0) on Gamma for all four observations

### TABLE: tab:ngc1365 — Observation log
Name           : tab:ngc1365 (ObsIDs, dates, raw duration, GTI livetime, bin count, net counts)
Table location : tex/may_11_2026.tex, \label{tab:ngc1365}
Data file      : values from XMM-Newton data reduction; not derived from posterior NPZ.
                 GTI filtering threshold: RATE <= 0.40 ct/s in the 10–12 keV band.

### TABLE: tab:ngc1365_priors — Prior distributions
Name           : tab:ngc1365_priors (prior distributions for all inferred parameters)
Table location : tex/may_11_2026.tex, \label{tab:ngc1365_priors}
Data file      : no NPZ — values read directly from inference scripts
                 code_backup/ngc_1365/inference_ngc1365_obs1.py
                 code_backup/ngc_1365/inference_ngc1365_obs2.py
                 code_backup/ngc_1365/inference_ngc1365_obs3.py
                 code_backup/ngc_1365/inference_ngc1365_obs4.py

---

## TIER 6 — Unabsorbed 3–10 keV Luminosity Time Series
Plot script      : code_backup/ngc_1365/tier6/make_luminosity_timeseries.py
Output directory : code_backup/ngc_1365/tier6/luminosity_timeseries/
Saved arrays     : code_backup/ngc_1365/tier6/luminosity_timeseries/luminosity_all_obs.npz

Model            : tbabs × (pexmon + ztbabs × zpcfabs × zpowerlw)  — Tier 6
Γ prior          : Uniform(1.0, 3.0)
Cross-sections   : Verner+1996 + Wilms+2000
Quantity plotted : Unabsorbed 3–10 keV luminosity L(t) [erg/s]
                   L(t) = 4π d_L² × K(t) × (1+z)^{-Γ} × keV_erg
                          × [10^{2−Γ} − 3^{2−Γ}] / (2−Γ)
                   where K(t) = 10^{φ(t)} is the zpowerlw normalisation
Distance         : d_L = 17.95 Mpc  (Silbermann et al. 1999, ApJ 515, 1 — Cepheid)
Credible bands   : 68% (p16–p84) and 90% (p5–p95) shown

Input NPZ files  :
  Obs 1: tier6/results/inference_ngc1365_xmm_2ks_obs1_pexmon_tbabs_verner_uniform_gamma.npz
  Obs 2: tier6/obs2/results/inference_ngc1365_xmm_2ks_obs2_pexmon_tbabs_verner_uniform_gamma.npz
  Obs 3: tier6/obs3/results/inference_ngc1365_xmm_2ks_obs3_pexmon_tbabs_verner_uniform_gamma_withwall.npz
         *** withwall version — valid Obs 3 result (no-wall run has nh_sigma pinned at prior) ***
  Obs 4: tier6/obs4/results/inference_ngc1365_xmm_2ks_obs4_pexmon_tbabs_verner_uniform_gamma.npz

| File | Description |
|------|-------------|
| lum_timeseries_obs1.{pdf,png} | Obs 1 (Jul 2012) unabsorbed L(t) + 68/90% CI |
| lum_timeseries_obs2.{pdf,png} | Obs 2 (Dec 2012) unabsorbed L(t) + 68/90% CI |
| lum_timeseries_obs3.{pdf,png} | Obs 3 (Jan 2013) unabsorbed L(t) + 68/90% CI — withwall |
| lum_timeseries_obs4.{pdf,png} | Obs 4 (Jul 2013) unabsorbed L(t) + 68/90% CI |
| lum_timeseries_all4.{pdf,png} | 4-panel combined figure, all observations |
| luminosity_all_obs.npz        | Luminosity percentiles (p5/p16/p50/p84/p95) for all obs |

---

## TIER 6 — Power Spectral Density of NH(t) and L(t)
Plot script      : code_backup/ngc_1365/tier6/make_psd_timeseries.py
Output directory : code_backup/ngc_1365/tier6/psd_timeseries/
Saved arrays     : code_backup/ngc_1365/tier6/psd_timeseries/psd_all_obs.npz

Model            : tbabs × (pexmon + ztbabs × zpcfabs × zpowerlw)  — Tier 6
Γ prior          : Uniform(1.0, 3.0)
Cross-sections   : Verner+1996 + Wilms+2000
Quantities       : PSD of log10(NH(t)) and PSD of log10(L_{3-10 keV}(t))
                   Both in units of dex² Hz^{-1}

PSD normalisation (one-sided, Parseval):
  P(f_k) = (2 Δt / N) × |FFT(x − x̄)_k|²    k = 1, …, N//2
  Δt = 2000 s (calendar bin spacing);  DC component excluded

Theoretical AR(1) PSD overlay (consistency check):
  P_theory(f) = σ² Δt (1−α²) / [1 + α² − 2α cos(2π f Δt)]
  α = exp(−1/τ_bins),  τ_bins = exp(nh_log_tau)  or  exp(phi_log_tau)
  Computed per posterior sample, summarised at 16/50/84 percentiles

Credible bands   : 68% (p16–p84) and 90% (p5–p95) for empirical PSD;
                   68% (p16–p84) for theoretical PSD

Input NPZ files  :
  Obs 1: tier6/results/inference_ngc1365_xmm_2ks_obs1_pexmon_tbabs_verner_uniform_gamma.npz
  Obs 2: tier6/obs2/results/inference_ngc1365_xmm_2ks_obs2_pexmon_tbabs_verner_uniform_gamma.npz
  Obs 3: tier6/obs3/results/inference_ngc1365_xmm_2ks_obs3_pexmon_tbabs_verner_uniform_gamma_withwall.npz
         *** withwall version — valid Obs 3 result ***
  Obs 4: tier6/obs4/results/inference_ngc1365_xmm_2ks_obs4_pexmon_tbabs_verner_uniform_gamma.npz

Bin counts per observation:
  Obs 1: T=59, Obs 2: T=53, Obs 3: T=50, Obs 4: T=56

| File | Description |
|------|-------------|
| psd_obs1.{pdf,png} | Obs 1 (Jul 2012) — 2-panel PSD: log10(NH) left, log10(L) right |
| psd_obs2.{pdf,png} | Obs 2 (Dec 2012) — 2-panel PSD |
| psd_obs3.{pdf,png} | Obs 3 (Jan 2013) — 2-panel PSD — withwall result |
| psd_obs4.{pdf,png} | Obs 4 (Jul 2013) — 2-panel PSD |
| psd_all4.{pdf,png} | 4×2 panel combined figure — all observations, both quantities |
| psd_all_obs.npz    | PSD percentiles (p5/p16/p50/p84/p95) and theory (p16/p50/p84) for all obs |

---

## TIER 6 — Combined Time Series + PSD (4×2 per quantity)
Plot script      : code_backup/ngc_1365/tier6/make_combined_timeseries_psd.py
Output directory : code_backup/ngc_1365/tier6/combined/

Layout           : 4 rows (one per observation) × 2 columns
                   Left column  = time series with 68% and 90% posterior CI
                   Right column = posterior PSD (68/90% CI) + AR(1) theory overlay (68% CI, gray dashed)

Input NPZ files  :
  Luminosity time series : tier6/luminosity_timeseries/luminosity_all_obs.npz
  PSD arrays             : tier6/psd_timeseries/psd_all_obs.npz
  NH time series         : loaded directly from inference NPZs (nh_y percentiles computed here)
    Obs 1: tier6/results/inference_ngc1365_xmm_2ks_obs1_pexmon_tbabs_verner_uniform_gamma.npz
    Obs 2: tier6/obs2/results/inference_ngc1365_xmm_2ks_obs2_pexmon_tbabs_verner_uniform_gamma.npz
    Obs 3: tier6/obs3/results/inference_ngc1365_xmm_2ks_obs3_pexmon_tbabs_verner_uniform_gamma_withwall.npz
    Obs 4: tier6/obs4/results/inference_ngc1365_xmm_2ks_obs4_pexmon_tbabs_verner_uniform_gamma.npz

NH time series units  : 10^22 cm^-2 (log scale); PSD units: dex^2 Hz^-1
L  time series units  : erg/s (log scale);         PSD units: dex^2 Hz^-1

Posterior NH_22 median ranges:
  Obs 1: 37.2–64.2 ×10²² cm⁻²  |  Obs 2: 22.1–37.5 ×10²² cm⁻²
  Obs 3:  5.3–31.2 ×10²² cm⁻²  |  Obs 4: 16.4–45.3 ×10²² cm⁻²

| File | Description |
|------|-------------|
| combined_lum_all4.{pdf,png} | 4×2: L(t) time series (left) + PSD of log10(L) (right), all 4 obs |
| combined_nh_all4.{pdf,png}  | 4×2: NH(t) time series (left) + PSD of log10(NH) (right), all 4 obs |

---

## PAPER FIGURES — NH + Luminosity combined (4×2, two figures)
Plot script      : plots/ngc1365_results_section/plot_nh_lum_combined.py
Output directory : plots/ngc1365_tier6/
Last updated     : 2026-07-08

### PLOT: ngc1365_nh_lum_timeseries
Name             : ngc1365_nh_lum_timeseries.pdf / ngc1365_nh_lum_timeseries.png
Output location  : plots/ngc1365_tier6/ngc1365_nh_lum_timeseries.pdf
                   plots/ngc1365_tier6/ngc1365_nh_lum_timeseries.png
Script name      : plot_nh_lum_combined.py
Script location  : plots/ngc1365_results_section/plot_nh_lum_combined.py
Layout           : 4 rows (one per observation) × 2 columns
                   Left  column = NH(t) posterior in log10(NH / cm^-2) with 68% and 90% CI
                   Right column = unabsorbed 3–10 keV L(t) [erg/s, log scale] with 68% and 90% CI
Input NPZ (lum)  : code_backup/ngc_1365/tier6/luminosity_timeseries/luminosity_all_obs.npz
Input NPZ (NH, Obs 1): code_backup/ngc_1365/tier6/results/inference_ngc1365_xmm_2ks_obs1_pexmon_tbabs_verner_uniform_gamma.npz
Input NPZ (NH, Obs 2): code_backup/ngc_1365/tier6/obs2/results/inference_ngc1365_xmm_2ks_obs2_pexmon_tbabs_verner_uniform_gamma.npz
Input NPZ (NH, Obs 3): code_backup/ngc_1365/tier6/obs3/results/inference_ngc1365_xmm_2ks_obs3_pexmon_tbabs_verner_uniform_gamma_withwall.npz
Input NPZ (NH, Obs 4): code_backup/ngc_1365/tier6/obs4/results/inference_ngc1365_xmm_2ks_obs4_pexmon_tbabs_verner_uniform_gamma.npz
Distance         : d_L = 18.28 Mpc (Silbermann et al. 1999, μ = 31.31 mag)
Obs 4 label      : Feb 2013 (corrected from Jul 2013)

### PLOT: ngc1365_nh_lum_psd
Name             : ngc1365_nh_lum_psd.pdf / ngc1365_nh_lum_psd.png
Output location  : plots/ngc1365_tier6/ngc1365_nh_lum_psd.pdf
                   plots/ngc1365_tier6/ngc1365_nh_lum_psd.png
Script name      : plot_nh_lum_combined.py
Script location  : plots/ngc1365_results_section/plot_nh_lum_combined.py
Layout           : 4 rows (one per observation) × 2 columns
                   Left  column = posterior PSD of log10(NH(t))  [dex² Hz^-1]
                   Right column = posterior PSD of log10(L(t))   [dex² Hz^-1]
                   Both columns: 68% + 90% empirical CI, AR(1) theory overlay (68% CI, gray dashed),
                   break frequency marker (dashed black vertical line + gray 68% CI band)
Input NPZ (PSD)  : code_backup/ngc_1365/tier6/psd_timeseries/psd_all_obs.npz
Input NPZ (tau/sigma, per obs): same four inference NPZs as ngc1365_nh_lum_timeseries above
                   nh_tau + nh_sigma used for NH AR(1) theory PSD
                   phi_tau + phi_sigma used for L AR(1) theory PSD
Break frequencies (posterior median):
  Obs 1: f_break(NH) = 0.0038 mHz  |  f_break(L) = 0.0213 mHz
  Obs 2: f_break(NH) = 0.0257 mHz  |  f_break(L) = 0.0201 mHz
  Obs 3: f_break(NH) = 0.0016 mHz  |  f_break(L) = 0.0044 mHz
  Obs 4: f_break(NH) = 0.0018 mHz  |  f_break(L) = 0.0083 mHz
Obs 4 label      : Feb 2013 (corrected from Jul 2013)
