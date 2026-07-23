# Markdown File Index — agn_sbi Project
# Last updated: 2026-07-20
#
# Quick reference for every .md file in this project tree.
# Grouped by topic. Paths are relative to project root:
#   /Users/home/Documents/Claude/project/agn_sbi/

---

## 1. Project Root — Top-Level Documents

| File | What it is |
|------|-----------|
| `README.md` | Project overview: state-space modelling of AGN X-ray variability; framework description, NH(t) recovery, what the repo contains |
| `PIPELINE.md` | Full reproduction pipeline — exact sequence of steps to regenerate all data, inference results, and figures from scratch |
| `PLOT_INDEX.md` | Master index of every figure in the project; paths, what each plot shows, which script generated it |
| `DISCUSSION_CORE_MESSAGES.md` | Living document of core scientific messages for the Discussion section (entries [1]–[14]); used to draft Section 5 of the paper |
| `DEFERRED.md` | Decisions deferred for discussion with Johannes — modelling choices not yet resolved, not to be acted on without explicit approval |
| `MD_FILE_INDEX.md` | **This file** — index of all .md files in the project |

---

## 2. Consultation

| File | What it is |
|------|-----------|
| `consultation/questions_for_johannes.md` | Eight open questions to raise with Johannes before finalising Tier 4 and the new spectral model (pexmon, NH_FC, Γ approximation, cross-section systematics, etc.) |

---

## 3. NGC 1365 — Code Backup Root

Located under `code_backup/ngc_1365/`

| File | What it is |
|------|-----------|
| `NGC1365_reduction_notes.md` | Full EPIC-pn data reduction notes for all four NGC 1365 observations — target coordinates, filtering steps, pile-up checks, GTI selection, every decision made |
| `xmm_data_reduction_guide.md` | Step-by-step XMM-Newton reduction guide (ObsID 0692840401 as worked example) — from raw ODF to final (T, N_chan) count array |
| `inference_assumptions_justifications.md` | Living document of every modelling assumption and its justification; intended as the basis for the Methods section |
| `comparison_results.md` | NGC 1365 parameter comparison table: this work vs. literature (Risaliti+2005, Rivers+2015, Brenneman+2013, etc.) |
| `flags_to_revisit.md` | Items noted during analysis that are not blocking but should be revisited before paper submission |

---

## 4. NGC 1365 — Tier Directory READMEs

Each tier README contains: a directory-specific header, the full four-tier layout diagram, file listings, and the key parameter table.

| File | What it is |
|------|-----------|
| `tier1/README.md` | Tier 1 guide — Normal(1.75, 0.20) Γ prior; f_cover posteriors, posterior predictive, Γ comparison figures |
| `tier2/README.md` | Tier 2 guide — Uniform(1.0, 3.0) Γ prior; **main paper results** for NH timeseries, τ/σ posteriors, convergence table |
| `tier3/README.md` | Tier 3 guide — Normal(1.78) variants; prior sensitivity comparison figure only |
| `tier4/README.md` | Tier 4 guide — new model: tbabs × (pexmon + ztbabs × zpcfabs × zpow); Verner+Wilms cross-sections; Normal(1.97, 0.15) Γ prior; Obs 2 pilot run, awaiting run approval |
| `obsolete/README.md` | Obsolete scripts guide — early 5ks-bin runs, V1 scripts, joint inference attempts; not used in paper |

---

## 5. NGC 1365 — Diagnostic Plot Interpretations

One file per observation, inside each Tier 1 results directory. Each records the
model setup, runtime, and a written interpretation of the convergence diagnostics.

| File | What it is |
|------|-----------|
| `results_partial_covering_2ks_obs1_T59_v2/diagnostic_plots/interpretation.md` | Obs 1 diagnostics — T=59 bins, runtime 117.97 min, Tier 1 run |
| `results_partial_covering_2ks_obs2_T60/diagnostic_plots/interpretation.md` | Obs 2 diagnostics — T=60 bins, Tier 1 run |
| `results_partial_covering_2ks_obs3_T50/diagnostic_plots/interpretation.md` | Obs 3 diagnostics — T=50 bins, initial Tier 1 run |
| `results_partial_covering_2ks_obs3_T50_v2/diagnostic_plots/interpretation.md` | Obs 3 diagnostics — T=50 bins, v2 Tier 1 run (revised) |
| `results_partial_covering_2ks_obs4_T57/diagnostic_plots/interpretation.md` | Obs 4 diagnostics — T=57 bins, Tier 1 run |

---

## 6. 1ES 1927+654 — Code Backup

Located under `code_backup/1Es_1927/`

| File | What it is |
|------|-----------|
| `dec_2018/reduction/reduction_notes.md` | EPIC-pn reduction notes for Dec 2018 observation (ObsID 0831790301) — corona-disappearance recovery epoch |
| `dec_2018/reduction/region_selection_notes.md` | Source/background region selection decisions for Dec 2018 |
| `may_2019/reduction/reduction_notes.md` | EPIC-pn reduction notes for May 2019 observation |
| `may_2019/reduction/region_selection_notes.md` | Source/background region selection decisions for May 2019 |
| `reduction/reduction_notes.md` | EPIC-pn reduction notes for the primary 1ES 1927 observation |
| `reduction/region_selection_notes.md` | Source/background region selection decisions for primary observation |

---

## 7. Plot Directory READMEs

Located under `plots/`

| File | What it is |
|------|-----------|
| `extreme_nh_comparison_3idx/README.md` | Investigation of whether the β≈4 artifact seen at index 382 is reproducible across other extreme-β simulation samples |
| `extreme_nh_psd_linear/README.md` | NH power spectral density plots on linear scale for extreme-NH samples |
| `extreme_nh_sample_diagnostic_idx382/README.md` | Detailed diagnostic for simulation sample index 382 showing anomalous low-frequency PSD anchor |
| `ngc1365_gamma_fcover_comparison/README.md` | Comparison plots of Γ and f_cover posteriors across all four NGC 1365 observations |
| `nh_psd_t53_partial_covering/README.md` | NH PSD plots from the T=53 partial-covering run |
| `psd_beta_recovery_tau2/README.md` | PSD β recovery study for simulations with τ=2 |
| `tau_posterior_tau38/README.md` | τ posterior recovery study for simulations with τ=38 |

---

## 8. Documentation — LaTeX Documents

Located under `doc/`

| File | What it is |
|------|-----------|
| `doc/feasibility_assessment.tex` | Main theoretical document: information budget, AR(1) effective sample size, Fisher information, prior choices (incl. log-uniform vs α_AR), Γ–NH degeneracy, signal-to-noise regimes. Compiles to `feasibility_assessment.pdf` (39 pages). |

---

## 9a. Beta Recovery / Appendix A — Code Backup

Located under `code_backup/Claude_Beta_Recovery/`

| File | What it is |
|------|-----------|
| `CORRECTION_LOG_2026-07-11.md` | Bug found & fixed: cached β label (weighted fit) vs. plotted fit line (unweighted refit) were computed inconsistently for `fig:nh_extreme_comparison`. Fix: `nh_phi_psd_beta_recovery_methodB_v2.py` / `nh_extreme_comparison_v2.py` now save and reuse the same per-sample `popt`. |
| `CORRECTION_LOG_2026-07-20.md` | Investigated whether the same class of bug (single fixed `curve_fit` initial guess for β, no multi-start) that required a real fix in Appendix A also affects the main-text N_H PSD recovery (Nh21_Tau2_Phi-3, incl. the β≈4.48 examples, samples 382/219/944). Checked directly against the cached data: confirmed present in the code, but empirically negligible here (mean/median/named-sample β unchanged to <0.01-0.27); no change needed to Figs. 5/6 or sec. 3.4.2. |
| `Appendix_A/reproduction_2026-07-20/psd_beta_recovery_per_draw_binned_fit.py` | Documented, re-runnable script for the corrected Appendix A β-recovery pipeline (per-draw + 9-bin fit, multi-start optimizer, percentile-based Table A.1 numbers) |
| `Appendix_A/reproduction_2026-07-20/GROUP_B_BETA_1.7_HANDOFF.md` | Self-contained starting point for the not-yet-started "Group B" SNR-sweep experiment (Johannes's items 12-14): exact existing data files, scripts, HMC config, measured per-run timing, and open decisions to resolve with Johannes before running new simulations |
| `CORRECTION_LOG_2026-07-21.md` | Bug found, not yet fixed: `tau_prior`/`mean_prior` swapped at the `run_inference_on_slice(...)` call in `colored_noise_light_curve_generation_with_one_beta_and_multiple_time_lengths.py` (function signature has `mean_prior` before `tau_prior`; the call passes them in the opposite order). Confirmed via saved `tau_param`/`mean_param` values across 4 files of the β=1.7 T=500 10-repeat dataset. Affects that dataset (used for Group B); does not affect the standalone base files behind the current Table A.1. |
| `Appendix_A/reproduction_2026-07-20/ar1_hmc_v2.py` | Corrected AR(1)+HMC model replacing `simple_HMC.py` (fixes wrong stationary-variance initialization + unrolled-loop-vs-scan inefficiency found 2026-07-21). Table 6 priors. Used by all `regenerate_*_v2_scan_*.py` and `regenerate_beta1.7_*.py` scripts. |
| `Appendix_A/reproduction_2026-07-20/SCRIPT_LOG.md` | Full inventory of every script in `reproduction_2026-07-20/`: purpose, input, output, and measured/estimated wall-clock duration for each. Covers the (now closed out) β=1.7 N=20 ensemble test and the β=3/4 SNR sweep, including the frozen-chain bug found in both. |
| `MAIN_TEXT_FROZEN_CHAIN_CHECK_HANDOFF.md` | Starting point for a new session: checks whether the frozen-chain bug found in Appendix A's `ar1_hmc_v2.py` also affects the main-text production model (`run_inference.py`) — Table 3's 26 combinations and the 4 NGC 1365 observation fits. Not yet started; candidate data paths and the exact detection method are pre-identified. |
| `Appendix_A/reproduction_2026-07-20/BAYESIAN_DIAGNOSTICS_LEARNINGS.md` | Conceptual/methodological write-up (not a script log): why τ pins at the prior edge for β>2 (AR1's slope-2 Lorentzian ceiling), why β=1.7's distribution was skewed (low ESS, not a real posterior shape), why a sharp/tight distribution can be the worst sign not the best (frozen chains; the old buggy model's suspiciously-perfect diagnostics), a full glossary of every diagnostic used (r_hat, ESS, n_unique, skew, divergences, direct ACF) with its pitfalls, and the derivation of the thinning intervals used (50, then 40) from measured τ_int rather than guesswork. |
| `CORRECTION_LOG_2026-07-23.md` | Bug found and fixed: Table A.1's currently-published numbers were traced to an external, 2025-03-dated dataset generated via `simple_HMC.py` (the same confirmed-buggy fixed-unit-variance AR(1) init as `CORRECTION_LOG_2026-07-21.md`), not the corrected `ar1_hmc_v2.py`. Regenerated all three rows from the corrected model (β=1.7 via the long chain, thinned by 50, to avoid the known short-chain ESS≈41 artifact). New numbers: none of the three cases bracket their true value (old table had 2 of 3 bracketing) — required updating Table A.1 itself plus two narrative sentences (sec. 3.4.2 and Appendix A) that stated the old "2 of 3" claim. |

---

## 9. Data and Repo

| File | What it is |
|------|-----------|
| `data/clarsach/authors.md` | Author credits for the clarsach X-ray spectral fitting library (Huppenkothen, Corrales) |
| `data/clarsach/data/README.md` | Description of the clarsach data files included in this project |
| `repo/README.md` | README for the public-facing repo version of the project — simulation-based inference pipeline overview, installation, usage |
