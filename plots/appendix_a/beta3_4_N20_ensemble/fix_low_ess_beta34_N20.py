"""
Re-run the 3 flagged realizations from the beta=3/4 N=20 ensembles that
passed the original n_unique>1 (not-frozen) check but showed severe
mixing problems (n_unique << 2000, the same low-ESS pathology as the
original beta=1.7 short-chain issue): beta=3/realization_01 (n_unique=44),
beta=4/realization_08 (n_unique=10), beta=4/realization_14 (n_unique=474).

Uses a stricter threshold this time (n_unique >= 1000, not just >1) and
retries with fresh seeds until it's met, up to MAX_RETRIES. Overwrites
the flagged realization's npz file in place once a healthy replacement
is found, keeping the same realization index so the ensemble stays at
a clean N=20 per beta.
"""
import os
import sys
import time
import numpy
import jax
import jax.numpy as jnp

sys.path.append("/Users/home/Documents/science/project/johannes/publication/paper_one/package/colorednoise-master")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import colorednoise as cn
from ar1_hmc_v2_widerprior import model, NUM_WARMUP, NUM_SAMPLES
from numpyro.infer import MCMC, NUTS

T = 1000
SHIFT_TERM = 7.0
THRESHOLD = 1000
MAX_RETRIES = 15

TARGETS = [
    ("beta3_N20_shortchain_data", 1, 3.0, 9001),
    ("beta4_N20_shortchain_data", 8, 4.0, 9008),
    ("beta4_N20_shortchain_data", 14, 4.0, 9014),
]

sampler = NUTS(model)

for out_dir, idx, beta_true, base_seed in TARGETS:
    path = os.path.join(out_dir, f"realization_{idx:02d}.npz")
    flux_rng = numpy.random.default_rng(base_seed)
    print(f"\n=== Re-running {path} (beta_true={beta_true}) ===", flush=True)

    best = None
    for attempt in range(MAX_RETRIES):
        flux = cn.powerlaw_psd_gaussian(beta_true, T, random_state=flux_rng)
        count = flux_rng.poisson(numpy.exp(flux - flux.mean() + SHIFT_TERM))
        count_j = jnp.asarray(count.astype("int32"))

        t0 = time.time()
        mcmc = MCMC(sampler, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(base_seed + attempt), count_j, T)
        posterior_samples = mcmc.get_samples()

        tau_param = numpy.array(posterior_samples["tau_param"])
        n_unique = len(numpy.unique(tau_param))
        dt = time.time() - t0
        print(f"  attempt {attempt}: n_unique={n_unique}  tau_mean={tau_param.mean():.2f}  wall_time={dt:.1f}s", flush=True)

        if n_unique >= THRESHOLD:
            best = (flux, count, posterior_samples, tau_param, n_unique)
            break
    else:
        print(f"  WARNING: never reached n_unique>={THRESHOLD} in {MAX_RETRIES} attempts; "
              f"keeping best available for inspection", flush=True)
        best = (flux, count, posterior_samples, tau_param, n_unique)

    flux, count, posterior_samples, tau_param, n_unique = best
    flux_predicted = numpy.array(posterior_samples["flux_predicted"]).T
    mean_param = numpy.array(posterior_samples["mean_param"])
    var_param = numpy.array(posterior_samples["var_param"])

    numpy.savez(
        path,
        generated_flux=flux,
        flux_predicted=flux_predicted,
        generated_count=count,
        tau_param=tau_param,
        mean_param=mean_param,
        var_param=var_param,
        n_unique_tau=n_unique,
        n_attempts=attempt + 1,
        note="regenerated_by_fix_low_ess_script",
    )
    print(f"  Saved replacement -> {path}  (final n_unique={n_unique})", flush=True)

print("\nDone.", flush=True)
