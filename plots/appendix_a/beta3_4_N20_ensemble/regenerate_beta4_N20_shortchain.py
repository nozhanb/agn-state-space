"""
Generate 20 independent colorednoise realizations at beta=4.0, T=1000,
shift=7 (same fixed configuration used for the published Table A.1 /
SNR-sweep beta=4 row), run each through the corrected AR(1)/HMC model
(wider-prior version, ar1_hmc_v2_widerprior.py -- the one actually behind
the published shift=7 numbers), SHORT chain (num_warmup=1000,
num_samples=2000), batched in one process.

WHY: to test whether the beta=3/4 "miss" in Table A.1 (recovered median
2.48/3.86 vs true 3.00/4.00) is single-realization periodogram/estimation
noise that would average out over independent realizations (as it did for
beta=1.7: single-realization 1.59 -> N=20 ensemble mean 1.71, landing on
the true 1.70), or a persistent bias that survives averaging (consistent
with the structural AR(1)-ceiling explanation already in the paper).

Frozen-chain check is built in from the start this time (n_unique on
tau_param, NOT a std threshold -- see CORRECTION_LOG_2026-07-21.md for
why the std check gave false positives/negatives). Retries with a fresh
seed on detection, up to MAX_RETRIES per realization, since shift=7 for
beta=3/4 hit this bug in the original (pre-fix) sweep pass.
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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beta4_N20_shortchain_data")
os.makedirs(OUT_DIR, exist_ok=True)

T = 1000
SHIFT_TERM = 7.0
BETA_TRUE = 4.0
N_REALIZATIONS = 20
MAX_RETRIES = 8

flux_rng = numpy.random.default_rng(2026072511)

sampler = NUTS(model)

t_start = time.time()
for i in range(N_REALIZATIONS):
    flux = cn.powerlaw_psd_gaussian(BETA_TRUE, T, random_state=flux_rng)
    count = flux_rng.poisson(numpy.exp(flux - flux.mean() + SHIFT_TERM))
    count_j = jnp.asarray(count.astype("int32"))

    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        seed = i * 100 + attempt
        mcmc = MCMC(sampler, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), count_j, T)
        posterior_samples = mcmc.get_samples()

        tau_param = numpy.array(posterior_samples["tau_param"])
        n_unique = len(numpy.unique(tau_param))
        dt = time.time() - t0

        if n_unique > 1:
            break
        print(f"realization {i} attempt {attempt}: FROZEN (n_unique={n_unique}), retrying with new seed", flush=True)
    else:
        print(f"realization {i}: FAILED after {MAX_RETRIES} attempts, still frozen -- saving anyway, flagged", flush=True)

    flux_predicted = numpy.array(posterior_samples["flux_predicted"]).T
    mean_param = numpy.array(posterior_samples["mean_param"])
    var_param = numpy.array(posterior_samples["var_param"])

    print(f"realization {i}: mean_count={count.mean():.1f}  tau_mean={tau_param.mean():.2f}  "
          f"n_unique={n_unique}  attempts={attempt+1}  wall_time={dt:.1f}s", flush=True)

    numpy.savez(
        os.path.join(OUT_DIR, f"realization_{i:02d}.npz"),
        generated_flux=flux,
        flux_predicted=flux_predicted,
        generated_count=count,
        tau_param=tau_param,
        mean_param=mean_param,
        var_param=var_param,
        n_unique_tau=n_unique,
        n_attempts=attempt + 1,
    )

total_dt = time.time() - t_start
print(f"\nDone. Total wall time: {total_dt:.1f}s ({total_dt/60:.1f} min)", flush=True)
