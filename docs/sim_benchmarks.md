# cosmocnc_jax Catalogue Simulator Benchmarks

## Setup
- GPU: NVIDIA RTX PRO 6000 Blackwell (96 GB)
- Observables: `q_so_sim` + `p_so_sim` (full forward model with correlated scatter)
- M_min = 1e14, M_max = 1e16, z ∈ [0.01, 3.0]
- ~166k total clusters, ~16k selected (q > 5)

## Init (one-time)

`compute_class_szfast()` was passing `ndim_redshifts = n_z`, causing Cython to compute at the full simulation grid. Fixed by capping at `min(n_z, 100)` since the JAX emulators handle arbitrary z-grids independently.

| Config | Before fix | After fix |
|--------|-----------|-----------|
| n_z=100 | 26s | 26s |
| n_z=1000, n_M=16384 | 147s | **26s** |
| n_z=1000, n_M=50000 | 152s | **18s** |

## MCMC-step costs (after init, JIT cached)

| Component | n_z=1000, n_M=16384 | n_z=1000, n_M=50000 |
|-----------|--------------------|--------------------|
| `update_params` | 3 ms | 3 ms |
| `get_hmf` | 7 ms | 17 ms |

## Catalogue generation (after JIT warmup ~3s)

| Config | GPU kernel | Full e2e (1 cat) | Batch (10 cats) |
|--------|-----------|-----------------|-----------------|
| n_z=100, n_M=16384 | 25 ms | 185 ms | 212 ms/cat |
| n_z=1000, n_M=16384 | 2 ms | 31 ms | 26 ms/cat |
| n_z=1000, n_M=50000 | 2 ms | 6.5 ms | 12 ms/cat |

## GPU memory
Stable at ~74 GB across all grid sizes (JAX preallocates).

## Key changes (2026-03-04)
- `cosmo.py`: Capped `ndim_redshifts` to `min(n_z, 100)` in `classy_sz_jax` init
- `tests/test_sim.py`: Fixed observables config to `[["q_so_sim", "p_so_sim"]]`
