# Reproducibility archive

This archive accompanies the manuscript **“Constitutive identifiability in torsional rheometry: exact homogeneous kernel and interface confounding in twist-induced pressure.”**

## Contents

- `input/`: finite-element solver sources, explicit Mooney-Rivlin and kernel-perturbed solver variants, 38x40 baseline/comparison states, nonlinear pressure profiles, interface-tangent profiles and fitting-window data.
- `reference/`: numerical summaries used for manuscript cross-checks.
- `meshcheck_30x40/`: independent edge-resolved nonlinear repetition of the constitutive and coordinated-interface states.
- `scripts/`: observation-space reproductions, nonlinear reruns, global-reaction postprocessing and verification scripts.
- `results/`: regenerated observation-space outputs, global reactions and finite-state comparison metrics.
- `mechanism/`: source-window, interface-transition, observation-window, direct/state, loading/material, radial-location, source-identity and resolution/projection calculations reported in OR9.
- `preload/double_path_null_control/`: representative joint-kernel calculation for two prescribed axial stretches reported in OR8.
- `thin_gap_relevance/`: matched-resolution H0/R0=0.10 tangent control reported in OR7.9, including solver inputs, replay script and machine-readable result.

## Solver implementation

`input/solver_core.py` is the common finite-element scaffold retained for compatibility with the rerun scripts. The manuscript calculations use the Mooney-Rivlin energy reconstructed in the rerun scripts. For transparency, the archive also supplies the expanded sources `input/solver_mooney_rivlin.py` and `input/solver_mooney_rivlin_kernel.py`, which show the baseline and kernel-perturbed energies directly without runtime source substitution. Stress postprocessing uses `input/postprocess_core.py`.

## Fast numerical verification

From the archive root run:

```text
python scripts/reproduce_finite_profile_metrics.py
python scripts/reproduce_subspace_geometry.py
python scripts/reproduce_observation_covariance.py
python scripts/reproduce_fitting_window_metrics.py
python scripts/verify_reproduction.py
python mechanism/verify_mechanism_results.py
python thin_gap_relevance/verify_r208.py
```

Additional verification scripts retained in `scripts/` reproduce the auxiliary controls used in the Online Resource. Internal version identifiers in some filenames are provenance labels only and are not scientific terminology used in the manuscript.

The source-identity calculation shows that strong boundary-near cancellation also occurs for the Mooney-Rivlin coefficient-partition source, so the effect is not unique to the Q=1 source among the tested rim-weighted torsional sources. The source-window singular-value calculation shows that the spatially windowed source-to-pressure map retains substantial higher output directions, whereas the complete rim-weighted source response is concentrated mainly in the leading modes. The manufactured interior-support calculation demonstrates strong dependence on radial source weighting but is not an additional member of the homogeneous constitutive-kernel class. Resolution/projection calculations assess sensitivity to window width and rank-2/rank-3 nuisance projection. The joint-kernel calculation shows that a representative perturbation protecting two prescribed axial stretches retains an O(1) noise-normalized pressure response under the same 0.05G envelope normalization. The thin-gap calculation extends the tangent finite-sensitivity/interface-span comparison to H0/R0=0.10 at approximately matched reference axial element size.

## Full finite-element reruns

`python scripts/rerun_nonlinear_finite_states.py` recomputes the principal 38x40 finite constitutive state and coordinated interface states from the supplied baseline state. `python scripts/rerun_nonlinear_30x40_meshcheck.py` independently rebuilds the 30x40 edge-resolved mesh and recomputes the corresponding nonlinear states.

Changed meshes, loading amplitudes or material baselines require fresh equilibrium solutions and JAX compilation. Stored outputs are included so the reported values can also be checked without full reruns.

## Software

The archive uses Python, NumPy, SciPy and JAX. The submission build was verified in a Python 3.13.5 environment with NumPy 2.3.5, SciPy 1.17.0 and JAX 0.9.0.1. Fast verification scripts do not require operating-system-specific paths.
