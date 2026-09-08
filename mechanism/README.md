# Radial source-accumulation and observer controls

This directory contains finite-BVP controls supporting OR9. The scientific directories are organized by the question tested:

- `cross_discretization_geometry/`: independent radial partition and second thickness geometry.
- `transition_onset_control/`: tests whether cancellation follows the modeled interface taper.
- `observer_window_translation/`: initial observer-translation calculation using pre-centered source pairs.
- `nested_direct_state/`: two-level direct-constitutive/equilibrium-state decomposition at three observer cutoffs.
- `loading_amplitude/`: repetitions at rim shear amplitudes 0.15 and 0.25.
- `material_baseline/`: repetitions at Mooney-Rivlin partition parameters 0.15 and 0.45.
- `absolute_radius_self_localization/`: absolute-midpoint scans at unseen observer cutoffs, including the retained negative result for the stronger single-global-winner mechanics rule.
- `boundary_neighborhood_nested/`: subsequent fixed boundary-neighborhood test of the narrower two-level decomposition statement.
- `source_identity_control/`: repeats the absolute-midpoint scans with a Mooney-Rivlin coefficient-partition source; the boundary-near effect persists and is therefore not Q=1-source-specific.
- `resolution_projection_control/`: local cell-size, smooth-window-width, rank-2/rank-3 and extended-outer-scan checks.

Retained R18x/R19x strings inside frozen JSON/source filenames are provenance identifiers, not manuscript section numbers. The publication interpretation is deliberately narrower than the earlier development hypothesis: OR9 uses these calculations as a diagnostic of observer/source accumulation in the tested finite BVP, not as a constitutive-null-specific cancellation law.

Run `python mechanism/verify_mechanism_results.py` for the historical eight-control verdict check and `python scripts/verify_round17_new_controls.py` for the additional source-identity/resolution/preload checks.

## Review-17 mechanism closure

- `band_operator_svd/`: SVD of the unprojected 25x19 localized Q=1 band-response matrices. In the scan's fixed equal-amplitude band coordinates the localized-source transfer is not generally rank two, while the complete edge-weighted Q=1 response is strongly concentrated in the leading output modes.
- `interior_support_control/`: manufactured C2 spatial source-support control. It is not a homogeneous constitutive law; it is used only to separate pressure-window truncation from radial source support. Restricting the explicit Q=1 tangent source to the specimen interior raises the interface-orthogonal fraction above 0.5 for all three tested observers.

Run `python scripts/verify_review17_mechanism_closure.py` from the archive root for the fixed-output verification of these two controls. The heavier reproduction scripts are retained in the two control directories.
