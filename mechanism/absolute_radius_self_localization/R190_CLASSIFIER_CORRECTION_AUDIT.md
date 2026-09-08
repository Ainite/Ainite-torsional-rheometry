# R190 classifier correction audit

The preregistered rule separates three clauses: localization, winner mechanics, and numerical validity. The first implementation accidentally defined the per-case `valid` flag as `numerical checks AND mechanics_pass`. The aggregate then interpreted any false `valid` as an `INVALID_NUMERICAL_GATE` verdict.

That implementation does not match the frozen prose. No numerical result or threshold is changed here. The correction only separates:

- `numerical_valid`: centered-step angle < 0.01 deg and both equilibrium residual infinity norms < 1e-7;
- `mechanics_pass`: the frozen direct/state winner mechanics gate;
- `pass_corrected_classifier`: numerical_valid AND localization_pass AND mechanics_pass.

All three cases pass numerical validity and localization. Only x_max=0.73 passes the winner-mechanics gate. Therefore, under the preregistered verdict rule, the corrected verdict is `PROSPECTIVE_OBSERVER_SELF_LOCALIZATION_NOT_SUPPORTED`, not `INVALID_NUMERICAL_GATE`.

The original aggregate is preserved as `results/R190_PROSPECTIVE_SELF_LOCALIZATION_GATE_RAW_CLASSIFIER.json`, and the original aggregator as `aggregate_r190_raw_classifier.py`.
