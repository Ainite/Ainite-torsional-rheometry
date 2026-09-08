# R194 classifier audit

The fixed localization rule is |midpoint-xmax| <= 0.05. For xmax=0.93 the selected midpoint is 0.88, whose mathematical distance is exactly 0.05; binary floating-point evaluation produced 0.050000000000000044 and therefore a false Boolean failure in the first aggregator. No scientific quantity, candidate grid, threshold, or selected winner is changed here. The corrected classifier evaluates the written inequality with a 1e-12 comparison tolerance.
