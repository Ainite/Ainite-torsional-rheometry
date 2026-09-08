# R181 feasibility amendment (before any scientific output)

The original preregistration specified 34x40 for both new cases. Two executions, with 120 s and 240 s wall-clock limits, produced no equilibrium state, tangent profile, projection metric, or scientific result; the only output was the preregistered rule. The bottleneck is JAX compilation / Hessian construction at the new element count.

Before observing any R181 scientific value, the axial count is reduced from 40 to 20 while preserving all radial choices: nr=34, edge_cells=10, bulk_cells=24 on 0<=R/R0<=0.9, edge_width/R0=0.1, the same C2 windows, centers, cancellation pair, projection definition, and numerical/pass thresholds. Both aspect-ratio cases use the same 34x20 discretization.

Claim boundary: a pass supports cross-radial-discretization replication and a low-cost cross-thickness replication. It does not establish full mesh independence because the axial resolution differs from R180. The original 34x40 gate remains unresolved rather than failed.
