# Source provenance

This independent repository was prepared on 2026-10-03 for the IEEE Access extension of the IROS2026 manuscript.

- Original geometric solver and path generators: `decwest/dwpp`, branch `dwvp`, commit `28bc0207`. The two ray/box functions in `solver.py` and the IROS path generators in `paths.py` are extracted under the original MIT license. The historical source remains unchanged.
- Package/configuration/result organization: informed by the committed `decwest/dwpp_python` package, commit `4719f452f1d44c57f3a07d1fc8f69a5db3e17412` (modular omni controller introduced at `4c0a8b5`). The new minimal configuration and runner were implemented independently; unrelated ECPP/DPP options and dirty RA-L experiment files were not copied.
- Public Nav2 contract: `nav2_omnidirectional_dwvp_controller`, developed with this package. Shared fixture parity verifies the regulated reachable box and selected command, not an entire Nav2 navigation rollout. Desired-vector and terminal laws are documented in the README. Plugin minimum-orientation-time and speed-regulation changes are intentional extensions, not claimed as identical to the earlier IROS implementation.
- IROS source manuscript: local `tex_docker_environment/projects/IROS2026/root.tex`. No RA-L manuscript, response, or rewritten results were used.
- Hardware geometry/data reference: `dwpp_test_simulation`, commit `51b79f5d8aa71baa3830ce6c3dec7da25b5fc31c`; `data/011_033` and its start-frame CSV importer. Its historical travel time is CSV duration. These real-robot trials are **not relabeled as new simulation results**, imported into the new experiment tables, or reused as new hardware repetitions.

Historical implementation differences are explicit: 20 Hz legacy simulator versus the new 30 Hz profile; old arc-length preview versus Humble Euclidean-circle preview; Euler omni integration versus exact body-twist integration; guarded finite desired yaw; common terminal control; equal feasible-box clipping for both baselines; and added cost/approach caps. The old simulator's cosine path was 0.75-scaled in the actual IROS hardware run; the present path uses the hardware scale.

The run manifest records the exact source-file content hash, repository revision, path hash, all configuration parameters, seed, interpreter/NumPy versions and operating system. Each output row identifies its complete trial. Timing is recorded separately because it varies with the host.

## Forward DWPP reference

`dwpp.py` copies `optimal_velocity_in_window` from `decwest/dwpp`, commit
`6e6fa05b02749b166aaaa342a05efd4989c36d52`, `simulator/controllers.py`
(SHA-256 `4652de23dbf5f134a395ef20c0cd633dd5d7d07e8a78016eaebb43db8efd84e5`). The source checkout was clean.
Copyright (c) 2024 Fumiya Ohnishi; the repository MIT license applies.
The curvature threshold, candidate ordering, corner distance and tie tolerance
are preserved. The surrounding wrapper shares the omni Euclidean preview,
SE(2) plant, symmetric acceleration limits, and fractional cost/approach box
contraction. It uses forward v in [0, vx_max], vy=0 and the same yaw limits;
there is no curvature slowdown. During terminal rotation the final positional
tangent supplies heading, so supplied path orientations never control DWPP.

`fixtures/dwpp_reference_cases.json` contains 42 kernel cases and 12 complete
nonterminal command cases evaluated directly from the original source. Generate
it locally with `tools/export_dwpp_reference.py`; the tests need no other checkout
and no network. Complete-command parity covers adaptive/fixed preview, positive
and negative curvature, off-ray windows, and approach regulation. The source
oracle's minimum approach speed is divided by sqrt(2) to match the shared
fractional contraction of the omni x/y box. The terminal wrapper and observation
noise are extensions, not part of the upstream parity claim.
