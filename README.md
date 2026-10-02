# Omnidirectional Dynamic Window Vector Pursuit

A small, independent Python package for reproducible comparison of omnidirectional Vector Pursuit with componentwise clipping and Dynamic Window Vector Pursuit (DWVP). It generates the IEEE Access simulation material from explicitly specified position **and** orientation references.

The ROS 2 controller is maintained separately in [nav2_omnidirectional_dwvp_controller](https://github.com/decwest/nav2_omnidirectional_dwvp_controller). This package does not need ROS.

## Run

With Python 3.10 or later and `uv`:

```bash
uv sync --extra dev
uv run pytest -q
uv run dwvp-study all
```

The complete study contains 248 condition entries: 16 mechanism comparisons, 8 obstacle/approach ablations, and 224 parameter-sweep entries. Reused nominal settings produce 204 distinct deterministic trajectories (198 distinct trajectories in the sweep entries); these are not independent repetitions. Use `dwvp-study mechanism`, `dwvp-study obstacles`, or `dwvp-study sweeps` separately. `--config configs/paper.yaml`, `--seed 0`, `--output results/paper`, and `--force` are available. A trial is reused only when its path, complete configuration, obstacles, seed and Python-source content hash match. Changing a source file invalidates cached trials. Every requested setting is retained, including any timeout or collision.

The canonical configuration uses 30 Hz, physical x/y limits ±0.22 m/s, yaw limits ±0.60 rad/s, x/y acceleration limits 0.22 m/s², yaw acceleration 0.60 rad/s², nominal translation speed 0.22 m/s, and adaptive lookahead 0.11–0.33 m with 1.5 s lookahead time. `dt` is calculated as `1/frequency`. The minimum orientation time is 0.20 s. No noise is added in the publication run; repeated deterministic executions are not statistical repetitions.

## Comparison and models

Both controllers receive the same desired vector, physical limits, previous applied velocity, speed regulation, and reachable box. VP projects the desired vector componentwise onto that box; DWVP solves the unweighted ray–box intersection/projection problem. Both then pass through the same physical plant clipping. Thus command-feasibility results refer to the **final feasible commands**, not the infeasible pre-projection demand. The latter is recorded separately.

The reference translational speed is the norm of the physical x/y box maxima; the nominal speed and RPP-style cost/approach reductions constrain the feasible box. Scaling the desired ray alone would have no effect. Regulation scales the physical x/y intervals, intersects them with the dynamic window, and retains the nearest original dynamic endpoint if the contraction is temporarily unreachable. The yaw interval is unchanged. Transient excess over the requested regulated speed can therefore occur while all physical velocity and acceleration limits remain satisfied.

Preview follows Humble Nav2 RPP's Euclidean lookahead-circle convention, interpolates x/y at the circle/segment intersection, and takes yaw from the first outer path pose. Reference yaw is supplied independently of the path tangent. Terminal control begins within the position tolerance and uses a finite yaw target with a stopping-distance limit. All methods use the same terminal behavior. Success requires position error ≤0.02 m, wrapped yaw error ≤1°, and all applied velocity components ≤0.001 in their respective units. Timeout is 120 s. Both first pose-tolerance time and settled travel time are recorded.

The plant integrates a constant body-frame twist exactly in SE(2). Its applied and measured velocities are identical: there is no actuator lag, wheel-level model, localization error, grid cost quantization, global replanning, or obstacle-avoidance controller. The constraint model is an axis-aligned body-velocity box, not a wheel-level reachable set.

The obstacle study assumes an HSR circular footprint of radius 0.22 m. Its two circular obstacles are `(x,y,radius)=(0.55,-0.45,0.10)` and `(1.42,0.55,0.10)` m. Cost regulation uses the distance from the robot centre to the nearest obstacle surface, consistent with inverse costmap inflation outside the inscribed region; this is distinct from footprint clearance. Inflation radius is 0.70 m, factor 3.0 m⁻¹, cost distance 0.60 m, cost gain 1, and minimum regulated speed 0.05 m/s. Approach scaling uses the post-cost speed and a 0.30 m distance, with minimum approach speed 0.05 m/s. Collision evaluation uses a conservative swept-circle bound (segment clearance minus the exact-step arc sagitta). Simulation continues after a collision so failures remain visible. This scene evaluates slowdown, not obstacle avoidance or a collision-free guarantee.

## Reference paths and studies

- IROS docking: 1 m + 1 m right-angle polyline, 201 samples, tangent yaw except a final −180° docking pose.
- IROS curve: `y=0.75*(1-cos(2*pi*1.5*x/1.5))`, x ∈ [0,1.5] m, 501 arc-length-resampled points, tangent yaw. This is the **hardware** reference scale; the old simulator used a larger path.
- Constant-heading corner: docking geometry with yaw fixed at zero.
- Independent-heading curve: cosine geometry with yaw `pi/2*(3*s²−2*s³)`, where `s` is normalized accumulated path length. Heading evolves smoothly from 0° to 90° independently of the tangent.

All trials start at zero pose and velocity. The mechanism study compares adaptive VP clipping, fixed-0.11 m VP clipping, fixed-0.33 m VP clipping, and adaptive DWVP. The primary controlled comparison is adaptive VP versus adaptive DWVP. Obstacle trials cross cost on/off with approach on/off for both methods. Sweeps cover lookahead time, fixed lookahead, acceleration scale, unequal lateral acceleration, cost distance/gain, and approach distance. Exact grids and all results are published, with no selection based on performance.

## Outputs

`results/paper/` contains:

- `manifest.json`: every trial specification, source and path hashes, dependencies, outcomes, and original-source provenance pointers.
- `{mechanism,obstacles,sweeps}/summary.{csv,json,tex}`: complete numerical results and manuscript table material.
- PDF/PNG figures, including trajectories, velocity/direction profiles, regulation ablations, and all seven parameter sweeps.
- `trials/<content-hash>/trial.json` and `trajectory.npz`: resumable full numerical data (generated locally; excluded from Git).
- `performance.csv`: Python controller-call mean, p95 and maximum wall-clock microseconds per trial. Timing is nondeterministic, excludes plotting and plant integration, and does not establish a ROS real-time guarantee.

Arrays distinguish unconstrained `demands`, final controller `commands`, plant `applied`, and ideal `measured` velocities. Position/yaw errors use the nearest sampled reference pose. Direction distortion is the angle between demand and applied vectors after dividing each component by its physical axis magnitude; the **controller's projection remains unweighted**. Zero-vector direction is undefined and excluded from the angular average. Translational and yaw jerk RMS are reported separately. Constraint excess magnitudes are normalized by per-axis limits.

Scientific metrics/arrays are deterministic for a fixed environment and seed. Wall-clock timing and compressed-container metadata are not claimed to be byte-deterministic. Dependency versions are locked in `uv.lock`.

Export only generated assets into a manuscript checkout:

```bash
uv run python tools/export_manuscript.py --destination /path/to/manuscript/generated
```

The manuscript exporter does not edit prose. Numerical configuration, source commit and results must be checked together before updating claims.

## Validation and provenance

Tests include an independent SciPy scalar-minimization oracle, signed and zero ray components, abrupt cap changes, pure yaw, stopping, deterministic seeds, timeouts, speed bounds, exact integration and swept-footprint collision detection. `fixtures/solver_cases.json` supplies 107 shared Python/C++ cases. To validate a built plugin fixture runner:

```bash
uv run python tools/check_plugin_parity.py /path/to/solver_fixture_runner
```

See [docs/provenance.md](docs/provenance.md) for exact source revisions and [docs/results.md](docs/results.md) for bounded interpretation of the generated results. Code is MIT-licensed; existing Fumiya Ohnishi copyright is preserved. This is research software and makes no closed-loop stability claim.
