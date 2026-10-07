# Omnidirectional Dynamic Window Vector Pursuit

A small, independent Python package for reproducible comparison of omnidirectional Vector Pursuit with componentwise clipping and Dynamic Window Vector Pursuit (DWVP). It generates the IEEE Access simulation material from explicitly specified position **and** orientation references.

The ROS 2 controller is maintained separately in [nav2_omnidirectional_dwvp_controller](https://github.com/decwest/nav2_omnidirectional_dwvp_controller). This package does not need ROS.

## Run

With Python 3.10 or later and `uv`:

```bash
uv sync --extra dev
uv run pytest -q
uv run dwvp-study legacy-all --config configs/paper.yaml --output results/legacy
```

The archived study contains 248 condition entries: 16 mechanism comparisons, 8 obstacle/approach ablations, and 224 parameter-sweep entries. Reused nominal settings produce 204 distinct deterministic trajectories (198 distinct trajectories in the sweep entries); these are not independent repetitions. Use `dwvp-study mechanism`, `dwvp-study obstacles`, or `dwvp-study sweeps` separately. `--config configs/paper.yaml`, `--seed 0`, `--output results/legacy`, and `--force` are available. A trial is reused only when its path, complete configuration, obstacles, seed and Python-source content hash match. Changing a source file invalidates cached trials. Every requested setting is retained, including any timeout or collision.

The canonical configuration uses 30 Hz, physical x/y limits ±0.22 m/s, yaw limits ±0.60 rad/s, x/y acceleration limits 0.22 m/s², yaw acceleration 0.60 rad/s², desired translation speed 0.32 m/s (capped at the physical box norm, approximately 0.311 m/s; no contraction of the original ±0.22 m/s axis envelope), and adaptive lookahead 0.11–0.33 m with 1.5 s lookahead time. `dt` is calculated as `1/frequency`. The minimum orientation time is 0.20 s. No noise is added in the publication run; repeated deterministic executions are not statistical repetitions.

## Comparison and models

Both controllers receive the same desired vector, physical limits, previous applied velocity, speed regulation, and reachable box. VP projects the desired vector componentwise onto that box; DWVP solves the unweighted ray–box intersection/projection problem. Both then pass through the same physical plant clipping. Thus command-feasibility results refer to the **final feasible commands**, not the infeasible pre-projection demand. The latter is recorded separately.

The default reference translational speed is the minimum magnitude allowed in all positive and negative x/y directions (0.22 m/s for symmetric limits). `vp_translation_speed` overrides it; `configs/paper.yaml` explicitly preserves the old box-diagonal demand. The same magnitude defines the orientation time. The nominal speed and RPP-style cost/approach reductions constrain the feasible box. In the archived profile, the demand magnitude is the box diagonal. Scaling the desired ray alone would have no effect. Regulation scales the physical x/y intervals, intersects them with the dynamic window, and retains the nearest original dynamic endpoint if the contraction is temporarily unreachable. The yaw interval is unchanged. Transient excess over the requested regulated speed can therefore occur while all physical velocity and acceleration limits remain satisfied.

Preview follows Humble Nav2 RPP's Euclidean lookahead-circle convention, interpolates x/y at the circle/segment intersection, and takes yaw from the first outer path pose. Reference yaw is supplied independently of the path tangent. Terminal control begins within the position tolerance and uses a finite yaw target with a stopping-distance limit. All methods use the same terminal behavior. Success requires position error ≤0.02 m, wrapped yaw error ≤1°, and all applied velocity components ≤0.001 in their respective units. Timeout is 120 s. Both first pose-tolerance time and settled travel time are recorded.

The plant integrates a constant body-frame twist exactly in SE(2). Its applied and measured velocities are identical: there is no actuator lag, wheel-level model, localization error, grid cost quantization, global replanning, or obstacle-avoidance controller. The constraint model is an axis-aligned body-velocity box, not a wheel-level reachable set.

The obstacle study assumes an HSR circular footprint of radius 0.22 m. Its two circular obstacles are `(x,y,radius)=(0.55,-0.45,0.10)` and `(1.42,0.55,0.10)` m. Cost regulation uses the distance from the robot centre to the nearest obstacle surface, consistent with inverse costmap inflation outside the inscribed region; this is distinct from footprint clearance. Inflation radius is 0.70 m, factor 3.0 m⁻¹, cost distance 0.60 m, cost gain 1, and minimum regulated speed 0.05 m/s. Approach scaling uses the post-cost speed and a 0.60 m distance, with minimum approach speed 0.05 m/s. Collision evaluation uses a conservative swept-circle bound (segment clearance minus the exact-step arc sagitta). Simulation continues after a collision so failures remain visible. This scene evaluates slowdown, not obstacle avoidance or a collision-free guarantee.

## Reference paths and studies

- IROS docking: 1 m + 1 m right-angle polyline, 201 samples, tangent yaw except a final −180° docking pose.
- IROS curve: `y=0.75*(1-cos(2*pi*1.5*x/1.5))`, x ∈ [0,1.5] m, 501 arc-length-resampled points, tangent yaw. This is the **hardware** reference scale; the old simulator used a larger path.
- Constant-heading corner: docking geometry with yaw fixed at zero.
- Independent-heading curve: cosine geometry with yaw `pi/2*(3*s²−2*s³)`, where `s` is normalized accumulated path length. Heading evolves smoothly from 0° to 90° independently of the tangent.

Archived trials start at zero pose and velocity. `simulate(..., initial_pose=[x, y, yaw])` can override the initial pose; velocity still starts at zero. The mechanism study compares adaptive VP clipping, fixed-0.11 m VP clipping, fixed-0.33 m VP clipping, and adaptive DWVP. The primary controlled comparison is adaptive VP versus adaptive DWVP. Obstacle trials cross cost on/off with approach on/off for both methods. Sweeps cover lookahead time, fixed lookahead, acceleration scale, unequal lateral acceleration, cost distance/gain, and approach distance. Exact grids and all results are published, with no selection based on performance.

## Outputs

`results/paper/` contains:

- `manifest.json`: every trial specification, source and path hashes, dependencies, outcomes, and original-source provenance pointers.
- `{mechanism,obstacles,sweeps}/summary.{csv,json,tex}`: complete numerical results and manuscript table material.
- PDF/PNG figures, including trajectories, velocity/direction profiles, regulation ablations, and all seven parameter sweeps.
- `trials/<content-hash>/trial.json` and `trajectory.npz`: resumable full numerical data (generated locally; excluded from Git).
- `performance.csv`: Python controller-call mean, p95 and maximum wall-clock microseconds per trial. Timing is nondeterministic, excludes plotting and plant integration, and does not establish a ROS real-time guarantee.

Arrays distinguish unconstrained `demands`, final controller `commands`, plant `applied`, and ideal `measured` velocities. Position error is measured to the closest polyline-segment projection. Reference yaw is interpolated at that same segment location using the shortest wrapped angular difference; a nearest-vertex yaw is not mixed with a segment position error. Direction distortion is the angle between demand and applied vectors after dividing each component by its physical axis magnitude; the **controller's projection remains unweighted**. Zero-vector direction is undefined and excluded from the angular average. Translational and yaw jerk RMS are reported separately. Constraint excess magnitudes are normalized by per-axis limits.

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

## Straight-path studies

Run every condition from the single profile, without network access:

```bash
mkdir -p build/uv-cache build/matplotlib build/tmp
export UV_CACHE_DIR="$PWD/build/uv-cache" MPLCONFIGDIR="$PWD/build/matplotlib" TMPDIR="$PWD/build/tmp" PYTHONDONTWRITEBYTECODE=1
uv run --offline --locked --python 3.11.11 dwvp-study all --config configs/access_v2.yaml --output results/access_v2 --seed 0 --workers 4
uv run --offline --locked --python 3.11.11 pytest -q --basetemp=build/pytest-tmp
uv run --offline --locked --python 3.11.11 python tools/validate_access_v2.py results/access_v2
```

`test1` through `test4` run individual studies; `--workers` controls independent
simulation processes (default 1). `--force` recomputes matching cached trials.
The old `mechanism`, `obstacles`, `sweeps`, and `legacy-all` commands remain
available, defaulting to `results/legacy/`. Saved `results/paper/` is preserved.

| Command | Conditions | Output directory |
|---|---|---|
| `test1` | 16 conditions: four initial lateral offsets; DWPP, Clipped VP, Scaled VP, DWVP | `test1/` |
| `test2` | 176 conditions: nine nominal heading ramps, one step, and acceleration sweeps; Clipped VP, Scaled VP, Scaled VP (vel. and acc.), DWVP | `test2/` |
| `test3` | 4 conditions: obstacle proximity regulation on/off; RPP and DWVP; goal approach regulation always enabled | `test3/` |
| `test4` | (a) Each VP baseline matched to DWVP travel time, (b) preview (120), (c) localization noise (1000) | `test4/` |
| `acceleration-sweep` | Optional acceleration sweep (100), previously test4 (c); initial offset and two heading ramps | `acceleration-sweep/` |
| `regulation-sweep` | Optional cost-distance, cost-gain and approach-distance sweep (36); Clipped VP, Scaled VP, DWVP | `regulation-sweep/` |
| `preview-noise` | Test 4(c) DWVP preview × noise grid (3600), also runnable separately | `preview-noise/` |
| `all` | Four studies including preview × noise, every time-match candidate and all 20 noise seeds; excludes regulation and acceleration optional sweeps | `test1/`–`test4/`, `preview-noise/` |

`configs/access_v2.yaml` contains controller settings and the complete study grid.
The default profile uses 0.22 m/s VP demand and adaptive preview time 0.75 s,
bounded between 0.11 and 0.33 m. Fixed-distance sweeps override these bounds.
All VP variants and DWVP use the same demand and reachable box; DWVP can select a ray scale
above one. It always selects the largest alpha at ray/box intersections and
breaks equal-distance nonintersection ties toward larger alpha. Goal slowdown
comes from approach regulation and terminal control. Keep `approach_distance > 0`:
with the nominal straight path, disabling it produces 0.092 m goal overshoot,
compared with zero at 0.6 m. These are simulation observations.
DWPP is a forward differential-drive reference with `(v, 0, omega)`,
ignores supplied path yaw, and uses the final positional tangent for terminal
rotation. Its transplanted command selector is validated against upstream outputs.
RPP uses the same lookahead point, PP curvature and terminal handling as DWPP.
Its demand is `(v, 0, curvature * v)`, where
`v = vx_max * speed_cap / box_speed`: 0.22 m/s without regulation, reduced by
the same cost/approach ratio used by the velocity box. It uses no curvature
regulation and selects no dynamic-window optimum;
the demand is clipped component by component to the shared reachable regulated
box before application.
Test3 reports RPP's pre-clip `unconstrained_demand_violation_pct` and DWVP's
selected-command `command_constraint_violation_pct`; the applied-command column
remains separate and is zero for both methods. The `demand_*_violation_steps`
and `command_*_violation_steps` columns split physical velocity and acceleration
exceedances, their overlap, and their union. The demand counts use the same
1e-10 velocity tolerance as `unconstrained_demand_violation_pct`, relative to
the previous applied velocity. Regulated-cap excess is a separate quantity.

`vp` (Clipped VP) retains component-wise clipping of the original VP demand.
`vp_scaled` (Scaled VP) first scales that demand by the largest common factor
in [0, 1] that fits the velocity box after obstacle/goal speed regulation,
without acceleration limits. It then clips each component to the common dynamic
window. `vp_scaled_accel` (Scaled VP (vel. and acc.)) additionally scales the
change from current velocity to that target by one common factor to fit
the per-axis acceleration limits times the control period, then clips to the
same window. This resembles the idea of Nav2 velocity smoother's
`scale_velocities`; it is not the same implementation. Both methods share
Clipped VP's terminal control. After an abruptly reduced speed cap, all methods
use the existing reachable braking endpoints when the new cap is unreachable
in one cycle. Recorded commands, before the plant's physical clip, are checked
against both physical constraints and the complete regulated dynamic window.

The test2 acceleration part crosses transition lengths 1.0, 0.6, 0.4, 0.3,
and 0.2 m with simultaneous x/y/yaw acceleration multipliers 0.25, 0.5, 0.75,
1.0, 1.5, and 2.0. At 0.3 m, a separate sweep changes only yaw acceleration by
0.25, 0.5, 1.0, and 2.0. Test4 adds Scaled VP wherever Clipped VP is used;
the auxiliary velocity-and-acceleration variant is limited to test2.

The publication grid uses Clipped VP, Scaled VP and DWVP only, with one panel
per transition length and quantity: post-transition orientation overshoot, the
time integral of absolute orientation error, and maximum absolute orientation error.
Its horizontal coordinate is `omega_max / (a_omega * T)`, using 0.6 rad/s and
the nominal preview time `T = 0.75 s`. The six acceleration multipliers map to
5.33, 2.67, 1.78, 1.33, 0.89 and 0.67. A dotted line at 2 corresponds to
0.4 rad/s² under the constant-T, maximum-yaw-rate braking approximation;
the actual adaptive orientation time can differ. Panels are
`test2/acceleration_ratio_ramp_{1,0p6,0p4,0p3,0p2}_{overshoot,heading_integral,heading}.{pdf,png}`,
with `acceleration_ratio_legend.{pdf,png}`. The 0.3 m time series at multipliers
0.25 and 1 are `ramp_0p3_acceleration_{0p25,1}_{speed,yaw,signed_heading}.{pdf,png}`,
with `acceleration_time_series_legend.{pdf,png}`. They extend beyond the ramp
through the goal criterion (or timeout), including terminal settling. Superseded acceleration figures are retained in each test's
`previous_acceleration_figures/` directory.

All Test 2 publication figures use the same three methods; `vp_scaled_accel`
remains in the saved data and validation. To refresh the default-setting figures
and archived multiplier panels from existing CSV/NPZ files only, use the
repository-local cache variables above and run:

```bash
uv run --offline --locked --python 3.11.11 python tools/refresh_test2_figures.py results/access_v2
uv run --offline --locked --python 3.11.11 python tools/validate_access_v2.py results/access_v2
```

This command preserves CSVs, trajectories, `REPORT.md`, and unaffected figures.
The manifest retains the simulation source hash and records the figure refresh.

Paths are 4 m long with 0.005 m spacing. Orientation ramps start at x=1 m; a zero
transition length creates a true step with duplicate position samples. Error
metrics now use the whole run: every saved pose from t=0 through the cycle
at which the simulator's goal criterion is met, including terminal settling.
The criterion requires position and orientation tolerances plus applied velocity
components at most 0.001 in magnitude; it is not the earlier first entry into
the goal tolerance. Timed-out runs contribute errors through the timeout and
retain their timeout flags. This replaces the former x≤3.25 m rule (and the
2.995 m cutoff for the 1 m approach-distance sweep). No spatial mask excludes
reverse progress or motion beyond the goal. `evaluation_end_m` is now empty;
`evaluation_interval` is `start_to_goal_or_timeout`, and `evaluation_complete`
means successful goal arrival. Ray/box nonintersection is evaluated
for all methods independently of whether their solver calls the projection
branch; terminal cycles are excluded from this count. Travel
distance is the integrated translation norm, not x progress. The convergence
metrics include first entry into the 2% band and entry followed by remaining in
that band through goal arrival; timeouts and runs ending outside the band have
undefined settling times. Crossing includes the complete saved motion. Minimum transition speed includes the
ramp and its preceding instantaneous preview length. Full duration includes
terminal settling. Position error is distance to the closest path segment.
Every condition records maximum, time-mean and time-integrated absolute position
and orientation errors. Integrals use trapezoids between every adjacent pair of
saved samples, including the initial and final pose; means divide by
`eval_duration_s`, which equals the full saved duration. There is no boundary
extrapolation. Arithmetic sample means over the same full run are retained as `eval_sample_mean_position_error_m` and
`eval_sample_mean_heading_error_deg`. `command_constraint_violation_pct` is the
percentage of all command cycles violating either the physical velocity or
acceleration bounds; `travel_time_s` keeps its full-run, success-only definition.
Physical command excess and regulated-cap excess have
separate magnitudes and durations; per-axis physical excess uses the respective
velocity or acceleration units. Regulated-cap excess remains in CSV only; it is
excluded from reports and figures. Crossing detection uses a 0.000001 m threshold.

Signed orientation error is `wrap(robot yaw - projected reference yaw)` in the saved
`signed_yaw_errors` array (radians). For the positive 90-degree ramps, positive
error is lead and negative error is lag. CSV columns `eval_max_heading_lead_deg`
and `eval_max_heading_lag_deg` record their nonnegative maxima over the same
full-run evaluation interval. `post_transition_heading_overshoot_deg` records the
maximum positive excess over final yaw after x exceeds the end of the ramp,
through goal arrival or timeout; it is missing if the ramp is not passed.
`transition_heading_lag_deg` restricts lag to the changing-reference interval;
for a zero-length step it uses the first outgoing sample. The existing
`eval_max_heading_lag_deg` remains the maximum over the whole evaluation window.
Test1 and the offset cases in test4 report `crossing_m` as position overshoot
and `settling_2pct_time_s` as settling time;
test2 and the ramp cases in test4 report orientation lag and post-transition overshoot.
`acceleration_time_vx_s`, `acceleration_time_vy_s`, and `acceleration_time_w_s`
are the maximum absolute physical axis velocity limits divided by their
acceleration limits (missing for zero acceleration). `lookahead_time_s` is the
configured adaptive preview time; `fixed_lookahead_m` identifies conditions
where fixed distance overrides it.

To re-aggregate all saved Access trajectories without rerunning dynamics:

```bash
mkdir -p /tmp/dwvp-full-run
export UV_CACHE_DIR="$PWD/build/uv-cache" MPLCONFIGDIR=/tmp/dwvp-full-run/matplotlib TMPDIR=/tmp/dwvp-full-run PYTHONDONTWRITEBYTECODE=1
uv run --offline --locked --python 3.11.11 python tools/recompute_access_metrics.py results/access_v2 --baseline results/access_v2/trials/full_run_baseline/manifest.json
uv run --offline --locked --python 3.11.11 pytest -q --basetemp=/tmp/dwvp-full-run/pytest
uv run --offline --locked --python 3.11.11 python tools/validate_access_v2.py results/access_v2
```

The baseline manifest is an immutable local copy of the old evaluation, kept
alongside the ignored trial histories. `full_run_changes.csv` records every
changed existing metric (using its original condition ID), and
`full_run_evaluation.json` records the baseline and trajectory SHA-256 values.
The manifest retains each original trial specification and trial ID as simulation
provenance, while condition IDs describe the new full-run evaluation.
`REPORT.md` includes old → new manuscript tables; `regenerated_files.txt` lists
all regenerated artifacts, including the archived acceleration panels.
Travel times, constraint shares, near-obstacle speeds, ramp-local lag, and the
matched-travel-time search retain their definitions. All figure labels use
“orientation” and angular-velocity axes use “Angular velocity [rad/s]”; filenames
and the three-method Test 2 publication figure set are preserved.

Time matching separately scales Clipped VP's and Scaled VP's x/y box, demand and nominal speed request by
the same factor, leaving acceleration and yaw limits fixed. Its sequential search
retains every candidate, including timeouts, and records achieved timing error
under `time_matches.vp` and `time_matches.vp_scaled` in the manifest.
A tolerance of one control period is used. Test2 crosses interval length and
acceleration; preview × noise is included in `all` as part of Test 4(c). The representative conditions are
initial offset 0.50 m and orientation intervals 1.0 m and 0.3 m. All approach
distances in the RPP sweep are positive (0.1, 0.3, 0.6 and 1.0 m). Noise is independent
Gaussian observation noise each cycle; true pose and measured velocity remain
exact. Noise seeds 0–19 are paired across methods and conditions. Zero-noise
seeds repeat the deterministic case and are not independent replications.

Each output directory contains a per-trial `summary.csv` and PDF/PNG figures.
`test4/noise_summary.csv` and `preview-noise/nominal_selection.csv` contain means, sample
standard deviations, metric-wise finite counts, successes, failures, timeouts and
evaluation completion counts. Statistics include observed failed/timeout runs;
undefined metrics remain missing. The report covers all 60 preview/noise
combinations (eight fixed distances 0.055–0.6 m, four adaptive times 0.5–1.5 s,
five paired position/yaw noise levels) and all three representative paths.
Run the grid alone with `uv run --offline --locked
--python 3.11.11 dwvp-study preview-noise --config configs/access_v2.yaml
--output results/preview-noise --seed 0 --workers 4`.

Test 4(c) includes a noise × preview table for each scene, with mean travel times
and success counts. The three `preview-noise/*_travel_time_vs_fixed_lookahead`
PDF/PNG panels use one line per noise level and same-color dotted references
at `L = V sigma_xy / (ay dt) = 30 sigma_xy`; their horizontal legend is separate.
This estimate compares lateral velocity jitter with the one-cycle acceleration
allowance; it omits the difference of independent noise samples, yaw noise and
terminal behavior. `travel_time_recovery.csv` records the smallest sampled fixed
distance whose mean time is at most 1.1 times the zero-noise mean in the same
scene, using both the **nominal adaptive preview** (`reference=nominal_preview`)
and the **same fixed distance** (`reference=same_fixed_lookahead`) as references,
requiring 20/20 successes in both conditions. Missing
recovery stays blank. This is a sampled threshold, not a guarantee. Adaptive
preview retains its existing 0.11–0.33 m limits.

`manifest.json` records all planned conditions, full configurations, seeds, source
and dependency hashes, outcomes and elapsed time. Partial runs use
`manifest_testN.json` or `manifest_preview-noise.json`. The manifest is checkpointed while running; individual
trials save immediately under ignored `trials/<hash>/`. Cache identity includes
initial pose, path, evaluation rules, environment and numerical source hash;
changes only to plotting/reporting can reuse numerical trajectories. The final
manifest also records the full source hash. Exceptions become explicit `error`
rows with tracebacks in trial metadata. `issues.csv` lists failed or physically
violating conditions.
`REPORT.md` is a generated factual handoff with old/new manuscript tables, a
complete regenerated-file list, validation results, and common
metric columns across tests. Test4's compact table averages per-condition
metrics; individual conditions and noise standard deviations remain in CSV.
Settling time is also shown in the test1 and test4 tables; undefined values
remain missing when averaging conditions. An optional
`--baseline /path/to/previous/manifest.json` matches trajectory specifications
across test renumbering and writes `method_changes.csv` and a manifest audit.
If old `trials/` are present, it also compares mean position error
and goal overshoot reconstructed from those histories. Other unchanged numeric
metrics use an absolute comparison tolerance of 1e-10. The comparison also
counts preserved original condition IDs and lists any missing original conditions.
The snapshot taken before adding the scaled baselines is retained locally at
`build/scaled-vp-task/prechange/manifest.json`; pass that path to `--baseline`
to reproduce the before/after comparison in the saved report.

The initial round-2 refresh reduced the preview cap from 0.33 to 0.165 m and
changed 962 conditions. The follow-up restores 0.33 m and corrects RPP's demand
to the regulated x-axis limit. `comment_round2_changes.csv` and the accompanying
`comment_round2.json`/`comment_round2_conditions.csv` retain the earlier audit.
The current audit is `comment_round2_followup.json`; its `_conditions.csv` checks
all matching original conditions and its `_restoration.csv` checks every metric
in the earlier change list against the original value (absolute tolerance 1e-10).
Both generations of trajectories remain intact. The old optional sweep values
remain in `regulation_sweep_previous.csv`; its old figures remain in `test4/`.
`lookahead_ranges.csv` reports full-run minima/maxima per test, separating adaptive
and fixed distances. Upper-active counts mean the uncapped adaptive distance
exceeded the bound by more than 1e-10; at-upper counts also include equality.
Counts sum all condition entries, including repeated nominal settings and seeds.

The historical pre-round-2 snapshot is under the ignored
`results/access_v2/trials/comment_round2_baseline/`; the follow-up entry-state
snapshot is `build/comment_round2_followup/`. The current figure-only refresh
uses `results/access_v2/trials/test2_grid_baseline/` and performs no simulations:

```bash
uv run --offline --locked --python 3.11.11 python tools/refresh_test2_grid.py results/access_v2
uv run --offline --locked --python 3.11.11 python tools/validate_access_v2.py results/access_v2
uv run --offline --locked --python 3.11.11 python tools/validate_access_v2.py results/access_v2_acceleration_sweep --study acceleration-sweep
```

Reuse requires unchanged simulation inputs, allowing an equal cap or a tighter,
inactive cap. The restored profile reuses the original 0.33 m-cap histories.
Saved trajectories are copied without changing their bytes,
their metrics are re-evaluated, and their origin is recorded in trial metadata.
Test3 and active-cap conditions are simulated. Cache hits avoid repeating
completed work. The snapshot and per-trial histories are local artifacts.
The grid refresh preserves all 1,427 saved conditions across the main suite
(1,327) and `results/access_v2_acceleration_sweep` (100). Only suite labels and
their condition IDs change; trial specifications, trial IDs and metrics are
unchanged. `test2_grid_refresh.json` records the source transition and hashes;
the earlier audit records retain their original source hashes. The refresh is
repeatable from its immutable local baseline. Run `acceleration-sweep` or
`regulation-sweep` with a separate `--output` directory to produce its own
manifest and report; neither is included in `all`.

The historical mean/lag alignment is documented in
`metrics_alignment_changes.csv` and `metrics_alignment.json`; those records
remain unchanged. The current re-aggregation command above performs full-run
evaluation and writes `full_run_changes.csv` and `full_run_evaluation.json`.
Trial IDs and specifications retain their original simulation identity;
`simulation_source_sha256` and `simulation_numerical_source_sha256` preserve
generation provenance, while the current source hashes identify the refreshed
metrics, report and figures. Trial metadata records `metrics_source_sha256`.

`test2/overshoot_prediction.csv` covers all simultaneous and angular-only
acceleration sweep conditions for VP, scaled VP and DWVP. It records omega
immediately before the first braking command whose preview target has final yaw,
that cycle's `T=max(0.20, k*lookahead/desired_translation_speed)`, and the frozen
prediction `max(0, omega**2/(2*aw)-omega*T)` beside measured post-ramp overshoot.
The final-yaw crossing occurs later, after braking, and is not the sampling event.
The prediction assumes constant T and maximum angular deceleration; the report
shows mismatches as well as matches. It is not a DWVP guarantee.

At the earlier alignment start, the purported all-zero overshoot column was already nonzero
in 535 conditions and matched direct trajectory calculations. The old report
omitted this column. The current files cannot establish the cause in an earlier
all-zero version; regression tests retain the supplied 0.3 m reference values
with their original 0.33 m cap.

Validate a completed run with `uv run --offline --locked --python 3.11.11 python tools/validate_access_v2.py results/access_v2`;
use `--study preview-noise` for the grid alone or `--study regulation-sweep` for the optional regulation sweep. Use the existing locked environment.
The saved results of this configuration are in `results/access_v2/`.

Figures require installed Times New Roman, use STIX math and PDF font type 42,
7–8 pt text, English axes, one panel per file and separate horizontal legends.
No title is drawn inside panels. Fixed-preview test1 runs can overlay the ideal
omni orbit and linear PP curve when the initial error is smaller than preview;
the nominal adaptive-preview figures do not overlay fixed-distance theory.

The test2 sweep figures show maximum orientation error versus transition length and
versus required angular velocity divided by its limit. Their reference line is
`vp_translation_speed * (pi/2) / w_max = 0.575959 m` (ratio 1); the separate step
condition is omitted from these finite-rate axes. The 0.3 m time series keeps the
horizontal prediction `w_max * 0.3 / (pi/2)`, with a separate legend file.
The acceleration sweep adds separate maximum orientation/position error panels for
each interval and a separate yaw-only sweep. At 0.3 m, nominal and half-acceleration
time series show translation speed, angular velocity, and signed orientation error for
Clipped VP, Scaled VP, and DWVP, with a separate three-method legend. The error and velocity panels show the whole run through the goal criterion
or timeout. The report compares every sweep cell without assuming DWVP is best.
