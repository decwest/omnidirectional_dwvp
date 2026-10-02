"""Reproducible studies with content-addressed, resumable individual trials."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from .config import Config
from .geometry import Obstacle
from .paths import make_paths
from .simulation import Result, simulate

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"vp": "VP (adaptive, clipping)", "vp_min": "VP (fixed 0.11 m, clipping)",
          "vp_max": "VP (fixed 0.33 m, clipping)", "dwvp": "DWVP (adaptive)"}
PATH_LABELS = {"iros_docking": "IROS docking", "iros_curve": "IROS curve",
               "constant_heading_corner": "Constant-heading corner", "independent_heading_curve": "Independent-heading curve"}
# Same static scene in every cost/approach ablation and cost-parameter sweep.
OBSTACLES = (Obstacle(.55, -.45, .10), Obstacle(1.42, .55, .10))


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def code_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":",1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def sha(data): return hashlib.sha256(canonical_json(data).encode()).hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_trial(output, path_name, path, method_key, config, *, obstacles=(), seed=0, force=False, source_hash=None):
    method = "vp" if method_key.startswith("vp") else "dwvp"
    spec = dict(path=path_name, path_sha256=hashlib.sha256(path.tobytes()).hexdigest(), method=method,
                config=asdict(config), obstacles=[asdict(o) for o in obstacles], seed=seed,
                source_sha256=source_hash or code_hash(), numpy_version=np.__version__, python_version=sys.version.split()[0])
    identity = sha(spec)
    directory = output / "trials" / identity
    metadata = directory / "trial.json"
    if not force and metadata.exists() and (directory / "trajectory.npz").exists():
        saved = json.loads(metadata.read_text())
        with np.load(directory / "trajectory.npz") as arrays:
            result = Result({key: arrays[key].copy() for key in arrays}, saved["metrics"], saved["performance"])
    else:
        result = simulate(path, method, config, obstacles, seed)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "trajectory.npz", **result.arrays)
        saved = dict(spec=spec, trial_id=identity, metrics=result.metrics, performance=result.performance)
        metadata.write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
    row = dict(path=path_name, method=method_key, label=LABELS.get(method_key, method_key), seed=seed, trial_id=identity)
    row.update(result.metrics)
    return row, result, spec


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path, rows):
    lines = [r"\begin{tabular}{llrrrr}", r"\hline", r"Path / setting & Controller & Pos. [m] & Yaw [deg] & Time [s] & Success \\", r"\hline"]
    for row in rows:
        name = PATH_LABELS.get(row["path"], row["path"])
        if "ablation" in row: name = row["ablation"].replace("_", " ")
        method = row["method"].replace("_", r"\_")
        duration = f'{row["travel_time_s"]:.1f}' if row["success"] else "--"
        lines.append(f'{name} & {method} & {row["mean_position_error_m"]:.3f} & {row["mean_heading_error_deg"]:.1f} & {duration} & {int(row["success"])} \\\\')
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def execute(study, output, config, seed=0, force=False):
    from .plotting import mechanism_figures, obstacle_figures, sweep_figures
    paths = make_paths()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = code_hash()
    manifest = dict(schema_version=1, package="omnidirectional-dwvp", source_sha256=source_hash,
                    git_revision=git_revision(), python=sys.version.split()[0], numpy=np.__version__,
                    platform=platform.platform(), cpu_model=cpu_model(), seed=seed, config=asdict(config),
                    notes=["Deterministic kinematics; no stochastic noise in canonical studies.",
                           "VP and DWVP use identical regulated reachable boxes and downstream clipping.",
                           "Wall-clock timing is nondeterministic and stored separately.",
                           "Applied velocity equals measured velocity in this ideal plant; no actuator or localization error.",
                           "No external obstacle avoidance, replanning or collision stop is simulated.",
                           "HSR circular footprint assumption: 0.22 m radius; conservative swept-circle collision check."],
                    dependency_lock_sha256=hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
                    provenance_document="docs/provenance.md", trials=[])
    performance = []
    selected = ("mechanism", "obstacles", "sweeps") if study == "all" else (study,)
    for name in selected:
        out = output / name
        out.mkdir(exist_ok=True)
        rows, results = [], {}
        def run(path_name, method, cfg, obstacles=(), extra=None):
            row, result, spec = run_trial(output, path_name, paths[path_name], method, cfg,
                                         obstacles=obstacles, seed=seed, force=force, source_hash=source_hash)
            if extra: row.update(extra)
            rows.append(row)
            manifest["trials"].append(dict(study=name, trial_id=row["trial_id"], spec=spec, summary=row))
            performance.append(dict(study=name, trial_id=row["trial_id"], **result.performance))
            return result
        if name == "mechanism":
            for path_name in paths:
                for method in ("vp", "vp_min", "vp_max", "dwvp"):
                    cfg = replace(config, use_cost_regulation=False,
                                  fixed_lookahead=.11 if method=="vp_min" else .33 if method=="vp_max" else None)
                    results[(path_name, method)] = run(path_name, method, cfg)
            mechanism_figures(out, paths, results)
            write_table(out / "summary.tex", rows)
        elif name == "obstacles":
            for cost, approach in ((False, False), (True, False), (False, True), (True, True)):
                label = ("cost" if cost else "no_cost") + ("_approach" if approach else "_no_approach")
                cfg = replace(config, use_cost_regulation=cost, approach_distance=config.approach_distance if approach else 0.)
                for method in ("vp", "dwvp"):
                    results[(label, method)] = run("constant_heading_corner", method, cfg, OBSTACLES, {"ablation": label})
            obstacle_figures(out, paths["constant_heading_corner"], results, OBSTACLES, config)
            write_table(out / "summary.tex", rows)
        else:
            grid = {
                "lookahead_time": [.5, .75, 1., 1.5, 2., 3.],
                "fixed_lookahead": [.055, .11, .165, .22, .33, .44, .55],
                "acceleration_scale": [.5, .75, 1., 1.5, 2.],
                "lateral_acceleration": [.11, .22, .44],
                "cost_scaling_dist": [.2, .4, .6, .8],
                "cost_scaling_gain": [.25, .5, .75, 1.],
                "approach_distance": [0., .1, .3, .6, 1.],
            }
            for parameter, values in grid.items():
                cost_study = parameter.startswith("cost_")
                names = ("constant_heading_corner",) if cost_study else tuple(paths)
                for value in values:
                    overrides = {"use_cost_regulation": cost_study}
                    if parameter == "acceleration_scale": overrides.update(ax=config.ax*value, ay=config.ay*value, aw=config.aw*value)
                    elif parameter == "lateral_acceleration": overrides.update(ay=value)
                    else: overrides[parameter] = value
                    cfg = replace(config, **overrides)
                    for path_name in names:
                        for method in ("vp", "dwvp"):
                            run(path_name, method, cfg, OBSTACLES if cost_study else (), {"parameter": parameter, "value": value})
                print(f"completed sweep {parameter}", flush=True)
            (out / "grid.json").write_text(json.dumps(grid, indent=2)+"\n")
            sweep_figures(out, rows)
            # Compact overview; complete per-setting rows are in summary.csv.
            lines = [r"\begin{tabular}{lrrr}", r"\hline", r"Sweep & Trials & Successes & Collisions \\", r"\hline"]
            for parameter in grid:
                chosen = [row for row in rows if row["parameter"] == parameter]
                lines.append(f'{parameter.replace("_", " ")} & {len(chosen)} & {sum(row["success"] for row in chosen)} & {sum(row["collision"] for row in chosen)} \\\\')
            lines.extend([r"\hline", r"\end{tabular}"])
            (out / "summary.tex").write_text("\n".join(lines)+"\n")
        write_csv(out / "summary.csv", rows)
        (out / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
        print(f'{name}: {len(rows)} trials; {sum(row["success"] for row in rows)} successful; {sum(row["collision"] for row in rows)} collisions', flush=True)
    manifest["condition_entries"] = len(manifest["trials"])
    manifest["distinct_trajectories"] = len({trial["trial_id"] for trial in manifest["trials"]})
    # Per-study manifests coexist; manifest.json is the canonical full publication run.
    manifest_path = output / ("manifest.json" if study=="all" else f"manifest_{study}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    write_csv(output / ("performance.csv" if study=="all" else f"performance_{study}.csv"), performance)
    (output / "performance.json").write_text(json.dumps(performance, indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("all", "mechanism", "obstacles", "sweeps"), nargs="?", default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "paper")
    parser.add_argument("--config", type=Path, default=None, help="Flat Config mapping in YAML; defaults are the frozen paper profile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recompute instead of resuming matching content-addressed trials")
    args=parser.parse_args()
    config=Config()
    if args.config:
        import yaml
        overrides=yaml.safe_load(args.config.read_text())
        config=replace(config, **overrides)
    execute(args.study, args.output, config, args.seed, args.force)


if __name__ == "__main__": main()
