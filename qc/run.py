"""Run QC on every dataset in datasets.json, run fault injection, write the report + release manifest.

    python -m qc.run            # reads raw/<name>/, writes reports/, release/, site/data.js
"""
import hashlib
import json
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from .checks import runs, build_reference, episode_features, estimate_lag, robust_z, run_checks
from .config import CONFIG, config_hash
from .data import load_dataset
from .inject import FAULTS

ROOT = Path(__file__).resolve().parent.parent


def code_hash():
    h = hashlib.sha256()
    for p in sorted((ROOT / "qc").glob("*.py")):
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


# ---------- dataset-level checks (metadata and schema) ----------

def dataset_checks(ds):
    out = []
    eps, meta, fps = ds["episodes"], ds["ep_meta"].set_index("episode_index"), ds["fps"]
    n_rows = {e["episode_index"]: len(e["t"]) for e in eps}
    bad_len = [i for i, n in n_rows.items() if int(meta.loc[i, "length"]) != n]
    out.append({"check": "episode metadata length == parquet rows", "ok": not bad_len,
                "detail": f"{len(eps)} episodes, {len(bad_len)} mismatches"})
    span = (meta["dataset_to_index"] - meta["dataset_from_index"] - meta["length"]).abs().max()
    out.append({"check": "dataset_from/to_index spans match lengths", "ok": int(span) == 0,
                "detail": f"max mismatch {int(span)} rows"})
    vid_cols = sorted({c.split("/")[1] for c in meta.columns if c.startswith("videos/")})
    worst = 0.0
    for v in vid_cols:
        dur = meta[f"videos/{v}/to_timestamp"] - meta[f"videos/{v}/from_timestamp"]
        worst = max(worst, float((dur - meta["length"] / fps).abs().max()))
    out.append({"check": "video segment duration == length / fps (all cameras)", "ok": worst < 1 / fps,
                "detail": f"{len(vid_cols)} cameras, worst offset {worst*1000:.1f} ms"})
    S = np.vstack([e["state"] for e in eps])
    st = ds["stats"]["observation.state"]
    d_max = float(np.max(np.abs(np.array(st["max"]) - S.max(0))))
    d_mean = float(np.max(np.abs(np.array(st["mean"]) - S.mean(0))))
    out.append({"check": "stats.json matches recomputed state stats", "ok": d_max < 1e-4 and d_mean < 1e-3,
                "detail": f"max abs diff: max {d_max:.1e}, mean {d_mean:.1e}"})
    return out


def schema_notes(ds):
    info = ds["info"]
    names = ds["joint_names"]
    return {
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "fps": ds["fps"],
        "episodes": len(ds["episodes"]),
        "frames": int(sum(len(e["t"]) for e in ds["episodes"])),
        "has_effort": "effort" in ds["episodes"][0],
        "joint_names_generic": all(n.startswith("motor_") for n in names),
        "cameras": sorted(k.split(".")[-1] for k, v in info["features"].items() if v.get("dtype") == "video"),
    }


def effort_profile(ds):
    if "effort" not in ds["episodes"][0]:
        return None
    E = np.vstack([e["effort"] for e in ds["episodes"]])
    steps = np.unique(np.round(np.abs(np.diff(np.unique(E[:, 1]))), 3))
    return {
        "quantum": float(steps[0]) if len(steps) else None,
        "zero_fraction": {n: round(float((E[:, j] == 0).mean()), 3) for j, n in enumerate(ds["joint_names"])},
    }


# ---------- fault injection ----------

def inject_all(name, eps, ref, fps, clean_verdicts):
    rows = []
    for fault, (fn, target) in FAULTS.items():
        hit_target = hit_any_fail = flagged_any = n = 0
        material = material_hit = 0
        fired = Counter()
        for ep in eps:
            if clean_verdicts[ep["episode_index"]] == "fail":
                continue
            seed = zlib.crc32(f"{name}/{ep['episode_index']}/{fault}".encode())
            bad = fn(ep, np.random.default_rng(seed))
            verdict, issues = run_checks(bad, ref, fps)
            checks = {i["check"] for i in issues}
            if bad["state"].shape == ep["state"].shape:
                # how much did the fault actually change the signal? a freeze on a still joint changes ~nothing
                mag = max(np.nanmax(np.abs(bad["state"] - ep["state"])), np.nanmax(np.abs(bad["action"] - ep["action"])))
                if mag > 0.05:
                    material += 1
                    material_hit += target in checks
            n += 1
            hit_target += target in checks
            hit_any_fail += verdict == "fail"
            flagged_any += verdict != "pass"
            fired.update(checks)
        rows.append({
            "dataset": name, "fault": fault, "target_check": target, "episodes": n,
            "recall_target": hit_target / n, "recall_fail": hit_any_fail / n, "recall_flagged": flagged_any / n,
            "checks_fired": dict(fired),
            "material_episodes": material,
            "recall_target_material": (material_hit / material) if material else None,
        })
    return rows


def frozen_stat(ep, min_run=CONFIG["frozen"]["min_run"]):
    """Largest action range seen while some joint's reading stayed exactly constant for >= min_run frames."""
    s, a, worst = ep["state"], ep["action"], 0.0
    for j in range(s.shape[1]):
        starts, lengths = runs(np.diff(s[:, j]) == 0)
        for st, ln in zip(starts, lengths):
            if ln + 1 >= min_run:
                worst = max(worst, float(np.ptp(a[st:st + ln + 1, j])))
    return worst


def frozen_sweep(name, eps):
    """Recall on material random freezes vs alarms on untouched episodes, for several thresholds."""
    fn = FAULTS["freeze_joint_random"][0]
    clean = [frozen_stat(e) for e in eps]
    bad = []
    for e in eps:
        b = fn(e, np.random.default_rng(zlib.crc32(f"{name}/{e['episode_index']}/freeze_joint_random".encode())))
        if np.abs(b["state"] - e["state"]).max() > 0.05:
            bad.append(frozen_stat(b))
    out = []
    for thr in np.round(np.arange(0.02, 0.31, 0.01), 2):
        out.append({"threshold": float(thr), "recall": float(np.mean([x > thr for x in bad])),
                    "clean_alarm_rate": float(np.mean([x > thr for x in clean]))})
    return {"clean_max": max(clean), "n_clean": len(clean), "n_material": len(bad), "points": out}


def outlier_trace(ds, joint, picks, step=10):
    """Per-frame p10/p50/p90 of one joint over the non-picked episodes, plus the picked episodes' traces."""
    j = ds["joint_names"].index(joint)
    others = np.stack([e["state"][:, j] for e in ds["episodes"] if e["episode_index"] not in picks])
    q = np.percentile(others, [10, 50, 90], axis=0)[:, ::step]
    return {"joint": joint, "fps": ds["fps"], "step": step,
            "p10": np.round(q[0], 3).tolist(), "p50": np.round(q[1], 3).tolist(), "p90": np.round(q[2], 3).tolist(),
            "picks": {str(e["episode_index"]): np.round(e["state"][::step, j], 3).tolist()
                      for e in ds["episodes"] if e["episode_index"] in picks}}


# ---------- main ----------

def main():
    t0 = time.time()
    sources = json.loads((ROOT / "datasets.json").read_text())
    cfg_h, code_h = config_hash(), code_hash()
    report = {"config": CONFIG, "config_hash": cfg_h, "code_hash": code_h, "datasets": [], "injection": []}
    manifest = {"manifest_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "qc_config_hash": cfg_h, "qc_code_hash": code_h,
                "policy": "drop episodes with any fail; keep warn episodes but list their flags", "releases": []}

    for src in sources:
        name = src["repo"].split("/")[-1]
        ds = load_dataset(ROOT / "raw" / name)
        eps, fps = ds["episodes"], ds["fps"]
        ref = build_reference(eps)

        per_ep, verdicts = [], {}
        for ep in eps:
            v, issues = run_checks(ep, ref, fps)
            lag, resid = estimate_lag(ep["state"], ep["action"])
            feats = episode_features(ep)
            zmax = max((abs(robust_z(x, *ref["features"][k])), k) for k, x in feats.items() if k in ref["features"])
            verdicts[ep["episode_index"]] = v
            per_ep.append({"episode": ep["episode_index"], "frames": len(ep["t"]), "verdict": v, "issues": issues,
                           "lag_frames": lag, "tracking_residual": round(resid, 5),
                           "max_robust_z": round(zmax[0], 2), "max_z_feature": zmax[1]})

        counts = Counter(verdicts.values())
        report["datasets"].append({
            "name": name, **src, "schema": schema_notes(ds), "dataset_checks": dataset_checks(ds),
            "effort": effort_profile(ds), "lag_median_frames": ref["lag_median"],
            "counts": {k: counts.get(k, 0) for k in ("pass", "warn", "fail")}, "episodes": per_ep,
        })
        report["injection"] += inject_all(name, eps, ref, fps, verdicts)
        report["datasets"][-1]["frozen_sweep"] = frozen_sweep(name, eps)
        if name == "aloha_static_coffee":
            report["trace"] = outlier_trace(ds, "left_forearm_roll", {0, 22})

        manifest["releases"].append({
            "release_id": f"{name}@{src['revision'][:7]}+qc.{cfg_h[:8]}",
            "source": {"repo": src["repo"], "revision": src["revision"], "license": src["license"]},
            "counts": {"total": len(eps), "kept": sum(v != "fail" for v in verdicts.values()),
                       "dropped": counts.get("fail", 0), "kept_with_flags": counts.get("warn", 0)},
            "kept_episodes": [e["episode"] for e in per_ep if e["verdict"] != "fail"],
            "dropped": [{"episode": e["episode"], "reasons": [i["detail"] for i in e["issues"]]}
                        for e in per_ep if e["verdict"] == "fail"],
            "flagged": [{"episode": e["episode"], "reasons": [i["detail"] for i in e["issues"]]}
                        for e in per_ep if e["verdict"] == "warn"],
        })

    report["runtime_s"] = round(time.time() - t0, 1)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "release").mkdir(exist_ok=True)
    (ROOT / "reports/qc_report.json").write_text(json.dumps(report, indent=1))
    (ROOT / "release/manifest.json").write_text(json.dumps(manifest, indent=1))
    site = {"report": report, "manifest": manifest}
    (ROOT / "site/data.js").write_text("window.QC = " + json.dumps(site, separators=(",", ":")) + ";\n")
    print(f"done in {report['runtime_s']} s")
    for d in report["datasets"]:
        print(d["name"], d["counts"], "lag", d["lag_median_frames"])
    return report, manifest


if __name__ == "__main__":
    main()
