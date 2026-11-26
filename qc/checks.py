"""Per-episode quality checks.

Each check returns a list of issues: {"check", "severity" ("fail" | "warn"), "detail"}.
Checks that need batch context (spike size, lag, outliers) read it from a reference
built once per dataset by build_reference(), so a release can be checked against the
previous release's reference instead of against itself.
"""
import numpy as np

from .config import CONFIG

# ---------- helpers ----------

def runs(mask):
    """Start index and length of every run of True values."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    return starts, np.flatnonzero(d == -1) - starts


def issue(check, severity, detail):
    return {"check": check, "severity": severity, "detail": detail}


def arm_joints(n_joints, cfg):
    return [j for j in range(n_joints) if j not in cfg["gripper_joints"]]


def estimate_lag(state, action, cfg=CONFIG):
    """Frames by which state trails action (median over moving arm joints) and the residual."""
    max_lag = cfg["lag"]["max_lag"]
    lags, resid = [], []
    for j in arm_joints(state.shape[1], cfg):
        s, a = state[:, j], action[:, j]
        if np.ptp(a) < 0.1:          # joint barely moves: lag is not identifiable
            continue
        errs = []
        for k in range(-max_lag, max_lag + 1):
            if k >= 0:
                errs.append(np.mean(np.abs(s[k:] - a[: len(a) - k])))
            else:
                errs.append(np.mean(np.abs(s[: len(s) + k] - a[-k:])))
        best = int(np.argmin(errs))
        lags.append(best - max_lag)
        resid.append(errs[best])
    if not lags:
        return None, None
    return int(np.median(lags)), float(np.mean(resid))


def episode_features(ep, cfg=CONFIG):
    """Summary numbers per episode used for cross-episode distribution checks."""
    s, a = ep["state"], ep["action"]
    names = ep.get("joint_names") or [f"j{j}" for j in range(s.shape[1])]
    f = {}
    for j, n in enumerate(names):
        f[f"state_mean/{n}"] = float(np.mean(s[:, j]))
        f[f"state_std/{n}"] = float(np.std(s[:, j]))
    half = s.shape[1] // 2
    f["path_len/left_arm"] = float(np.abs(np.diff(a[:, :half], axis=0)).sum())
    f["path_len/right_arm"] = float(np.abs(np.diff(a[:, half:], axis=0)).sum())
    _, resid = estimate_lag(s, a, cfg)
    if resid is not None:
        f["tracking_residual"] = resid
    if "effort" in ep:
        for j, n in enumerate(names):
            f[f"effort_abs_mean/{n}"] = float(np.mean(np.abs(ep["effort"][:, j])))
    return f


def robust_z(x, med, mad):
    return 0.6745 * (x - med) / mad


# ---------- reference built from a batch ----------

def build_reference(episodes, cfg=CONFIG):
    def step_p999(key):
        per_ep = [np.percentile(np.abs(np.diff(e[key], axis=0)), 99.9, axis=0) for e in episodes]
        return np.median(per_ep, axis=0)

    lags = [estimate_lag(e["state"], e["action"], cfg)[0] for e in episodes]
    lags = [l for l in lags if l is not None]
    feats = [episode_features(e, cfg) for e in episodes]
    keys = sorted(set().union(*feats))
    feat_ref = {}
    for k in keys:
        v = np.array([f[k] for f in feats if k in f])
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad > 1e-9:
            feat_ref[k] = (med, mad)
    lengths = np.array([len(e["t"]) for e in episodes])
    len_med = float(np.median(lengths))
    return {
        "step_p999": {"state": step_p999("state"), "action": step_p999("action")},
        "lag_median": int(np.median(lags)) if lags else 0,
        "features": feat_ref,
        "length": (len_med, float(np.median(np.abs(lengths - len_med)))),
    }


# ---------- checks ----------

def check_finite(ep, ref, fps, cfg):
    out = []
    for key in ("state", "action", "effort", "t"):
        if key in ep:
            bad = ~np.isfinite(ep[key])
            if bad.any():
                out.append(issue("finite", "fail", f"{int(bad.sum())} NaN/inf values in {key}"))
    return out


def check_timing(ep, ref, fps, cfg):
    out = []
    t = ep["t"]
    dt = np.diff(t)
    period = 1.0 / fps
    if (dt < 0).any():
        out.append(issue("timing", "fail", f"timestamps go backwards {int((dt < 0).sum())} times"))
    gaps = dt > cfg["timing"]["gap_factor"] * period
    if gaps.any():
        i = int(np.flatnonzero(gaps)[0])
        out.append(issue("timing", "fail",
                         f"{int(gaps.sum())} gaps; first at frame {i}: {dt[i]*1000:.0f} ms vs {period*1000:.0f} ms"))
    pos = dt[dt > 0]
    if len(pos):
        drift = np.median(pos) / period - 1
        if abs(drift) > cfg["timing"]["fps_drift_tol"]:
            out.append(issue("timing", "fail", f"median frame period off by {drift*100:+.1f}% (fps drift)"))
    fi = ep["frame_index"]
    if not np.array_equal(fi, np.arange(len(fi))):
        out.append(issue("timing", "fail", "frame_index is not 0..n-1 contiguous (dropped or repeated frames)"))
    return out


def check_duplicates(ep, ref, fps, cfg):
    out = []
    dt = np.diff(ep["t"])
    if (dt == 0).any():
        out.append(issue("duplicate", "fail", f"{int((dt == 0).sum())} repeated timestamps"))
    same = (np.diff(ep["state"], axis=0) == 0).all(1) & (np.diff(ep["action"], axis=0) == 0).all(1)
    _, lengths = runs(same)
    longest = int(lengths.max()) if len(lengths) else 0
    if longest > cfg["duplicate"]["max_identical_rows"]:
        out.append(issue("duplicate", "warn", f"{longest + 1} bit-identical rows in a row"))
    return out


def check_frozen(ep, ref, fps, cfg):
    out = []
    s, a = ep["state"], ep["action"]
    for j in range(s.shape[1]):
        starts, lengths = runs(np.diff(s[:, j]) == 0)
        for st, ln in zip(starts, lengths):
            if ln + 1 < cfg["frozen"]["min_run"]:
                continue
            rng = np.ptp(a[st: st + ln + 1, j])
            if rng > cfg["frozen"]["min_action_range"]:
                out.append(issue("frozen", "fail",
                                 f"{_name(ep, j)} reading frozen for {ln+1} frames while its action moved {rng:.2f} rad"))
                break
    return out


def check_saturation(ep, ref, fps, cfg):
    out = []
    s, a = ep["state"], ep["action"]
    over = cfg["saturation"]["overshoot"]
    for j in arm_joints(s.shape[1], cfg):
        for extreme, sign in ((s[:, j].max(), 1), (s[:, j].min(), -1)):
            starts, lengths = runs(s[:, j] == extreme)
            for st, ln in zip(starts, lengths):
                if ln < cfg["saturation"]["min_run"]:
                    continue
                beyond = sign * (a[st: st + ln, j] - extreme)
                if beyond.max() > over:
                    out.append(issue("saturation", "warn",
                                     f"{_name(ep, j)} pinned at {extreme:.3f} for {ln} frames while action asked {beyond.max():.2f} rad further"))
                    break
    return out


def check_spikes(ep, ref, fps, cfg):
    out = []
    for key in ("state", "action"):
        steps = np.abs(np.diff(ep[key], axis=0))
        thr = np.maximum(cfg["spike"]["factor"] * ref["step_p999"][key], cfg["spike"]["floor"])
        hit = steps > thr
        if hit.any():
            f, j = np.argwhere(hit)[0]
            out.append(issue("spike", "fail",
                             f"{int(hit.sum())} jumps in {key}; first {_name(ep, j)} at frame {f}: {steps[f, j]:.3f} rad (limit {thr[j]:.3f})"))
    return out


def check_lag(ep, ref, fps, cfg):
    lag, _ = estimate_lag(ep["state"], ep["action"], cfg)
    if lag is None:
        return []
    if abs(lag - ref["lag_median"]) > cfg["lag"]["tolerance"]:
        return [issue("lag", "fail",
                      f"state trails action by {lag} frames vs batch median {ref['lag_median']} (teleop/sync shift)")]
    return []


def check_length(ep, ref, fps, cfg):
    med, mad = ref["length"]
    n = len(ep["t"])
    if mad == 0:
        if n != med:
            return [issue("length", "warn", f"{n} frames; every other episode has {int(med)}")]
        return []
    z = robust_z(n, med, mad)
    if abs(z) > cfg["outlier"]["length_robust_z"]:
        return [issue("length", "warn", f"{n} frames, robust z {z:+.1f}")]
    return []


def check_distribution(ep, ref, fps, cfg):
    feats = episode_features(ep, cfg)
    worst = []
    for k, v in feats.items():
        if k in ref["features"]:
            z = robust_z(v, *ref["features"][k])
            if abs(z) > cfg["outlier"]["robust_z"]:
                worst.append((abs(z), k, z))
    if not worst:
        return []
    worst.sort(reverse=True)
    top = ", ".join(f"{k} z={z:+.1f}" for _, k, z in worst[:3])
    return [issue("distribution", "warn", f"{len(worst)} features beyond |z|>{cfg['outlier']['robust_z']:g}: {top}")]


def _name(ep, j):
    names = ep.get("joint_names")
    return names[j] if names else f"joint{j}"


CHECKS = {
    "finite": check_finite,
    "timing": check_timing,
    "duplicate": check_duplicates,
    "frozen": check_frozen,
    "saturation": check_saturation,
    "spike": check_spikes,
    "lag": check_lag,
    "length": check_length,
    "distribution": check_distribution,
}


def run_checks(ep, ref, fps, cfg=CONFIG):
    issues = check_finite(ep, ref, fps, cfg)
    if issues:  # NaN/inf poisons every later statistic, so stop here
        return "fail", issues
    for name, fn in CHECKS.items():
        if name == "finite":
            continue
        issues += fn(ep, ref, fps, cfg)
    sev = {i["severity"] for i in issues}
    verdict = "fail" if "fail" in sev else "warn" if "warn" in sev else "pass"
    return verdict, issues
