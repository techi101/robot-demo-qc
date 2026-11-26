"""Fault injection: corrupt copies of real episodes in known ways, then see which checks fire.

Each fault returns a new episode dict (the input is never modified) and names the check
that is expected to catch it. Faults are placed with a seeded RNG so results reproduce.
"""
import numpy as np

from .config import CONFIG


def _copy(ep):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()}


def _mid(rng, n, width):
    return int(rng.integers(n // 5, n - n // 5 - width))


def _arm(n_joints):
    return [j for j in range(n_joints) if j not in CONFIG["gripper_joints"]]


def drop_frames(ep, rng):
    """Lose 5 consecutive frames (USB hiccup, logger stall)."""
    e = _copy(ep)
    i = _mid(rng, len(e["t"]), 5)
    keep = np.r_[0:i, i + 5:len(e["t"])]
    for k in ("t", "frame_index", "state", "action", "effort"):
        if k in e:
            e[k] = e[k][keep]
    return e


def duplicate_frames(ep, rng):
    """3 consecutive rows each written twice (retry on write)."""
    e = _copy(ep)
    i = _mid(rng, len(e["t"]), 3)
    idx = np.r_[0:i, np.repeat(np.arange(i, i + 3), 2), i + 3:len(e["t"])]
    for k in ("t", "frame_index", "state", "action", "effort"):
        if k in e:
            e[k] = e[k][idx]
    return e


def timestamp_jump(ep, rng):
    """Clock jumps forward 0.25 s partway through (NTP step, clock re-sync)."""
    e = _copy(ep)
    i = _mid(rng, len(e["t"]), 1)
    e["t"][i:] += 0.25
    return e


def fps_drift(ep, rng):
    """Recorder actually ran 3% slow but the file claims nominal fps."""
    e = _copy(ep)
    e["t"] = e["t"] * 1.03
    return e


def freeze_joint_moving(ep, rng):
    """One joint sensor freezes for 1.5 s while that joint is being moved."""
    e = _copy(ep)
    s, a, w = e["state"], e["action"], 75
    best = None
    for j in _arm(s.shape[1]):
        for i in range(0, len(a) - w, 25):
            r = np.ptp(a[i:i + w, j])
            if best is None or r > best[0]:
                best = (r, j, i)
    _, j, i = best
    s[i:i + w, j] = s[i, j]
    return e


def freeze_joint_random(ep, rng):
    """Same freeze, but at a random joint and time, moving or not (the honest version)."""
    e = _copy(ep)
    j = int(rng.choice(_arm(e["state"].shape[1])))
    i = _mid(rng, len(e["t"]), 75)
    e["state"][i:i + 75, j] = e["state"][i, j]
    return e


def spike_large(ep, rng):
    """A 0.3 rad one-frame glitch in a commanded joint (teleop leader arm packet error)."""
    e = _copy(ep)
    j = int(rng.choice(_arm(e["action"].shape[1])))
    e["action"][_mid(rng, len(e["t"]), 1), j] += 0.3
    return e


def spike_small(ep, rng):
    """A 0.08 rad one-frame glitch: close to real motion steps, expected to be hard."""
    e = _copy(ep)
    j = int(rng.choice(_arm(e["action"].shape[1])))
    e["action"][_mid(rng, len(e["t"]), 1), j] += 0.08
    return e


def nan_value(ep, rng):
    """One state value is NaN (serialization bug)."""
    e = _copy(ep)
    e["state"][_mid(rng, len(e["t"]), 1), int(rng.integers(0, e["state"].shape[1]))] = np.nan
    return e


def action_lag(ep, rng):
    """Action stream shifted 10 frames late relative to state (sync between leader and follower lost)."""
    e = _copy(ep)
    e["action"] = np.vstack([np.repeat(e["action"][:1], 10, 0), e["action"][:-10]])
    return e


def saturate_joint(ep, rng):
    """The busiest joint's reading clips at 60% of its range (encoder range or calibration limit)."""
    e = _copy(ep)
    s = e["state"]
    j = max(_arm(s.shape[1]), key=lambda k: np.ptp(s[:, k]))
    s[:, j] = np.minimum(s[:, j], s[:, j].min() + 0.6 * np.ptp(s[:, j]))
    return e


def unit_error(ep, rng):
    """One joint logged in degrees instead of radians (mixed firmware versions)."""
    e = _copy(ep)
    j = max(_arm(e["state"].shape[1]), key=lambda k: np.ptp(e["state"][:, k]))
    e["state"][:, j] = np.degrees(e["state"][:, j])
    return e


def swap_arms(ep, rng):
    """Left and right arm columns swapped (wrong port mapping on one recording day)."""
    e = _copy(ep)
    h = e["state"].shape[1] // 2
    for k in ("state", "action"):
        e[k] = np.hstack([e[k][:, h:], e[k][:, :h]])
    return e


def untouched(ep, rng):
    """Control: no change. Anything that fires here is a false alarm or a real issue in the source."""
    return _copy(ep)


# fault name -> (function, check expected to catch it)
FAULTS = {
    "none (control)": (untouched, None),
    "drop_frames": (drop_frames, "timing"),
    "duplicate_frames": (duplicate_frames, "duplicate"),
    "timestamp_jump": (timestamp_jump, "timing"),
    "fps_drift": (fps_drift, "timing"),
    "freeze_joint_moving": (freeze_joint_moving, "frozen"),
    "freeze_joint_random": (freeze_joint_random, "frozen"),
    "spike_large": (spike_large, "spike"),
    "spike_small": (spike_small, "spike"),
    "nan_value": (nan_value, "finite"),
    "action_lag": (action_lag, "lag"),
    "saturate_joint": (saturate_joint, "saturation"),
    "unit_error": (unit_error, "distribution"),
    "swap_arms": (swap_arms, "distribution"),
}
