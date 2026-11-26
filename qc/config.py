"""QC thresholds. Every number here is a decision; the README explains why each one was picked.

The config is hashed into the release manifest, so changing a threshold changes the release id.
"""
import hashlib
import json

CONFIG = {
    "timing": {
        "gap_factor": 1.5,          # a step longer than 1.5 frame periods is a dropped frame
        "fps_drift_tol": 0.01,      # median dt may differ from 1/fps by at most 1%
    },
    "duplicate": {
        "max_identical_rows": 40,   # state+action bit-identical for more than 40 rows in a row (0.8 s at 50 fps)
    },
    "frozen": {
        "min_run": 25,              # a joint reading exactly constant for >= 25 frames (0.5 s) ...
        "min_action_range": 0.10,   # ... while the commanded action for that joint moved > 0.10 rad (sweep in README)
    },
    "saturation": {
        "min_run": 10,              # state pinned at its own episode extreme for >= 10 frames ...
        "overshoot": 0.05,          # ... while the action asks for > 0.05 rad beyond it
    },
    "spike": {
        "factor": 4.0,              # a one-step change > 4x the reference p99.9 step size ...
        "floor": 0.05,              # ... and > 0.05 rad in absolute terms
    },
    "lag": {
        "max_lag": 15,              # search action->state lag in 0..15 frames
        "tolerance": 3,             # episode lag may differ from the batch median by <= 3 frames
    },
    "outlier": {
        "robust_z": 6.0,            # |robust z| above 6 on any per-episode feature -> warn
        "length_robust_z": 4.0,
    },
    "gripper_joints": [6, 13],      # leader/follower gripper units differ, so value comparisons skip them
}


def config_hash(cfg=CONFIG):
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
