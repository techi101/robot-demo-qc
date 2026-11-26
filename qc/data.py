"""Load a LeRobot v3 dataset (parquet only) into one dict of numpy arrays per episode."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    frames = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True
    )
    ep_meta = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))], ignore_index=True
    )
    names = info["features"]["observation.state"]["names"]
    joint_names = names["motors"] if isinstance(names, dict) else names

    episodes = []
    for ep_id, g in frames.groupby("episode_index", sort=True):
        ep = {
            "episode_index": int(ep_id),
            "joint_names": list(joint_names),
            "t": g["timestamp"].to_numpy(np.float64),
            "frame_index": g["frame_index"].to_numpy(np.int64),
            "state": np.stack(g["observation.state"].to_numpy()).astype(np.float64),
            "action": np.stack(g["action"].to_numpy()).astype(np.float64),
        }
        if "observation.effort" in g:
            ep["effort"] = np.stack(g["observation.effort"].to_numpy()).astype(np.float64)
        episodes.append(ep)

    return {
        "fps": float(info["fps"]),
        "joint_names": list(joint_names),
        "info": info,
        "episodes": episodes,
        "ep_meta": ep_meta,
        "stats": json.loads((root / "meta/stats.json").read_text()),
    }
