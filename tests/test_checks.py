"""Tests on synthetic episodes, so they run without downloading anything."""
import copy

import numpy as np
import pytest

from qc.checks import build_reference, run_checks, runs
from qc.config import CONFIG, config_hash
from qc.inject import FAULTS

FPS = 50.0


def synthetic_episode(seed, n=500, joints=14, lag=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    phase = rng.uniform(0, 2 * np.pi, joints)
    freq = rng.uniform(0.1, 0.4, joints)
    action = 0.6 * np.sin(2 * np.pi * freq * t[:, None] + phase)
    state = np.vstack([np.repeat(action[:1], lag, 0), action[:-lag]])
    state = np.round(state + rng.normal(0, 0.002, state.shape), 3)  # encoder noise + quantization
    return {"episode_index": seed, "t": t, "frame_index": np.arange(n), "state": state, "action": action}


@pytest.fixture(scope="module")
def batch():
    eps = [synthetic_episode(s) for s in range(20)]
    return eps, build_reference(eps)


def test_clean_episodes_pass(batch):
    eps, ref = batch
    for ep in eps:
        verdict, issues = run_checks(ep, ref, FPS)
        assert verdict == "pass", issues


@pytest.mark.parametrize("fault", ["drop_frames", "duplicate_frames", "timestamp_jump", "fps_drift",
                                   "freeze_joint_moving", "spike_large", "nan_value", "action_lag",
                                   "saturate_joint", "unit_error"])
def test_injected_fault_is_caught_by_its_check(batch, fault):
    eps, ref = batch
    fn, target = FAULTS[fault]
    bad = fn(eps[3], np.random.default_rng(0))
    verdict, issues = run_checks(bad, ref, FPS)
    assert target in {i["check"] for i in issues}, issues
    assert verdict != "pass"


def test_injection_does_not_modify_input(batch):
    eps, _ = batch
    before = eps[5]["state"].copy()
    for fn, _ in FAULTS.values():
        fn(eps[5], np.random.default_rng(1))
    assert np.array_equal(eps[5]["state"], before)


def test_nan_short_circuits(batch):
    eps, ref = batch
    bad = FAULTS["nan_value"][0](eps[0], np.random.default_rng(0))
    verdict, issues = run_checks(bad, ref, FPS)
    assert verdict == "fail" and [i["check"] for i in issues] == ["finite"]


def test_runs_helper():
    starts, lengths = runs(np.array([0, 1, 1, 0, 1, 1, 1], bool))
    assert starts.tolist() == [1, 4] and lengths.tolist() == [2, 3]


def test_config_hash_tracks_thresholds():
    other = copy.deepcopy(CONFIG)
    other["spike"]["factor"] = 5.0
    assert config_hash(CONFIG) == config_hash(copy.deepcopy(CONFIG))
    assert config_hash(other) != config_hash(CONFIG)
