import pytest

from gaming_agent import mock
from gaming_agent.mock import Inputs, normalized_score, run


def test_normalized_score_clips_to_floor_and_ceiling():
    assert normalized_score(1, 2, 10) == 0
    assert normalized_score(6, 2, 10) == 0.5
    assert normalized_score(20, 2, 10) == 1
    with pytest.raises(ValueError):
        normalized_score(5, 5, 5)


def test_skip_preserves_schedule_and_wait_cancels_timed_keys():
    inputs = Inputs()
    inputs.replace(0, ("sequence", {"steps": [
        {"at_ms": 0, "tool": "press_key", "params": {"key": "a", "duration_ms": 500}},
        {"at_ms": 100, "tool": "click", "params": {"x": 10, "y": 10}},
    ]}))
    assert inputs.event_batch(0) == [{"type": "keydown", "key": "a"}]
    inputs.replace(1, ("skip", {}))
    assert any(event["type"] == "click" for event in inputs.event_batch(2))
    inputs.replace(3, ("wait", {"duration_ms": 100}))
    assert inputs.event_batch(3) == [{"type": "keyup", "key": "a"}]
    assert not inputs.event_batch(10)


def test_browser_benchmark_runs_one_policy_across_games():
    results = run(seconds=4)
    assert [row["game"] for row in results] == ["aim", "catch", "maze"]
    assert results[0]["raw"] >= 10
    assert results[1]["raw"] >= 1
    assert results[2]["raw"] >= 1
    assert all(0 <= row["normalized"] <= 1 and row["p95_ms"] < 90_000 for row in results)


@pytest.mark.parametrize(("seconds_per_reply", "arrival_frame"), [(0.05, 4), (0.25, 8)])
def test_simulated_replies_land_at_the_next_decision_boundary(monkeypatch, seconds_per_reply, arrival_frame):
    class Page:
        frame = 0
        click_frames = []

        def evaluate(self, js, args):
            if "synthetic.start" in js:
                return {"image": "eA=="}
            frames = []
            for events in args:
                if any(event["type"] == "click" for event in events):
                    self.click_frames.append(self.frame)
                self.frame += 1
                frames.append({"frame": self.frame, "image": "eA==", "score": len(self.click_frames)})
            return frames

    class Policy:
        clock = 0.0

        def step(self, obs, specs):
            self.clock += seconds_per_reply
            if obs.elapsed_s == 0:
                return "click", {"x": 20, "y": 20}
            return "skip", {}

    page, policy = Page(), Policy()
    monkeypatch.setattr(mock, "perf_counter", lambda: policy.clock)
    mock._run_game(page, policy, "aim", 1.2, 23, True)
    assert page.click_frames == [arrival_frame]
