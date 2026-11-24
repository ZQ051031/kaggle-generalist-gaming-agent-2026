import argparse
import base64
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from .mock_protocol import Observation, tool_specs
from .policy import Policy
from .protocol import Actions, SKIP


FRAME_MS = 50
STRIDE = 4
GAMES = {
    "aim": ("Click red targets as quickly as you can.", "Click with the mouse.", 0, 4),
    "catch": ("Move the green paddle horizontally to catch yellow balls.", "Left and right arrows.", 0, 0.33),
    "maze": ("Guide the green avatar around walls to collect yellow coins.", "Use the arrow keys.", 0, 0.5),
}


def normalized_score(value, floor, human):
    if human <= floor:
        raise ValueError("baseline must exceed floor")
    return min(1.0, max(0.0, (value - floor) / (human - floor)))


class Inputs:
    def __init__(self):
        self.pending = defaultdict(list)
        self.timed = set()
        self.held = set()

    def _queue(self, frame, event, **params):
        self.pending[frame].append({"type": event, **params})

    def event_batch(self, frame):
        events = []
        for event in self.pending.pop(frame, []):
            key = event.get("key")
            kind = event["type"]
            if kind == "timed_down":
                self.timed.add(key)
                event["type"] = "keydown"
            elif kind == "timed_up":
                self.timed.discard(key)
                if key in self.held:
                    continue
                event["type"] = "keyup"
            elif kind == "keydown":
                self.held.add(key)
            elif kind == "keyup":
                self.held.discard(key)
                self.timed.discard(key)
            events.append(event)
        return events

    def replace(self, frame, action):
        if action == SKIP:
            return
        self.pending.clear()
        for key in sorted(self.timed):
            self._queue(frame, "timed_up", key=key)
        tool, params = action
        if tool == "sequence":
            for step in params["steps"]:
                self._schedule(frame + math.ceil(step["at_ms"] / FRAME_MS), step["tool"], step["params"])
        else:
            self._schedule(frame, tool, params)

    def _schedule(self, frame, tool, params):
        duration = max(1, math.ceil(params.get("duration_ms", 0) / FRAME_MS))
        if tool in ("key_down", "key_up"):
            self._queue(frame, "keydown" if tool == "key_down" else "keyup", key=params["key"])
        elif tool in ("press_key", "hold_keys"):
            for key in params.get("keys", [params.get("key")]):
                self._queue(frame, "timed_down", key=key)
                self._queue(frame + duration, "timed_up", key=key)
        elif tool in ("click", "move", "drag", "scroll"):
            x, y = params["x"], params["y"]
            point = {"clientX": x, "clientY": y}
            button = {"left": 0, "middle": 1, "right": 2}[params.get("button", "left")]
            self._queue(frame, "mousemove", **point)
            if tool == "click":
                for event in ("mousedown", "mouseup", "click" if button == 0 else "auxclick"):
                    self._queue(frame, event, **point, button=button)
            elif tool == "scroll":
                self._queue(frame, "wheel", **point, deltaX=params["delta_x"], deltaY=params["delta_y"])
            elif tool == "drag":
                self._queue(frame, "mousedown", **point)
                for i in range(1, duration + 1):
                    self._queue(frame + i, "mousemove", buttons=1,
                                clientX=x + (params["to_x"] - x) * i / duration,
                                clientY=y + (params["to_y"] - y) * i / duration)
                self._queue(frame + duration, "mouseup", clientX=params["to_x"], clientY=params["to_y"])


def _run_game(page, policy, game, seconds, seed, simulated_latency):
    description, controls, floor, human = GAMES[game]
    limit = max(STRIDE, round(seconds * 1000 / FRAME_MS / STRIDE) * STRIDE)
    seconds = limit * FRAME_MS / 1000
    specs = tool_specs()
    adapter = Actions(specs, 320, 240)
    inputs = Inputs()
    snapshot = page.evaluate("([game, seed]) => synthetic.start(game, seed)", [game, seed])
    image = base64.b64decode(snapshot["image"])
    frame = score = 0
    frames = ()
    previous_action = None
    latencies = []
    while frame < limit:
        info = {"description": description, "controls": controls, "score": score,
                "remaining_s": (limit - frame) * FRAME_MS / 1000, "width": 320, "height": 240}
        history = frames[:-1]
        if len(history) > 8:
            history = tuple(history[i * (len(history) - 1) // 7] for i in range(8))
        obs = Observation(image, frames[-STRIDE:], history, info, frame * FRAME_MS / 1000, previous_action)
        started = perf_counter()
        try:
            tool, params = policy.step(obs, specs)
            action = adapter.accepts({"tool": tool, "params": params})
        except Exception as error:
            raise RuntimeError(f"policy failed in {game} at frame {frame}") from error
        ms = (perf_counter() - started) * 1000
        latencies.append(ms)
        if ms >= 90_000:
            break
        previous_action = {"accepted": action is not None, "tool": tool}
        advance = STRIDE
        if simulated_latency:
            advance *= max(1, math.ceil(ms / (FRAME_MS * STRIDE)))
        else:
            inputs.replace(frame, action or SKIP)
        arrival = min(limit, frame + advance)
        snapshots = page.evaluate("batches => synthetic.advance(batches)",
                                  [inputs.event_batch(i) for i in range(frame, arrival)])
        frames = tuple(base64.b64decode(s["image"]) for s in snapshots)
        frame = snapshots[-1]["frame"]
        score, image = snapshots[-1]["score"], frames[-1]
        if simulated_latency and frame < limit:
            inputs.replace(frame, action or SKIP)
    latencies.sort()
    return {"game": game, "raw": score, "normalized": normalized_score(score, floor * seconds, human * seconds),
            "p95_ms": latencies[math.ceil(len(latencies) * .95) - 1]}


def run(seconds=20, seed=23, chrome=None, simulated_latency=False, policy=None, games=None):
    from playwright.sync_api import sync_playwright

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    policy = policy if policy is not None else Policy()
    binary = chrome or os.environ.get("GGA_CHROME")
    if not binary:
        system_chrome = Path("/opt/google/chrome/chrome")
        binary = str(system_chrome) if system_chrome.is_file() else (
            shutil.which("chromium") or shutil.which("google-chrome")
        )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=binary, headless=True)
        page = browser.new_page(viewport={"width": 320, "height": 240})
        page.goto(Path(__file__).with_name("mock_games.html").as_uri())
        return [_run_game(page, policy, game, seconds, seed + i, simulated_latency)
                for i, game in enumerate(games or GAMES)]


def main():
    parser = argparse.ArgumentParser(description="Run synthetic browser games (not Kaggle scoring).")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--chrome", help="optional Chrome/Chromium executable")
    parser.add_argument("--simulated-latency", action="store_true", help="model elapsed time as in-game delay")
    args = parser.parse_args()
    scores = run(args.seconds, args.seed, args.chrome, args.simulated_latency)
    print("Synthetic benchmark only (not a competition score)")
    for row in scores:
        print(f"{row['game']}: raw={row['raw']} normalized={row['normalized']:.3f} p95_step={row['p95_ms']:.2f} ms")
    print(f"Synthetic mean: {sum(row['normalized'] for row in scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
