# Generalist Gaming Agent

An offline, screenshot-only baseline for Kaggle's
[Generalist Gaming Agent](https://www.kaggle.com/competitions/generalist-gaming-agent)
competition. It uses the same policy across games and does not need a GPU,
model weights, game IDs or hidden game state. See the
[competition fact check](docs/COMPETITION.md) for the verified interface and deadlines.

## Test locally without Kaggle data

With Python 3.11+:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium  # omit if Chrome or Chromium is already available
python -m pytest -q
python -m gaming_agent.mock --seconds 20 --seed 23
```

Pass `--chrome /path/to/chrome` or set `GGA_CHROME` to select a browser. Playwright
runs three offline HTML Canvas games: target clicking, ball catching, and coin
collection around a maze wall. The simulator captures JPEGs, maps actions to
browser input events, and advances at 20 logical frames per second with four
frames per decision. The default clock is deterministic; use
`--simulated-latency` to model measured inference delays crossing decision
boundaries. It approximates, rather than replaces, Kaggle's real-time evaluator.
Its clipped 0–1 scores use hand-set synthetic reference rates, **not** human
measurements or real competition scores. Latency excludes browser/screenshot
transport and is not a Kaggle GPU benchmark.

## Run with the official data and submit

1. Sign into your own Kaggle account, join through the
   [competition page](https://www.kaggle.com/competitions/generalist-gaming-agent),
   and accept the rules. Then download the public practice dataset from the
   [Data tab](https://www.kaggle.com/competitions/generalist-gaming-agent/data) if
   you want to run local practice. Do not use private test data.
2. **Optional local practice:** extract the dataset into `data/generalist-gaming-agent`
   so `gga/protocol.py` is directly under that directory. Install the mock
   dependencies and Chromium as above. Then run Kaggle's provided game harness
   from the dataset root (it plays the game but does not report a score):

   ```sh
   REPO="$(pwd)"
   cd data/generalist-gaming-agent
   PYTHONPATH="$REPO:$PWD" GGA_HEADLESS=1 python -m gga.game \
     --game off_line --agent "$REPO/agent.py" --minutes 1
   ```

3. Import [`solution.ipynb`](solution.ipynb) as a Kaggle Notebook, **Add Input**
   → **Competition** → **Generalist Gaming Agent**, and set **Internet** to
   **Off**. The notebook automatically locates the dataset; set `DATA_ROOT`
   in its setup cell if your attachment has another layout. CPU is sufficient
   for the default policy. No installs or downloads run in the notebook.
4. **Save & Run All**. The notebook writes `/kaggle/working/agent.py` and calls
   the official `GGAInferenceServer().run()` in its last cell unconditionally.
   A regular notebook run should write the required placeholder
   `submission.parquet`. In the view for that saved version, click **Submit to
   Competition**. Only the Kaggle rerun against all 32 games computes a score.

The competition package is gated. This repo uses the public Python signature;
the real protocol and notebook submission should be smoke-tested with your
attached data before relying on them. No Kaggle login or submission was made
to build this baseline.

## How the policy works

- Brief parsing identifies known control styles, objective/player colors,
  movement constraints and obstacles. Per-game state resets when the elapsed
  time rolls back. No game names or sample-specific logic are used by the agent.
- The CPU path downsamples JPEGs, compares recent frames, extracts connected
  color blobs and infers basic targeting, steering and obstacle avoidance.
  When it cannot ground an objective, it explores legal controls and uses
  an optional exposed score or approximate visual HUD changes as feedback.
  That visual signal is a heuristic, not OCR or a reliable score detector.
- Tool proposals are checked against viewport bounds and the available tool
  specifications. The fallback cannot infer complex game mechanics or menus
  reliably; the synthetic results are not estimates for hidden games.
- **Optional VLM:** attach compatible, publicly available weights and any
  offline wheels needed for a working `vllm` install. Set `MODEL_DIR` in the
  notebook and use a GPU. The optional launcher starts a loopback server from
  that local directory and gives readiness up to 180 seconds. The policy never
  blocks on generation, keeps one model request in flight, discards stale
  responses, and falls back to the CPU policy if model output is unavailable
  or invalid. The optional path has not been benchmarked on Kaggle hardware.

After changing shipped Python sources, run `python build_notebook.py` to
refresh the self-contained notebook. `python build_notebook.py --check` checks
that its embedded sources match the repository.
