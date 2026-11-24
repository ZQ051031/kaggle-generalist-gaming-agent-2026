import argparse
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
SOURCES = ("agent.py", "gaming_agent/__init__.py", "gaming_agent/model.py",
           "gaming_agent/policy.py", "gaming_agent/protocol.py", "gaming_agent/vision.py")

CONFIG = """
import os
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
DATA_ROOT = None  # override with Path("/kaggle/input/...") if autodetection fails
MODEL_DIR = None  # optional: Path("/kaggle/input/.../model-weights")

if DATA_ROOT is None:
    candidates = (
        INPUT / "competitions/generalist-gaming-agent",
        INPUT / "generalist-gaming-agent",
        *(p.parent.parent for pattern in ("*/gga/protocol.py", "*/*/gga/protocol.py")
          for p in INPUT.glob(pattern)),
    )
else:
    candidates = (Path(DATA_ROOT),)

DATA_ROOT = next((p for p in candidates if (p / "gga/protocol.py").is_file()
                  and (p / "kaggle_evaluation/gga/gga_inference_server.py").is_file()), None)
if DATA_ROOT is None:
    raise FileNotFoundError("Attach the Generalist Gaming Agent competition data, or set DATA_ROOT.")

WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path[:0] = [str(WORK), str(DATA_ROOT)]
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (
    str(WORK), str(DATA_ROOT), os.environ.get("PYTHONPATH", ""),
)))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

import numpy
from PIL import Image
"""

MODEL = """
if MODEL_DIR is not None:
    import importlib.util
    import subprocess
    import time
    from urllib.request import ProxyHandler, build_opener

    model = Path(MODEL_DIR).resolve()
    if (not model.is_relative_to(INPUT.resolve()) or not model.is_dir()
        or importlib.util.find_spec("vllm") is None):
        print("Attached model directory or vLLM missing; using the CPU policy.")
    else:
        port, name = "8765", "gga-local"
        with (WORK / "model-server.log").open("w") as log:
            process = subprocess.Popen([
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model), "--served-model-name", name,
                "--host", "127.0.0.1", "--port", port,
                "--max-model-len", "4096", "--max-num-seqs", "1",
            ], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 180
        opener = build_opener(ProxyHandler({}))
        while process.poll() is None and time.monotonic() < deadline:
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=1):
                    os.environ.update(GGA_MODEL_NAME=name, GGA_MODEL_PORT=port)
                break
            except OSError:
                time.sleep(1)
        else:
            if process.poll() is None:
                process.kill()
                process.wait()
            print("Model server did not start; using the CPU policy. See model-server.log.")
"""


def _cell(cell_type, cell_id, source):
    cell = {"cell_type": cell_type, "id": cell_id, "metadata": {},
            "source": dedent(source).strip().splitlines(keepends=True)}
    if cell_type == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def render():
    code = ["# Generated from the repository sources by build_notebook.py.",
            "SOURCES = {"]
    for name in SOURCES:
        source = (ROOT / name).read_text()
        code.extend([f"    {name!r}: (", *(f"        {line!r}" for line in source.splitlines(keepends=True)),
                     "    ),"])
    code.extend(["}", "for relative, source in SOURCES.items():",
                 "    path = WORK / relative", "    path.parent.mkdir(parents=True, exist_ok=True)",
                 '    path.write_text(source, encoding="utf-8")'])
    cells = [
        _cell("markdown", "start", """
        # Generalist Gaming Agent
        Attach the **Generalist Gaming Agent** competition data and turn Internet off.
        Save & Run All. The normal run writes a placeholder; Kaggle scores the submission
        only during its competition rerun. The CPU policy needs no model attachments.
        """),
        _cell("code", "setup-data", CONFIG),
        _cell("code", "write-agent", "\n".join(code)),
        _cell("code", "optional-model", MODEL),
        _cell("code", "serve", """
        from kaggle_evaluation.gga.gga_inference_server import GGAInferenceServer

        server = GGAInferenceServer()
        server.run()
        """),
    ]
    return json.dumps({"cells": cells,
                       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                   "name": "python3"}},
                       "nbformat": 4, "nbformat_minor": 5}, indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    notebook = ROOT / "solution.ipynb"
    expected = render()
    if args.check:
        if notebook.read_text() != expected:
            raise SystemExit("solution.ipynb is stale; run python build_notebook.py")
        print("solution.ipynb matches the tracked sources")
    else:
        notebook.write_text(expected)
