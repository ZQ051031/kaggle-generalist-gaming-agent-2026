import json
import os
import sys
from pathlib import Path

from build_notebook import ROOT, SOURCES, render


def test_notebook_matches_repo_and_runs_with_synthetic_kaggle_layout(monkeypatch, tmp_path):
    notebook = (ROOT / "solution.ipynb").read_text()
    assert notebook == render()

    inputs = tmp_path / "input"
    working = tmp_path / "working"
    dataset = inputs / "competitions/generalist-gaming-agent"
    protocol = dataset / "gga/protocol.py"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("""
from dataclasses import dataclass
from gaming_agent.mock_protocol import Observation, ToolSpec

@dataclass
class Action:
    tool: str
    params: dict
""")
    server = dataset / "kaggle_evaluation/gga/gga_inference_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("""
from pathlib import Path
from gga.protocol import Action, Observation
from agent import Agent

class GGAInferenceServer:
    def run(self):
        observation = Observation(b'', (), (), {}, 0)
        assert Agent().step(observation, []) == Action('skip', {})
        Path('submission.parquet').touch()
""")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("gga.protocol", "gga", "kaggle_evaluation.gga.gga_inference_server",
                 "kaggle_evaluation.gga", "kaggle_evaluation", "agent"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(os, "environ", os.environ.copy())

    namespace = {}
    for cell in json.loads(notebook)["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).replace('"/kaggle/input"', repr(str(inputs)))
        source = source.replace('"/kaggle/working"', repr(str(working)))
        exec(compile(source, cell["id"], "exec"), namespace)
    assert Path.cwd() == working
    assert (working / "submission.parquet").is_file()
    for filename in SOURCES:
        assert (working / filename).read_text() == (ROOT / filename).read_text()
