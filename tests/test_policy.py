from io import BytesIO
from queue import SimpleQueue

from PIL import Image, ImageDraw

from gaming_agent.mock_protocol import Observation, tool_specs
from gaming_agent.model import LocalModel
from gaming_agent.policy import Policy
from gaming_agent.protocol import Actions, TOOLS
from gaming_agent.vision import Blob, route, thumbnail


def observation(description="", controls="", elapsed=0.0, objects=(), score=0, result=None):
    image = Image.new("RGB", (320, 240), "#101521")
    draw = ImageDraw.Draw(image)
    for color, bounds in objects:
        draw.rectangle(bounds, color)
    data = BytesIO()
    image.save(data, format="JPEG")
    return Observation(data.getvalue(), (), (),
                       {"description": description, "controls": controls, "width": 320,
                        "height": 240, "score": score},
                       elapsed, result)


def test_click_coordinates_and_schema():
    obs = observation("Click red targets", "Click with the mouse",
                      objects=[("red", (265, 143, 285, 163)), ("green", (152, 112, 168, 128))])
    tool, params = Policy().step(obs, tool_specs())
    assert tool == "click"
    assert abs(275 - params["x"]) <= 3
    assert abs(153 - params["y"]) <= 3


def test_steers_from_visible_player_to_target():
    obs = observation("Move the green paddle horizontally to catch yellow balls.", "Left and right arrows.",
                      objects=[("#37ff35", (77, 202, 113, 213)), ("yellow", (240, 120, 256, 136))])
    assert Policy().step(obs, tool_specs()) == ("press_key", {"key": "ArrowRight", "duration_ms": 200})


def test_route_avoids_crossing_visible_wall():
    obs = observation(objects=[("#586477", (147, 85, 173, 150))])
    assert route(thumbnail(obs.image), Blob(129, 116, "green", 20, 0),
                 Blob(221, 116, "yellow", 20, 0), 320, 240)[0] == 0


def test_maze_steering_checks_axis_even_when_diagonal_is_clear():
    obs = observation("Guide the green avatar around walls to collect yellow coins.", "Use arrow keys.",
                      objects=[("#37ff35", (102, 172, 118, 188)), ("yellow", (250, 76, 266, 92)),
                               ("#586477", (160, 172, 215, 191))])
    tool, params = Policy().step(obs, tool_specs())
    assert tool == "press_key"
    assert params["key"] == "ArrowUp"


def test_feedback_and_elapsed_rollback():
    policy = Policy()
    specs = tool_specs()
    policy.step(observation(controls="Press a or press b", elapsed=5), specs)
    first = policy.last_choice
    assert first[0] == "key"
    policy.step(observation(controls="Press a or press b", elapsed=5.2, score=1), specs)
    assert policy.feedback[first] == (1.0, 1)
    policy.step(observation(controls="Press Enter", elapsed=0), specs)
    assert policy.brief.keys == ("Enter",)
    assert not policy.feedback


def test_schema_supports_all_mock_tools_and_rejects_invalid_actions():
    actions = Actions(tool_specs(), 320, 240)
    key = {"key": "Space"}
    xy = {"x": 0, "y": 239}
    proposals = {
        "click": xy, "move": xy, "drag": {**xy, "to_x": 319, "to_y": 0, "duration_ms": 200},
        "scroll": {**xy, "delta_x": 0, "delta_y": -10},
        "press_key": {**key, "duration_ms": 50}, "hold_keys": {"keys": ["Space"], "duration_ms": 50},
        "key_down": key, "key_up": key, "wait": {"duration_ms": 100}, "skip": {},
        "sequence": {"steps": [{"at_ms": 0, "tool": "click", "params": xy}]},
    }
    assert proposals.keys() == TOOLS
    for tool, params in proposals.items():
        assert actions.accepts({"tool": tool, "params": params}) == (tool, params)
    assert actions.make("click", x=-1, y=5) is None
    assert actions.make("click", x=True, y=5) is None
    assert actions.make("click", x=7) is None
    assert actions.make("press_key", key="", duration_ms=100) is None
    assert actions.make("sequence", steps=proposals["sequence"]["steps"] * 201) is None


def test_model_freshness_and_schema_fallback():
    model = LocalModel(8000, "test")
    model.results = SimpleQueue()
    model.results.put((1, 2, {"tool": "click", "params": {"x": 2, "y": 2}}))
    assert model.poll(2, 3, "", b"", {}) is None
    model.results.put((2, 2, {"tool": "click", "params": {"x": 2, "y": 2}}))
    assert model.poll(2, 6, "", b"", {}) is None
    model.results.put((2, 6, {"tool": "click", "params": {"x": 2, "y": 2}}))
    assert model.poll(2, 8, "", b"", {})["tool"] == "click"

    class InvalidModel:
        def poll(self, *_):
            return {"tool": "click", "params": {"x": 500, "y": 300}}

    obs = observation(controls="Press Enter")
    assert Policy(InvalidModel()).step(obs, tool_specs())[0] == "press_key"
