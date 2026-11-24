import re
from dataclasses import dataclass

import numpy as np

from .model import LocalModel
from .protocol import Actions, SKIP, field
from .vision import COLORS, analyze, hud_signature, route, thumbnail


ARROWS = ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown")
WASD = ("a", "d", "w", "s")


@dataclass(frozen=True, slots=True)
class Brief:
    text: str
    pointer: str | None
    direction: tuple
    keys: tuple
    player: str | None
    target: str | None
    dodge: bool
    axis: str | None
    walls: bool

    @classmethod
    def parse(cls, info):
        inner = field(info, "brief", info)
        description = str(field(inner, "description", ""))
        controls = str(field(inner, "controls", field(info, "controls", "")))
        text = f"{description}\n{controls}"
        lower = text.lower()
        input_text = controls.lower()
        pointer = ("drag" if re.search(r"\bdrag\b", lower)
                   else "click" if re.search(r"\b(click|tap|shoot with (the )?mouse)\b", lower)
                   else "move" if re.search(r"\b(mouse|cursor|pointer|hover)\b", lower) else None)
        direction = (WASD if re.search(r"\bwasd\b|\bw\s*[/,]\s*a\s*[/,]\s*s\s*[/,]\s*d\b", input_text)
                     else ARROWS if re.search(r"\barrow|\bleft\b.*\bright\b|←|→", input_text) else ())
        keys = list(direction)
        for token in ("Space", "Enter", "Escape", "Shift", "Control", "Tab"):
            if re.search(r"\b" + token.lower() + (r"(?:bar)?\b" if token == "Space" else r"\b"), input_text):
                keys.append(token)
        keys.extend(re.findall(r"\b(?:press|hold|key)\s+[`'\"\[]?([a-z0-9])\b", input_text))
        color = "(" + "|".join(COLORS) + ")"
        actor = r"(?:player|ship|paddle|avatar|car|character|square|cursor|basket|hero)"
        objective = r"(?:target|coin|gem|orb|food|ball|block|item|enemy|hazard|obstacle|dot|star)s?"
        player = re.search(r"\b" + color + r"\s+" + actor + r"\b", description.lower())
        target = re.search(r"\b" + color + r"\s+" + objective + r"\b", description.lower())
        axis = ("x" if re.search(r"\bhorizontal|left\s*(?:and|/|or)\s*right", lower)
                else "y" if re.search(r"\bvertical|up\s*(?:and|/|or)\s*down", lower) else None)
        return cls(text, pointer, direction, tuple(dict.fromkeys(keys)),
                   player[1] if player else None, target[1] if target else None,
                   bool(re.search(r"\b(dodge|avoid|evade)\b", description.lower())), axis,
                   bool(re.search(r"\b(walls?|maze)\b", description.lower())))


class Policy:
    def __init__(self, model=None):
        self.model = model if model is not None else LocalModel.from_env()
        self.game = 0
        self.elapsed = None
        self._reset()

    def _reset(self):
        self.game += 1
        self.turn = 0
        self.previous = self.hud = self.score = self.last_choice = self.last_target = None
        self.feedback = {}
        self.brief = None

    def _reward(self, info, hud, result):
        score = field(info, "score")
        if type(score) in (float, int) and type(self.score) in (float, int):
            signal = float(np.clip(score - self.score, -1, 1))
        elif self.hud is not None and hud is not None and self.hud.shape == hud.shape:
            signal = min(0.2, float(np.mean(self.hud != hud)))
        else:
            signal = 0.0
        self.score, self.hud = score, hud
        if self.last_choice is None:
            return
        if (field(result, "accepted") is False or field(result, "ok") is False
                or field(result, "status") in ("refused", "rejected", "invalid")):
            signal = -1.0
        total, count = self.feedback.get(self.last_choice, (0.0, 0))
        self.feedback[self.last_choice] = (total + signal, count + 1)

    def _choose(self, options):
        for action, choice in options:
            if action and choice is None:
                self.last_choice = None
                return action
        scored = []
        for action, choice in options:
            if action:
                total, count = self.feedback.get(choice, (0.0, 0))
                priority = total / max(count, 1) + 0.45 * np.sqrt(np.log(self.turn + 1) / (count + 1))
                scored.append((priority, action, choice))
        if scored:
            _, action, self.last_choice = max(scored, key=lambda row: row[0])
            return action
        self.last_choice = None
        return SKIP

    def _react(self, actions, blobs, image, width, height):
        brief = self.brief
        blobs = [b for b in blobs if b.y > height * .09]
        players = [b for b in blobs if b.color == brief.player]
        player = max(players, key=lambda b: b.area, default=None)
        targets = [b for b in blobs if b not in players and (not brief.target or b.color == brief.target)]
        anchor = (player.x, player.y) if player else (width // 2, height // 2)

        def rank(b):
            x, y = anchor
            distance = abs(b.x - x) / width + abs(b.y - y) / height
            repeated = self.last_target is not None and abs(b.x - self.last_target[0]) + abs(b.y - self.last_target[1]) < 12
            return 1.5 * b.motion - distance + min(b.area, 200) / 400 - (0.2 if repeated else 0)

        target = max(targets, key=rank, default=None)
        if brief.pointer == "drag":
            return actions.drag(player, target) if player and target else None
        if brief.pointer and target:
            self.last_target = (target.x, target.y)
            return actions.pointer(brief.pointer, target.x, target.y)
        if not (brief.direction and player and target):
            return None
        dx, dy = target.x - player.x, target.y - player.y
        if brief.walls and not (brief.axis or brief.dodge):
            waypoint = route(image, player, target, width, height)
            if waypoint is not None:
                dx, dy = waypoint
        if brief.dodge:
            dx, dy = -dx, -dy
            if brief.axis == "x" and (player.x < width * .13 or player.x > width * .87):
                dx = width / 2 - player.x
            if brief.axis == "y" and (player.y < height * .13 or player.y > height * .87):
                dy = height / 2 - player.y
        left, right, up, down = brief.direction
        choose_x = brief.axis == "x" or (not brief.axis and abs(dx) / width >= abs(dy) / height)
        distance = abs(dx) if choose_x else abs(dy)
        if distance < 5 and not brief.dodge:
            return actions.make("wait", duration_ms=200) or SKIP
        key = (left if dx < 0 else right) if choose_x else (up if dy < 0 else down)
        return actions.key(key, 100 if distance < 20 else 200)

    def step(self, obs, tools):
        elapsed = field(obs, "elapsed_s")
        if type(elapsed) in (int, float):
            if self.elapsed is not None and elapsed < self.elapsed:
                self._reset()
            self.elapsed = elapsed
        info = field(obs, "info", {})
        if self.brief is None:
            self.brief = Brief.parse(info)
        self.turn += 1
        jpeg = field(obs, "image")
        image = thumbnail(jpeg)
        if image is None:
            return SKIP
        width = field(info, "width", image.shape[1])
        height = field(info, "height", image.shape[0])
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            return SKIP
        actions = Actions(tools, width, height)
        self._reward(info, hud_signature(image), field(obs, "last_action"))
        history = field(obs, "history", ())
        earlier = thumbnail(field(history[-1], "image", history[-1])) if isinstance(history, (list, tuple)) and history else None
        blobs = analyze(image, earlier if earlier is not None else self.previous, width, height)
        self.previous = image
        reactive = self._react(actions, blobs, image, width, height)
        if self.model is not None:
            proposal = self.model.poll(self.game, self.turn, self.brief.text, jpeg, actions.specs)
            if proposed := actions.accepts(proposal):
                return self._choose([(proposed, None)])
        if reactive:
            return self._choose([(reactive, None)])
        keys = self.brief.keys or (() if self.brief.pointer else (*ARROWS, "Enter", "Space"))
        choices = [(actions.key(key), ("key", key)) for key in keys]
        if self.brief.pointer or not self.brief.keys:
            tool = "move" if self.brief.pointer == "drag" else self.brief.pointer or "click"
            for x, y in ((.5, .5), (.25, .5), (.75, .5), (.5, .75), (.5, .25)):
                choices.append((actions.pointer(tool, int(width * x), int(height * y)), (tool, x, y)))
        return self._choose(choices)
