import base64
import json
import os
from queue import Empty, SimpleQueue
from threading import Thread
from urllib.request import ProxyHandler, Request, build_opener


class LocalModel:
    def __init__(self, port, model, timeout=2.0):
        self.endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self.results = SimpleQueue()
        self.running = False

    @classmethod
    def from_env(cls):
        model = os.environ.get("GGA_MODEL_NAME")
        if not model:
            return None
        try:
            port = int(os.environ.get("GGA_MODEL_PORT", "8000"))
        except ValueError:
            return None
        return cls(port, model) if 1 <= port <= 65535 else None

    def _request(self, game, turn, brief, jpeg, schemas):
        proposal = None
        try:
            data = base64.b64encode(jpeg).decode("ascii")
            prompt = (f"Play this browser game. {brief[:3000]}\n"
                      f"Available input tools: {json.dumps(schemas)[:10000]}\n"
                      'Return only a JSON object {"tool": "name", "params": {}}. '
                      "Use coordinates in screenshot pixels. Use skip when unsure.")
            payload = {"model": self.model, "temperature": 0, "max_tokens": 128, "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}, {
                    "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                }]},
            ]}
            opener = build_opener(ProxyHandler({}))
            request = Request(self.endpoint, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with opener.open(request, timeout=self.timeout) as response:
                content = json.loads(response.read(65536))["choices"][0]["message"]["content"]
                proposal = json.loads(content)
        except (OSError, ValueError, LookupError, TypeError):
            pass
        finally:
            self.results.put((game, turn, proposal))

    def poll(self, game, turn, brief, jpeg, schemas):
        proposal = None
        try:
            previous_game, previous_turn, result = self.results.get_nowait()
        except Empty:
            pass
        else:
            self.running = False
            if previous_game == game and turn - previous_turn <= 3:
                proposal = result
        if not self.running and turn % 8 == 1 and len(jpeg) <= 1_500_000:
            self.running = True
            Thread(target=self._request, args=(game, turn, brief, jpeg, schemas), daemon=True).start()
        return proposal
