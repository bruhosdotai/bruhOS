"""Qwen-VL planner served locally, e.g. ``vllm serve Qwen/Qwen2.5-VL-7B-Instruct``.

Same schema as the OpenAI adapter; optionally attaches one camera frame as an image part.
The observation JSON stays the primary input — the frame is context, not the control signal.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable

from bruhos.schemas import Observation

from .openai import OpenAICompatPlanner


class QwenVLPlanner(OpenAICompatPlanner):
    name = "qwen_vl"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 api_key: str = "EMPTY", frame_provider: Callable[[], bytes | None] | None = None, **kw: Any):
        super().__init__(base_url=base_url, model=model, api_key=api_key, **kw)
        self.frame_provider = frame_provider

    def user_content(self, instruction: str, obs: Observation) -> Any:
        text = json.dumps({"instruction": instruction, "observation": obs.to_dict()})
        jpeg = self.frame_provider() if self.frame_provider else None
        if not jpeg:
            return text
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]
