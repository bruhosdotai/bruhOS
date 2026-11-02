from bruhos.brain.adapters.groot_s2 import GR00TS2Planner
from bruhos.brain.adapters.qwen_vl import QwenVLPlanner
from bruhos.cli import _brain

QWEN_CFG = {"brain": {"name": "qwen_vl", "base_url": "https://gpu.example:8000/v1", "model": "m"}}


def test_brain_flag_for_another_brain_ignores_config_settings():
    # [brain] settings belong to qwen_vl; groot_s2 takes none of them.
    assert isinstance(_brain("groot_s2", QWEN_CFG), GR00TS2Planner)


def test_config_settings_apply_to_the_configured_brain():
    for name in (None, "qwen_vl"):
        p = _brain(name, QWEN_CFG)
        assert isinstance(p, QwenVLPlanner)
        assert p.base_url == "https://gpu.example:8000/v1"
