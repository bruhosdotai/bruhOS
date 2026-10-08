"""Simulation.

``kinematic`` is a small, deterministic 2D tabletop world that runs anywhere (CI, laptop, Pi)
and implements the same Body primitives as the TonyPi. ``tabletop_task.toml`` is the scene
spec for the Isaac Lab digital twin planned for v0.2; both share the same instruction /
skill / outcome schema so sim and real failures land in the same eval.
"""

from .kinematic import SimBody, World, WorldObject

__all__ = ["SimBody", "World", "WorldObject"]
