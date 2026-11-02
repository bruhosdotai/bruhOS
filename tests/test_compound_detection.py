from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Detection, SkillCall
from bruhos.sim.kinematic import SimBody, World, WorldObject


class DetectPlanner(Planner):
    def plan(self, instruction, obs, catalog):
        return [SkillCall("detect", {"target": instruction})]


def test_compound_detection_can_resolve_its_own_name_in_sim():
    body = SimBody(World(objects=[WorldObject("traffic_light", "red", 0.1, 0)]))
    episode = Runtime(body, DetectPlanner()).run("red_traffic_light")
    assert episode.success, episode.end_reason
    assert episode.steps[0].outcome["data"]["found"] == "red_traffic_light"


def test_compound_labels_still_require_all_target_tokens():
    detection = Detection("traffic_light", "red")
    assert detection.matches("traffic-light")
    assert detection.matches("red")
    assert not detection.matches("green_traffic_light")
