import pytest

from bruhos.brain.base import Planner, parse_plan
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall, parse_call
from bruhos.sim.kinematic import SimBody, World, WorldObject
from bruhos.trace import TraceWriter, load_episode, verify_episode


@pytest.mark.parametrize('text, target', [
    ('detect(target="crate,large")', 'crate,large'),
    ("detect('crate,large')", 'crate,large'),
    ('detect("crate=large")', 'crate=large'),
])
def test_quoted_target_is_one_literal_argument(text, target):
    assert parse_call(text, {'detect': 'target'}) == SkillCall('detect', {'target': target})


@pytest.mark.parametrize('text, expected', [
    ('walk_toward(target=red_block, stop_distance_m=0.2)',
     SkillCall('walk_toward', {'target': 'red_block', 'stop_distance_m': 0.2})),
    ("detect(worker's_block)", SkillCall('detect', {'target': "worker's_block"})),
    ('detect("crate\'s,large")', SkillCall('detect', {'target': "crate's,large"})),
])
def test_unquoted_separators_and_other_quote_characters_keep_their_meaning(text, expected):
    assert parse_call(text, {'detect': 'target', 'walk_toward': 'target'}) == expected


def test_literal_comma_target_reaches_the_simulator_and_saved_trace(tmp_path):
    class SuppliedPlan(Planner):
        def plan(self, instruction, obs, catalog):
            return parse_plan(['detect(target="crate,large")'])

    body = SimBody(World(objects=[WorldObject('crate,large', None, 0.1, 0.0)]))
    episode = Runtime(body, SuppliedPlan()).run('Detect the fictional labelled crate.')
    assert episode.success, episode.end_reason
    assert episode.steps[0].call == {'skill': 'detect', 'args': {'target': 'crate,large'}}
    assert episode.steps[0].outcome['data']['found'] == 'crate,large'
    saved = load_episode(TraceWriter(tmp_path).write(episode))
    assert verify_episode(saved)
    assert saved.steps[0].call == episode.steps[0].call
