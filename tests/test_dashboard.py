from functools import partial
from http.server import ThreadingHTTPServer
import json
import threading
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from bruhos.dashboard import Handler, status_data
from bruhos.runtime import Runtime
from bruhos.brain import make_planner
from bruhos.sim.kinematic import SimBody
from bruhos.trace import TraceWriter


def test_saved_runtime_trace_and_corrupt_neighbor(tmp_path):
    ep = Runtime(SimBody(), make_planner('rules')).run('Wave hello.')
    TraceWriter(tmp_path).write(ep)
    broken = tmp_path/'broken'
    broken.mkdir()
    (broken/'episode.json').write_text('{}')
    data = status_data(tmp_path)
    assert len(data['episodes']) == 1
    assert data['episodes'][0]['episode_id'] == ep.episode_id
    assert data['episodes'][0]['verified']
    assert data['skipped'] == 1
    steps = tmp_path/ep.episode_id/'steps.jsonl'
    records = steps.read_text().splitlines()
    record = json.loads(records[0]);record['outcome']['detail'] = 'changed'
    records[0] = json.dumps(record)
    steps.write_text('\n'.join(records)+'\n')
    assert not status_data(tmp_path)['episodes'][0]['verified']


def test_empty_root_and_http_asset_boundary(tmp_path):
    assert status_data(tmp_path)['episodes'] == []
    (tmp_path/'secret.txt').write_text('private')
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, root=tmp_path))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        for path in ('/', '/app.js', '/styles.css', '/assets/favicon.png'):
            with urlopen(url+path) as response:
                assert response.status == 200
                assert response.headers['X-Content-Type-Options'] == 'nosniff'
        with urlopen(url+'/status') as response:
            assert json.load(response)['episodes'] == []
        for path in ('/secret.txt', '/../secret.txt', '/camera.jpg'):
            with pytest.raises(HTTPError) as err:
                urlopen(url+path)
            assert err.value.code == 404
    finally:
        server.shutdown();server.server_close();worker.join(2)


def test_latest_twelve_are_sorted_and_external_links_skipped(tmp_path):
    ep = Runtime(SimBody(), make_planner('rules')).run('Wave hello.')
    for index in range(14):
        ep.episode_id = f'episode-{index}'
        ep.started_at = index
        TraceWriter(tmp_path).write(ep)
    (tmp_path/'linked').symlink_to(tmp_path/'episode-0', target_is_directory=True)
    from bruhos.dashboard import load_episode
    with patch('bruhos.dashboard.load_episode', wraps=load_episode) as loader:
        data = status_data(tmp_path)
    assert loader.call_count == 12
    assert all('plan' not in e for e in data['episodes'])
    assert [e['started_at'] for e in data['episodes']] == list(range(13, 1, -1))
    assert data['skipped'] == 1


@pytest.mark.parametrize('timestamp', [True, float('nan'), float('inf'), 10**400])
def test_invalid_timestamp_does_not_break_status_json(tmp_path, timestamp):
    ep = Runtime(SimBody(), make_planner('rules')).run('Wave hello.')
    ep.started_at = timestamp
    TraceWriter(tmp_path).write(ep)
    data = status_data(tmp_path)
    assert data['episodes'] == []
    assert data['skipped'] == 1
    json.dumps(data, allow_nan=False)
