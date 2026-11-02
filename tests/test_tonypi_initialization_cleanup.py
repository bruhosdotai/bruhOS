from types import SimpleNamespace

import pytest

from bruhos.body import tonypi


def test_partial_sensor_setup_closes_an_acquired_resource(tmp_path, monkeypatch):
    resource = (tmp_path / 'fictional-camera').open('wb')
    monkeypatch.setattr(tonypi, '_load_agc', lambda _: SimpleNamespace())

    def failing_setup(body, cfg):
        body.cap = SimpleNamespace(release=resource.close)
        body.imu = None
        raise ValueError('fictional later sensor setup failure')

    monkeypatch.setattr(tonypi.TonyPiBody, '_setup_sensors', failing_setup)
    try:
        with pytest.raises(ValueError, match='later sensor setup failure'):
            tonypi.TonyPiBody({})
        assert resource.closed
    finally:
        resource.close()


@pytest.fixture
def video_camera(tmp_path, monkeypatch):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    path = tmp_path / 'fictional-camera.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (16, 16))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.release()
    captures = []
    original = cv2.VideoCapture

    def capture(source):
        assert source == str(path)  # Never open a hardware camera.
        camera = original(source)
        assert camera.isOpened()
        captures.append(camera)
        return camera

    monkeypatch.setattr(cv2, 'VideoCapture', capture)
    monkeypatch.setattr(tonypi, '_load_agc', lambda _: SimpleNamespace())
    yield {'camera': {'device': str(path), 'width': 16, 'height': 16}}, captures
    for camera in captures:
        camera.release()


def test_real_video_capture_is_released_when_later_config_is_invalid(video_camera):
    cfg, captures = video_camera
    cfg['imu'] = {'address': 'not-an-address'}
    with pytest.raises(ValueError, match='invalid literal'):
        tonypi.TonyPiBody(cfg)
    assert len(captures) == 1
    assert not captures[0].isOpened()


def test_successful_video_setup_stays_open_until_body_close(video_camera):
    cfg, captures = video_camera
    cfg['imu'] = {'enabled': False}
    body = tonypi.TonyPiBody(cfg)
    try:
        assert captures[0].isOpened()
        assert body.observe().frame == 1
    finally:
        body.close()
    assert not captures[0].isOpened()
