"""Pi detect package: OpenAPI /v1/detect client, factory, ground_image."""

from __future__ import annotations

import json

import pytest

from bruhos.body.perception import CameraModel, ColorBlobDetector
from bruhos.detect import (
    BBox,
    DetectRequest,
    DetectResponse,
    RemoteDetectClient,
    bboxes_to_detections,
    make_hot_loop_detector,
    make_local_helper,
    make_remote_client,
)
from bruhos.scan.ground import FrameResult, GroundedDet, ground_image, run_gpu_detect


def test_bbox_from_dict_score_alias():
    b = BBox.from_dict({
        "label": "cup",
        "score": 0.9,
        "bbox": [0.1, 0.2, 0.3, 0.4],
    }, default_normalized=True)
    assert b.label == "cup"
    assert b.confidence == 0.9
    assert b.normalized is True


def test_detect_request_openapi_shape():
    req = DetectRequest(
        image_jpeg=b"\xff\xd8",
        queries=["cup", "bottle"],
        model="detect",
        threshold=0.2,
        coordinate_space="normalized",
        frame_id="cam0",
    )
    d = req.to_json_dict()
    assert d["model"] == "detect"
    assert d["queries"] == ["cup", "bottle"]
    assert d["threshold"] == 0.2
    assert d["frame_id"] == "cam0"
    assert "image" in d
    assert "classes" not in d


def test_detect_response_wire_only():
    resp = DetectResponse.from_dict({
        "model": "detect",
        "status": "wire_only",
        "coordinate_space": "normalized",
        "image_size": {"width": 640, "height": 480},
        "detections": [],
        "backend": "stub",
    })
    assert resp.wire_only
    assert not resp.ok_real
    assert resp.width == 640


def test_detect_response_owlv2_pixel_box_without_coordinate_space():
    resp = DetectResponse.from_dict({
        "status": "ok",
        "model": "owlv2",
        "image": {"width": 320, "height": 240},
        "detections": [
            {"label": "orange block", "score": 0.47, "box": [80.0, 49.75, 204.0, 181.0]},
        ],
    })
    assert resp.width == 320 and resp.height == 240
    assert resp.coordinate_space == "pixel"
    assert resp.detections[0].normalized is False
    assert resp.detections[0].confidence == 0.47


def test_detect_response_ok_with_boxes():
    resp = DetectResponse.from_dict({
        "model": "detect",
        "status": "ok",
        "coordinate_space": "normalized",
        "image_size": {"width": 64, "height": 48},
        "backend": "owlv2",
        "detections": [
            {"label": "mug", "score": 0.8, "bbox": [0.1, 0.2, 0.3, 0.4]},
        ],
    })
    assert resp.ok_real
    assert resp.detections[0].confidence == 0.8
    assert resp.detections[0].normalized is True


def test_bboxes_to_detections_normalized():
    cam = CameraModel(640, 480, 62.0)
    boxes = [BBox("block", 0.4, 0.4, 0.6, 0.6, 0.95, normalized=True)]
    dets = bboxes_to_detections(boxes, cam)
    assert len(dets) == 1
    assert dets[0].label == "block"
    assert abs(dets[0].bearing_deg) < 5.0


def test_remote_client_posts_v1_detect():
    seen: dict = {}

    def transport(url, headers, body, timeout):
        seen["url"] = url
        seen["headers"] = headers
        seen["body"] = json.loads(body)
        return json.dumps({
            "model": "detect",
            "status": "ok",
            "coordinate_space": "normalized",
            "image_size": {"width": 64, "height": 48},
            "backend": "owlv2",
            "detections": [{"label": "cup", "score": 0.7, "bbox": [0.1, 0.1, 0.5, 0.5]}],
        }).encode()

    client = RemoteDetectClient(
        detect_url="http://gpu.example:2083",
        model="detect",
        timeout=12.0,
        api_key="sekrit",
        transport=transport,
    )
    assert client.endpoint == "http://gpu.example:2083/v1/detect"
    resp = client.detect_jpeg(b"\xff\xd8fake", queries=["cup"], threshold=0.3)
    assert seen["url"] == client.endpoint
    assert seen["headers"]["Authorization"] == "Bearer sekrit"
    assert seen["body"]["queries"] == ["cup"]
    assert seen["body"]["model"] == "detect"
    assert seen["body"]["threshold"] == 0.3
    assert resp.detections[0].label == "cup"
    assert resp.status == "ok"


def test_remote_strips_trailing_v1():
    c = RemoteDetectClient(detect_url="http://host/v1/", transport=lambda *a: b'{"detections":[],"coordinate_space":"normalized","image_size":{"width":1,"height":1},"model":"detect"}')
    assert c.endpoint == "http://host/v1/detect"


def test_make_remote_client_none_without_url(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_DETECT_URL", raising=False)
    monkeypatch.delenv("BRUHOS_DETECT_URL", raising=False)
    assert make_remote_client({}) is None


def test_make_remote_client_defaults_model(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_DETECT_URL", raising=False)
    c = make_remote_client({"detect": {"detect_url": "http://127.0.0.1:2083"}})
    assert c is not None
    assert c.model == "detect"


def test_hot_loop_factory_hsv():
    det = make_hot_loop_detector({"detect": {"backend": "hsv"}, "camera": {"width": 320, "height": 240}})
    assert isinstance(det, ColorBlobDetector)


def test_local_helper_hsv_jpeg():
    import cv2
    import numpy as np

    helper = make_local_helper({"detect": {"backend": "hsv"}})
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[:] = (200, 80, 40)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    resp = helper.detect(DetectRequest(image_jpeg=bytes(buf), queries=["block"]))
    assert resp.backend == "local:hsv"
    assert resp.status == "ok"


def test_run_gpu_detect_enrich():
    class Fake:
        def detect_jpeg(self, jpeg, **kw):
            return DetectResponse.from_dict({
                "model": "detect", "status": "ok",
                "coordinate_space": "normalized",
                "image_size": {"width": 64, "height": 48},
                "backend": "owlv2",
                "detections": [{"label": "shelf", "score": 0.9, "bbox": [0.2, 0.2, 0.8, 0.8]}],
            })

    fr = run_gpu_detect(
        label="see", jpeg=_tiny(), client=Fake(), queries=["shelf"],
        cam=CameraModel(64, 48, 62.0),
    )
    assert fr.ok and fr.detections
    assert fr.detections[0].label == "shelf"
    assert fr.detections[0].normalized is False  # enrich → pixels
    assert "bearing_deg" in fr.detections[0].to_dict()


def test_run_gpu_detect_wire_only_is_error():
    class Fake:
        def detect_jpeg(self, jpeg, **kw):
            return DetectResponse.from_dict({
                "model": "detect", "status": "wire_only",
                "coordinate_space": "normalized",
                "image_size": {"width": 64, "height": 48},
                "detections": [], "backend": "stub",
            })

    fr = run_gpu_detect(label="see", jpeg=_tiny(), client=Fake(), queries=["cup"])
    assert not fr.ok
    assert "wire_only" in (fr.error or "")


def test_ground_image_falls_back_to_vl_on_wire_only():
    class FakeDetect:
        def detect_jpeg(self, jpeg, **kw):
            return DetectResponse.from_dict({
                "model": "detect", "status": "wire_only",
                "coordinate_space": "normalized",
                "image_size": {"width": 64, "height": 48},
                "detections": [],
            })

    def fake_vl(**kwargs):
        return FrameResult(
            image="see", backend="vl-boxes:test", seconds=0.01,
            width=64, height=48,
            detections=[GroundedDet("tv", 1, 1, 20, 20, conf=0.5)],
        )

    fr = ground_image(
        label="see", jpeg=_tiny(), cam=CameraModel(64, 48, 62.0),
        cfg={
            "detect": {"detect_url": "http://x", "ground": "auto"},
            "scan": {"base_url": "http://x/v1", "ground_model": "qwen"},
        },
        hint_objects=["tv"],
        detect_client=FakeDetect(),
        vl_fn=fake_vl,
    )
    assert fr.ok
    assert fr.backend.startswith("vl-boxes")
    assert fr.detections[0].label == "tv"


def test_factory_onnx_requires_path():
    with pytest.raises(ValueError, match="onnx_path"):
        make_hot_loop_detector({"detect": {"backend": "onnx"}})


def _tiny() -> bytes:
    import cv2
    import numpy as np

    img = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)
