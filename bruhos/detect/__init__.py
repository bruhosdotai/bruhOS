"""Pi vision detect: hot-loop local + sparse GPU ``/v1/detect`` client.

Architecture (Pi vs GPU box):

- **On Pi:** camera, local detector (HSV / nano ONNX), control, thin HTTP client
- **Not on Pi:** ``zerollama serve``, OWLv2, Grounding DINO, detect sibling product
- **GPU:** ``POST {ZEROLLAMA_DETECT_URL}/v1/detect`` when policy needs open-vocab
"""

from bruhos.detect.factory import (
    make_hot_loop_detector,
    make_local_backend,
    make_local_helper,
    make_remote_client,
)
from bruhos.detect.remote import RemoteDetectClient
from bruhos.detect.types import (
    BBox,
    DetectRequest,
    DetectResponse,
    bboxes_to_detections,
    frame_to_jpeg,
)

__all__ = [
    "BBox",
    "DetectRequest",
    "DetectResponse",
    "RemoteDetectClient",
    "bboxes_to_detections",
    "frame_to_jpeg",
    "make_hot_loop_detector",
    "make_local_backend",
    "make_local_helper",
    "make_remote_client",
]
