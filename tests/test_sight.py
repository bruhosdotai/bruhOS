import json
import time

from bruhos.eye.publisher import motion_fraction
from bruhos.eye.reader import EyeCaption
from bruhos.eye.state import EyeState, read_state, write_state
from bruhos.eye.understand import parse_understand, understand_frame
from bruhos.listen.sight import vision_context_item
from bruhos.speech.base import SpeechConfig
from bruhos.speech.realtime import DuplexSession


def _paint(width: int, height: int, x0: int, x1: int) -> bytes:
    frame = bytearray(width * height)
    for y in range(height):
        for x in range(x0, x1):
            frame[y * width + x] = 220
    return bytes(frame)


def test_motion_flow_reports_a_rightward_shift():
    from bruhos.eye.flow import motion_flow

    width, height = 32, 24
    prev = _paint(width, height, 4, 10)
    curr = _paint(width, height, 7, 13)
    flow = motion_flow(prev, curr, width=width, height=height)
    assert flow["flow_x"] > 0
    assert "right" in flow["direction"]
    assert flow["movers"]


def test_motion_flow_is_still_when_the_frame_does_not_change():
    from bruhos.eye.flow import motion_flow

    frame = _paint(32, 24, 4, 10)
    flow = motion_flow(frame, frame, width=32, height=24)
    assert flow["direction"] == "still"
    assert flow["movers"] == []


def test_look_nudge_turns_toward_an_offcenter_mover():
    from bruhos.eye.track import look_nudge

    flow = {"movers": [{"x": 0.1, "y": 0.5, "dx": 1.0, "dy": 0.0}]}
    assert look_nudge(flow, motion=0.2) == "left"
    flow = {"movers": [{"x": 0.9, "y": 0.5, "dx": 1.0, "dy": 0.0}]}
    assert look_nudge(flow, motion=0.2) == "right"
    flow = {"movers": [{"x": 0.5, "y": 0.1, "dx": 0.0, "dy": 1.0}]}
    assert look_nudge(flow, motion=0.2) == "up"
    assert look_nudge({"movers": [{"x": 0.5, "y": 0.5, "dx": 1, "dy": 0}]}, motion=0.2) is None
    assert look_nudge(flow, motion=0.0) is None


def test_manual_head_hold_blocks_follow(tmp_path):
    from bruhos.eye.track import HeadFollow, hold_head

    sent = []
    follow = HeadFollow("http://unused/api/head", root=tmp_path, send=sent.append, quiet_s=0)
    flow = {"movers": [{"x": 0.1, "y": 0.5, "dx": 2.0, "dy": 0.0}]}
    assert follow.consider(flow, 0.2) == "left"
    hold_head(5.0, tmp_path)
    assert follow.consider(flow, 0.2) is None


def test_motion_fraction_ignores_a_still_frame():
    still = bytes([10, 20, 30, 40])
    assert motion_fraction(still, still) == 0.0
    moved = bytes([10, 200, 30, 40])
    assert motion_fraction(still, moved) == 0.25
    assert motion_fraction(b"", still) == 0.0
    assert motion_fraction(still, still + b"\x00") == 0.0


def test_state_round_trip(tmp_path):
    state = EyeState(
        frame_seq=3,
        understood_seq=2,
        caption="A red cup.",
        objects=["red cup"],
        motion=0.1,
        mean=50.0,
        updated=time.time(),
        model="ministral-3:3b",
    )
    write_state(state, tmp_path)
    got = read_state(tmp_path)
    assert got is not None
    assert got.caption == "A red cup."
    assert got.objects == ["red cup"]
    assert got.frame_seq == 3


def test_understand_frame_posts_jpeg_to_chat_completions():
    seen = {}

    def transport(url, headers, body, timeout):
        seen["url"] = url
        seen["body"] = json.loads(body)
        return json.dumps({
            "choices": [{"message": {"content": "A person holding a card."}}],
        }).encode()

    caption, objects = understand_frame(
        b"jpeg",
        base_url="http://127.0.0.1:2083/v1",
        model="ministral-3:3b",
        transport=transport,
    )
    assert caption == "A person holding a card."
    assert objects == []
    assert seen["url"] == "http://127.0.0.1:2083/v1/chat/completions"
    assert seen["body"]["model"] == "ministral-3:3b"
    assert seen["body"]["keep_alive"] == "30m"
    parts = seen["body"]["messages"][0]["content"]
    assert parts[0]["text"] == "What is in front of the camera? One sentence."
    assert parts[1]["type"] == "image_url"


def test_understand_falls_back_when_model_missing():
    import urllib.error

    seen = []

    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        seen.append(payload["model"])
        if payload["model"] == "ministral-3:3b":
            raise urllib.error.HTTPError(url, 404, "missing", hdrs={}, fp=None)
        return json.dumps({
            "choices": [{"message": {"content": "A desk."}}],
        }).encode()

    caption, _ = understand_frame(
        b"jpeg",
        base_url="http://127.0.0.1:2083/v1",
        model="ministral-3:3b",
        transport=transport,
    )
    assert caption == "A desk."
    assert seen == ["ministral-3:3b", "lfm3-omni:latest"]


def test_parse_understand_accepts_plain_caption():
    caption, objects = parse_understand("Just a desk.")
    assert caption == "Just a desk."
    assert objects == []


def test_dark_frame_skips_http(tmp_path):
    calls = {"n": 0}

    def understand(jpeg):
        calls["n"] += 1
        return "should not run", []

    from bruhos.eye.publisher import EyePublisher

    pub = EyePublisher(
        base_url="http://unused/v1",
        root=tmp_path,
        min_mean=40.0,
        cooldown_s=0.0,
        understand=understand,
    )
    pub._maybe_understand(b"jpeg", mean=1.0, motion=1.0, first_lit=True)
    assert calls["n"] == 0
    assert read_state(tmp_path).caption == "too dark to see"


def test_vision_context_item_does_not_ask_for_a_reply():
    item = vision_context_item("a red cup")
    assert item["type"] == "conversation.item.create"
    assert "response.create" not in json.dumps(item)
    assert item["item"]["role"] == "system"
    assert item["item"]["content"][0]["text"] == "In front of you: a red cup"


def test_ear_ignores_stale_caption(tmp_path):
    write_state(
        EyeState(caption="old scene", updated=time.time() - 60, frame_seq=1, understood_seq=1),
        tmp_path,
    )
    reader = EyeCaption(tmp_path, max_age_s=20)
    assert reader.refresh() == ""
    write_state(
        EyeState(caption="fresh scene", updated=time.time(), frame_seq=2, understood_seq=2),
        tmp_path,
    )
    assert reader.refresh() == "fresh scene"


def test_duplex_instructions_include_current_sight():
    class Sight:
        text = "a person holding a card"

    text = DuplexSession(SpeechConfig(chat_model="qwen2.5:14b"), sight=Sight())._instructions()
    assert "The camera shows: a person holding a card." in text
    assert "one or two short sentences of plain English" in text
    assert "Mention the camera only" in text
    bare = DuplexSession(SpeechConfig(chat_model="qwen2.5:14b"))._instructions()
    assert "The camera shows:" not in bare
    assert "Mention the camera only" in bare


def test_plain_caption_drops_chinese_json_and_ocr_junk():
    from bruhos.speech.realtime import build_instructions

    chinese = build_instructions("A bill on the table. 股东法治")
    assert "The camera shows: A bill on the table." in chinese
    assert "股东" not in chinese
    junk = build_instructions('{"caption": "openh股东"}')
    assert "The camera shows:" not in junk
    ocr = build_instructions("openh42XX##股东")
    assert "The camera shows:" not in ocr
    assert "No Chinese" in junk


def test_build_instructions_keeps_caption_short():
    from bruhos.speech.realtime import build_instructions

    long = "First sentence here. Second sentence here. Third should be dropped."
    text = build_instructions(long, when=0)
    assert "The camera shows: First sentence here." in text
    assert "Second sentence here." not in text


def test_duplex_pushes_caption_via_session_update(monkeypatch):
    """Pixels stay on vision. Only the caption crosses, as instructions."""
    import threading

    sent: list[dict] = []

    class FakeWS:
        def __init__(self, url, timeout=20.0):
            self._q = [
                json.dumps({
                    "type": "session.created",
                    "session": {
                        "audio": {"input": {"format": {"rate": 16000}}},
                        "asr": "qwen3-asr:latest",
                    },
                }),
                None,
            ]

        def settimeout(self, _t):
            return None

        def recv_text(self):
            if not self._q:
                raise RuntimeError("closed")
            item = self._q.pop(0)
            if item is None:
                raise RuntimeError("closed")
            return item

        def send_text(self, raw):
            sent.append(json.loads(raw))

        def close(self):
            return None

    class Sight:
        text = ""

        def __init__(self):
            self._fn = None

        def subscribe(self, fn):
            self._fn = fn
            return lambda: None

        def push(self, caption: str):
            self.text = caption
            assert self._fn is not None
            self._fn(caption)

    class Mic:
        def __init__(self):
            self.q = __import__("queue").Queue()
            self.q.put(b"\x00\x00" * 480)

        def close(self):
            self.q.put(None)

    class Player:
        def write(self, data, rate):
            return None

        def flush(self):
            return None

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    sight = Sight()
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        chat_model="qwen2.5:14b",
        realtime="duplex",
        realtime_turn="server_vad",
    )
    stop = threading.Event()

    def _push_then_stop():
        time.sleep(0.05)
        sight.push("A man covers his face. He looks slightly down. Extra clutter.")
        stop.set()

    threading.Thread(target=_push_then_stop, daemon=True).start()
    DuplexSession(cfg, sight=sight).run(stop, device="plughw:Device,0", player=Player(), mic=Mic(), once=True)

    kinds = [m["type"] for m in sent]
    assert "conversation.item.create" not in kinds
    updates = [m for m in sent if m["type"] == "session.update"]
    assert len(updates) >= 2
    last = updates[-1]["session"]["instructions"]
    assert "The camera shows: A man covers his face." in last
    assert "A man covers his face." in last
    assert "He looks slightly down." not in last
    assert "Extra clutter" not in last
    assert "audio" not in updates[-1]["session"]
