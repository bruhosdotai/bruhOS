"""WonderEcho Pro CI1302 wake/sleep over USB-serial (CH340)."""

from __future__ import annotations

WAKE = b"\xaa\x55\x03\x00\xfb"
SLEEP = b"\xaa\x55\x02\x00\xfb"

# Hiwonder stock firmware also emits motion phrases; we only need wake/sleep for ear loop.
KNOWN = {
    WAKE: "wakeup",
    SLEEP: "sleep",
    b"\xaa\x55\x00\x01\xfb": "forward",
    b"\xaa\x55\x00\x02\xfb": "back",
    b"\xaa\x55\x00\x03\xfb": "turn_left",
    b"\xaa\x55\x00\x04\xfb": "turn_right",
}


class WonderEcho:
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        *,
        wait_s: float = 60.0,
        debug: bool = False,
    ):
        import serial  # pyserial — pip install pyserial / bruhos[ear]
        import time
        from pathlib import Path

        self.port = port
        self.debug = debug
        deadline = time.monotonic() + max(0.0, wait_s)
        while not Path(port).exists():
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"WonderEcho serial not found: {port}")
            time.sleep(0.25)

        self._ser = serial.Serial(
            None, baud, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE, timeout=0.05
        )
        self._ser.rts = False
        self._ser.dtr = False
        self._ser.setPort(port)
        self._ser.open()
        self._ser.reset_input_buffer()
        if self.debug:
            print(f"[ear:serial] open {port} baud={baud}", flush=True)

    def read_event(self) -> str | None:
        raw = self._ser.read(5)
        if not raw:
            return None
        # resync if we got a partial / noise frame
        if len(raw) < 5 or raw[0] != 0xAA:
            if self.debug and raw:
                print(f"[ear:serial] resync trash={raw.hex()}", flush=True)
            self._ser.reset_input_buffer()
            return None
        name = KNOWN.get(raw)
        if self.debug:
            print(
                f"[ear:serial] frame={raw.hex()} -> {name or 'unknown'}",
                flush=True,
            )
        return name

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass
