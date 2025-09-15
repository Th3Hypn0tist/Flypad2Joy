#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flypad2joy_notify_neutral_rt_v2.py
- BLE -> ViGEm (Xbox 360) bridge (Windows).
- CPU-jitter hardened (producer/consumer), *with correct buttons & triggers*.
- Producer (notify) stores the full latest packet safely.
- Consumer (200 Hz thread) maps axes linearly + neutral clamp + jitter guard.
- No EMA/anti-spike. Optional keepalive write.

Config: flypad.conf ([axes], [axis_centers], [general], [buttons], [triggers])
Filters used here: deadzone, neutral_eps, drop_all_zero_frame, jitter_guard_ms.
"""

import asyncio, os, sys, time, contextlib, configparser, threading
from typing import Dict, Any, Optional, Tuple
from bleak import BleakScanner, BleakClient, BLEDevice, AdvertisementData
from vgamepad import VX360Gamepad, XUSB_BUTTON

# ---------- Paths & constants ----------
CONF_PATH = os.path.join(os.path.dirname(__file__), "flypad.conf")
EXPECTED_MIN_LEN = 6

# Keepalive (set to False if device does not need it)
HB_ENABLE, HB_INTERVAL, HB_PAYLOAD = True, 0.5, b"\x01"

# Consumer update rate
CONSUMER_HZ = 200
CONSUMER_DT = 1.0 / CONSUMER_HZ

# ---------- Windows priority & timer ----------
IS_WINDOWS = os.name == "nt"
def _bump_priority_and_timer():
    if not IS_WINDOWS:
        return
    try:
        # HIGH_PRIORITY_CLASS = 0x00000080
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        proc = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(proc, 0x00000080)
    except Exception:
        pass
    try:
        # Request 1ms timer resolution
        import ctypes
        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod(1)
    except Exception:
        pass

def _restore_timer():
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        winmm = ctypes.WinDLL("winmm")
        winmm.timeEndPeriod(1)
    except Exception:
        pass

# ---------- Minimal config ----------
TEMPLATE = """[general]
target_name_prefix = FLYPAD_
ctrl_char_uuid = 9e35fa01-4344-44d4-a2e2-0c7f6046878b
invert_y = true
button_lsb_index = 1
button_msb_index = 2

[axes]
lx = 5
ly = 6
rx = 3
ry = 4

[axis_centers]
lx = 128
ly = 128
rx = 128
ry = 128

[buttons]
A = 4,0
B = 3,0
1 = 1,0
2 = 2,0
LB = 7,0
LT = 0,1
RB = 5,0
RT = 6,0
TAKEOFF = 0,0

[triggers]
lt_mode = bit
lt_bit = 0
lt_msb = 1
lt_byte = -1
lt_invert = 0

rt_mode = bit
rt_bit = 6
rt_msb = 0
rt_byte = -1
rt_invert = 0

[filters]
deadzone = 3
neutral_eps = 1
drop_all_zero_frame = true
jitter_guard_ms = 20   ; if consumer detects >20ms since last raw AND near neutral -> force 0
"""

def ensure_conf():
    if not os.path.exists(CONF_PATH):
        with open(CONF_PATH, "w", encoding="utf-8") as f:
            f.write(TEMPLATE)
        print(f"[i] Template written: {CONF_PATH}")
        print("    -> Edit [axes] indices if needed.")
        sys.exit(0)

def read_conf() -> Dict[str, Any]:
    ensure_conf()
    cp = configparser.ConfigParser()
    cp.read(CONF_PATH, encoding="utf-8")
    if not cp.has_section("general") or not cp.has_section("axes"):
        print(f"[!] Invalid config. Need [general] and [axes] in {CONF_PATH}."); sys.exit(1)

    g   = dict(cp.items("general"))
    ax  = dict(cp.items("axes"))
    cen = dict(cp.items("axis_centers")) if cp.has_section("axis_centers") else {}
    flt = dict(cp.items("filters")) if cp.has_section("filters") else {}
    btn = dict(cp.items("buttons")) if cp.has_section("buttons") else {}
    trg = dict(cp.items("triggers")) if cp.has_section("triggers") else {}

    def b2(x, dflt=False):
        s = str(x).strip().lower()
        return dflt if s=="" else (s in ("1","true","yes","on"))
    def geti(d, k, dflt):
        try: return int(d.get(k, dflt))
        except: return dflt

    conf = {
        "prefix": g.get("target_name_prefix", "FLYPAD_"),
        "uuid": g.get("ctrl_char_uuid", "9e35fa01-4344-44d4-a2e2-0c7f6046878b"),
        "invert_y": b2(g.get("invert_y", True), True),
        "lsb_idx": geti(g, "button_lsb_index", -1),
        "msb_idx": geti(g, "button_msb_index", -1),
        "axes": {k: geti(ax, k, -1) for k in ("lx","ly","rx","ry")},
        "centers": {k: geti(cen, k, 128) for k in ("lx","ly","rx","ry")},
        "filters": {
            "dz": geti(flt, "deadzone", 3),
            "neutral_eps": geti(flt, "neutral_eps", 1),
            "drop_all_zero": b2(flt.get("drop_all_zero_frame", True), True),
            "jitter_guard_ms": geti(flt, "jitter_guard_ms", 20),
        },
        "buttons": {},
        "triggers": {
            "lt_mode": trg.get("lt_mode", "bit"),
            "lt_bit":  trg.get("lt_bit",  "-1"),
            "lt_msb":  trg.get("lt_msb",  "0"),
            "lt_byte": trg.get("lt_byte", "-1"),
            "lt_invert": trg.get("lt_invert", "0"),
            "rt_mode": trg.get("rt_mode", "bit"),
            "rt_bit":  trg.get("rt_bit",  "-1"),
            "rt_msb":  trg.get("rt_msb",  "0"),
            "rt_byte": trg.get("rt_byte", "-1"),
            "rt_invert": trg.get("rt_invert", "0"),
        },
    }
    # buttons
    for k, v in btn.items():
        try:
            bit_str, msb_str = [x.strip() for x in v.split(",", 1)]
            conf["buttons"][k.upper()] = (int(bit_str), int(msb_str))
        except:
            conf["buttons"][k.upper()] = (-1, 0)

    for k in ("lx","ly","rx","ry"):
        if conf["axes"][k] < 0:
            print(f"[!] Missing axis index: {k} in [axes]. Edit {CONF_PATH}."); sys.exit(1)
    if conf["lsb_idx"] < 0:
        print(f"[!] Missing button_lsb_index in [general]. Edit {CONF_PATH}."); sys.exit(1)
    return conf

# ---------- Mapping ----------
def map_linear_u8_to_xinput(raw_u8: int, center: int, deadzone: int, invert: bool) -> int:
    v = 0 if raw_u8 < 0 else (255 if raw_u8 > 255 else raw_u8)
    if invert:
        v = 255 - v
    d = float(v - center)
    if abs(d) <= deadzone:
        return 0
    mag = abs(d) - deadzone
    pos = d >= 0
    span_pos = max(1.0, 255.0 - center - deadzone)
    span_neg = max(1.0, center - deadzone)
    full = 32767 if pos else 32768
    out = int(round(mag * full / (span_pos if pos else span_neg)))
    return min(32767, out) if pos else -min(32768, out)

# ---------- BLE helpers ----------
def is_target(d: BLEDevice, a: AdvertisementData, prefix: str) -> bool:
    return (d.name or "").startswith(prefix)

# ---------- Bridge (producer/consumer) ----------
class Bridge:
    """
    Producer: BLE notify sets self._pkt (full bytes) and self._pkt_ts.
    Consumer thread: ticks at fixed rate, maps raw->XInput, updates pad, handles buttons & triggers.
    """
    __slots__ = ("pad","idx","centers","invy","dz","lsb","msb","buttons","tr",
                 "drop_all_zero","neutral_eps","jitter_guard",
                 "_pkt","_pkt_ts","_lock","_run","_thr")

    def __init__(self, conf: Dict[str, Any]):
        self.pad = VX360Gamepad()
        self.idx = (conf["axes"]["lx"], conf["axes"]["ly"], conf["axes"]["rx"], conf["axes"]["ry"])
        self.centers = (conf["centers"]["lx"], conf["centers"]["ly"], conf["centers"]["rx"], conf["centers"]["ry"])
        self.invy = conf["invert_y"]
        self.dz = conf["filters"]["dz"]
        self.neutral_eps = conf["filters"]["neutral_eps"]
        self.jitter_guard = max(0.0, conf["filters"]["jitter_guard_ms"] / 1000.0)
        self.drop_all_zero = conf["filters"]["drop_all_zero"]

        self.lsb = conf["lsb_idx"]; self.msb = conf.get("msb_idx", -1)
        self.buttons = conf["buttons"]; self.tr = conf["triggers"]

        self._pkt: bytes = bytes([128] * ((max(self.idx) + 1) if self.idx else 4))
        self._pkt_ts: float = time.perf_counter()
        self._lock = threading.Lock()
        self._run = threading.Event(); self._run.clear()
        self._thr: Optional[threading.Thread] = None

    # ---- producer path ----
    def on_notify(self, data: bytes):
        # store full packet & timestamp
        with self._lock:
            self._pkt = bytes(data)
            self._pkt_ts = time.perf_counter()

    # ---- helpers to read from a packet safely ----
    def _getb(self, pkt: bytes, i: int) -> int:
        return pkt[i] if 0 <= i < len(pkt) else 0

    def _axes_from_pkt(self, pkt: bytes) -> Tuple[int,int,int,int]:
        i0,i1,i2,i3 = self.idx
        return (self._getb(pkt,i0), self._getb(pkt,i1), self._getb(pkt,i2), self._getb(pkt,i3))

    def _btnbit_from_pkt(self, pkt: bytes, name: str) -> int:
        tup = self.buttons.get(name)
        if not tup: return 0
        bit, msb = tup
        if bit < 0: return 0
        idx = self.msb if msb else self.lsb
        return (self._getb(pkt, idx) >> bit) & 1

    def _trig_from_pkt(self, pkt: bytes, which: str) -> int:
        mode = self.tr.get(f"{which}_mode", "bit")
        if mode == "byte":
            idx = int(self.tr.get(f"{which}_byte", -1))
            inv = int(self.tr.get(f"{which}_invert", 0))
            v = self._getb(pkt, idx)
            return (255 - v) if inv else v
        bit = int(self.tr.get(f"{which}_bit", -1))
        msb = int(self.tr.get(f"{which}_msb", 0))
        idx = self.msb if msb else self.lsb
        if bit < 0: return 0
        return 255 if ((self._getb(pkt, idx) >> bit) & 1) else 0

    # ---- consumer path ----
    def _tick(self):
        # snapshot packet & ts
        with self._lock:
            pkt = self._pkt
            ts  = self._pkt_ts

        now = time.perf_counter()
        starved = (now - ts) > self.jitter_guard if self.jitter_guard > 0 else False

        # axes
        r0,r1,r2,r3 = self._axes_from_pkt(pkt)

        # Drop obvious idle/glitch frames (all axes zero) → treat as neutral output,
        # but still allow buttons/triggers to update from the packet
        all_zero_axes = (r0|r1|r2|r3) == 0

        c0,c1,c2,c3 = self.centers
        eps = self.neutral_eps
        dz  = self.dz
        invy = self.invy

        if all_zero_axes and self.drop_all_zero:
            lx = ly = rx = ry = 0
        elif (abs(r0-c0) <= eps and abs(r1-c1) <= eps and
              abs(r2-c2) <= eps and abs(r3-c3) <= eps):
            lx = ly = rx = ry = 0
        elif starved and (abs(r0-c0) <= eps*2 and abs(r1-c1) <= eps*2 and
                          abs(r2-c2) <= eps*2 and abs(r3-c3) <= eps*2):
            lx = ly = rx = ry = 0
        else:
            lx = map_linear_u8_to_xinput(r0, c0, dz, False)
            ly = map_linear_u8_to_xinput(r1, c1, dz, invy)
            rx = map_linear_u8_to_xinput(r2, c2, dz, False)
            ry = map_linear_u8_to_xinput(r3, c3, dz, invy)

        # apply axes
        self.pad.left_joystick(x_value=lx, y_value=ly)
        self.pad.right_joystick(x_value=rx, y_value=ry)

        # buttons from packet snapshot (no hacks)
        for n, b in (("A",XUSB_BUTTON.XUSB_GAMEPAD_A),
                     ("B",XUSB_BUTTON.XUSB_GAMEPAD_B),
                     ("1",XUSB_BUTTON.XUSB_GAMEPAD_X),
                     ("2",XUSB_BUTTON.XUSB_GAMEPAD_Y),
                     ("LB",XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER),
                     ("RB",XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER),
                     ("TAKEOFF",XUSB_BUTTON.XUSB_GAMEPAD_START)):
            (self.pad.press_button if self._btnbit_from_pkt(pkt, n) else self.pad.release_button)(b)

        # triggers from packet snapshot
        self.pad.left_trigger(value=self._trig_from_pkt(pkt, "lt"))
        self.pad.right_trigger(value=self._trig_from_pkt(pkt, "rt"))

        self.pad.update()

    def start(self):
        if getattr(self, "_thr", None):
            return
        self._run.set()
        self._thr = threading.Thread(target=self._loop, name="pad-consumer", daemon=True)
        self._thr.start()

    def stop(self):
        self._run.clear()
        thr = getattr(self, "_thr", None)
        if thr:
            thr.join(timeout=1.0)
        self._thr = None

    def _loop(self):
        _bump_priority_and_timer()
        try:
            nxt = time.perf_counter()
            while self._run.is_set():
                self._tick()
                nxt += CONSUMER_DT
                dt = nxt - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    nxt = time.perf_counter()
        finally:
            _restore_timer()

# ---------- Keepalive ----------
async def heartbeat(cli: BleakClient, writable):
    if not HB_ENABLE or not writable:
        return
    ch = writable[0]
    while getattr(cli, "is_connected", False):
        try:
            try: await cli.write_gatt_char(ch, HB_PAYLOAD, response=False)
            except: await cli.write_gatt_char(ch, HB_PAYLOAD, response=True)
        except Exception:
            pass
        await asyncio.sleep(HB_INTERVAL)

# ---------- Main loop ----------
async def run(conf: Dict[str, Any]):
    prefix, uuid = conf["prefix"], conf["uuid"]

    evt = asyncio.Event(); found: Optional[BLEDevice] = None
    def on_adv(d: BLEDevice, a: AdvertisementData):
        nonlocal found
        if is_target(d, a, prefix): found = d; evt.set()

    sc = BleakScanner(on_adv)
    await sc.start()
    try:
        print(f"[i] Scanning for '{prefix}' … Ctrl+C to stop.")
        while True:
            found = None; evt.clear()
            try: await asyncio.wait_for(evt.wait(), timeout=10.0)
            except asyncio.TimeoutError: continue
            if not found: continue

            bridge = Bridge(conf)
            try:
                async with BleakClient(found, timeout=12.0) as cli:
                    # Gather any writable char for keepalive
                    if not cli.services or len(list(cli.services)) == 0:
                        getter = getattr(cli, "get_services", None)
                        if callable(getter):
                            with contextlib.suppress(Exception):
                                await getter()
                    writable = []
                    try:
                        for s in cli.services:
                            for c in s.characteristics:
                                props = set(c.properties)
                                if "write_without_response" in props or "write" in props:
                                    writable.append(c.uuid)
                    except Exception:
                        pass

                    ka = asyncio.create_task(heartbeat(cli, writable)) if writable else None
                    bridge.start()

                    await cli.start_notify(uuid, lambda _h, data: bridge.on_notify(bytes(data)))

                    print("[i] Subscribed. Consumer thread running at", CONSUMER_HZ, "Hz.")
                    while getattr(cli, "is_connected", False):
                        await asyncio.sleep(0.25)

                    bridge.stop()
                    if ka:
                        ka.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await ka

            except Exception as e:
                print(f"[!] Connect error: {e} — retrying…")
                await asyncio.sleep(1.5)
    finally:
        await sc.stop()

if __name__ == "__main__":
    try:
        conf = read_conf()
        asyncio.run(run(conf))
    except KeyboardInterrupt:
        print("\nBye")
