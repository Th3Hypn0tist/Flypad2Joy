#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flypad2joy_anti_spike_robust.py
- BLE -> ViGEm (Xbox 360) bridge (Windows).
- Anti-spike v2: median-of-3, edge-spike clamp, all-zero frame drop, idle center lock.
- No battery readout. Optional keepalive write.

Config file: flypad.conf  (see [filters] for new knobs)
"""

import asyncio, os, time, sys, configparser, contextlib, statistics
from typing import Dict, Any, Tuple
from collections import deque
from bleak import BleakScanner, BleakClient, BLEDevice, AdvertisementData
from vgamepad import VX360Gamepad, XUSB_BUTTON

CONF_PATH = os.path.join(os.path.dirname(__file__), "flypad.conf")
EXPECTED_MIN_LEN = 6

# Optional BLE keepalive payload
HB_ENABLE, HB_INTERVAL, HB_PAYLOAD = True, 0.5, b"\x01"

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
A = 3,0
B = 4,0
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
alpha_slow = 0.35
alpha_fast = 0.85
jump_threshold = 10
deadzone = 3
hysteresis = 1
sign_guard = 2

; --- robust anti-spike extras ---
spike_idle_ms = 700       ; idle -> soft-start window
spike_samples = 3
spike_raw_step = 6        ; max raw delta/sample during settle
spike_out_step = 8000     ; max output delta/sample during settle
alpha_idle = 0.20         ; slower "fast" alpha just after idle

median_window = 3         ; median prefilter window (3 recommended)
edge_spike_dv = 50        ; treat jumps > this as spikes to clamp (raw units)
drop_all_zero_frame = true ; skip frame if ALL axes are 0 (idle/glitch)
center_lock_window = 2    ; if returning from idle and raw≈center, snap EMA to center
"""

# ---------- Config ----------

def ensure_conf() -> None:
    if not os.path.exists(CONF_PATH):
        with open(CONF_PATH, "w", encoding="utf-8") as f:
            f.write(TEMPLATE)
        print(f"[i] Template written: {CONF_PATH}")
        print("    -> Fill [axes] lx/ly/rx/ry if needed.")
        sys.exit(0)

def read_conf() -> Dict[str, Any]:
    ensure_conf()
    cp = configparser.ConfigParser()
    cp.read(CONF_PATH, encoding="utf-8")

    if not cp.has_section("general") or not cp.has_section("axes"):
        print(f"[!] Invalid config. Edit {CONF_PATH} (need [general] and [axes]).")
        sys.exit(1)

    g   = dict(cp.items("general"))
    ax  = dict(cp.items("axes"))
    cen = dict(cp.items("axis_centers")) if cp.has_section("axis_centers") else {}
    fl  = dict(cp.items("filters"))      if cp.has_section("filters")      else {}
    btn = dict(cp.items("buttons"))      if cp.has_section("buttons")      else {}
    tr  = dict(cp.items("triggers"))     if cp.has_section("triggers")     else {}

    def b2(x, dflt=False):
        s = str(x).strip().lower()
        return dflt if s=="" else (s in ("1","true","yes","on"))
    def geti(d, k, dflt):
        try: return int(d.get(k, dflt))
        except: return dflt
    def getf(d, k, dflt):
        try: return float(d.get(k, dflt))
        except: return dflt

    conf = {
        "prefix": g.get("target_name_prefix", "FLYPAD_"),
        "uuid": g.get("ctrl_char_uuid", "9e35fa01-4344-44d4-a2e2-0c7f6046878b"),
        "invert_y": b2(g.get("invert_y", True), True),
        "lsb_idx": geti(g, "button_lsb_index", -1),
        "msb_idx": geti(g, "button_msb_index", -1),
        "axes": {
            "lx": geti(ax, "lx", -1),
            "ly": geti(ax, "ly", -1),
            "rx": geti(ax, "rx", -1),
            "ry": geti(ax, "ry", -1),
        },
        "centers": {
            "lx": geti(cen, "lx", 128),
            "ly": geti(cen, "ly", 128),
            "rx": geti(cen, "rx", 128),
            "ry": geti(cen, "ry", 128),
        },
        "filters": {
            "a_s": getf(fl, "alpha_slow", 0.35),
            "a_f": getf(fl, "alpha_fast", 0.85),
            "jump": getf(fl, "jump_threshold", 10.0),
            "dz":  geti(fl, "deadzone", 3),
            "hy":  geti(fl, "hysteresis", 1),
            "sg":  geti(fl, "sign_guard", 2),

            "sp_idle_ms": geti(fl, "spike_idle_ms", 700),
            "sp_samples": geti(fl, "spike_samples", 3),
            "sp_raw_step": geti(fl, "spike_raw_step", 6),
            "sp_out_step": geti(fl, "spike_out_step", 8000),
            "alpha_idle": getf(fl, "alpha_idle", 0.20),

            "median_window": max(1, geti(fl, "median_window", 3)),
            "edge_spike_dv": geti(fl, "edge_spike_dv", 50),
            "drop_all_zero_frame": b2(fl.get("drop_all_zero_frame", True), True),
            "center_lock_window": geti(fl, "center_lock_window", 2),
        },
        "buttons": {},
        "triggers": {
            "lt_mode": tr.get("lt_mode", "bit"),
            "lt_bit":  tr.get("lt_bit",  "-1"),
            "lt_msb":  tr.get("lt_msb",  "0"),
            "lt_byte": tr.get("lt_byte", "-1"),
            "lt_invert": tr.get("lt_invert", "0"),
            "rt_mode": tr.get("rt_mode", "bit"),
            "rt_bit":  tr.get("rt_bit",  "-1"),
            "rt_msb":  tr.get("rt_msb",  "0"),
            "rt_byte": tr.get("rt_byte", "-1"),
            "rt_invert": tr.get("rt_invert", "0"),
        }
    }

    for k, v in btn.items():
        try:
            bit_str, msb_str = [x.strip() for x in v.split(",", 1)]
            conf["buttons"][k.upper()] = (int(bit_str), int(msb_str))
        except:
            conf["buttons"][k.upper()] = (-1, 0)

    for k in ("lx","ly","rx","ry"):
        if conf["axes"][k] < 0:
            print(f"[!] Missing axis index: {k}. Edit {CONF_PATH} -> [axes].")
            sys.exit(1)
    if conf["lsb_idx"] < 0:
        print(f"[!] Missing button_lsb_index in [general]. Edit {CONF_PATH}.")
        sys.exit(1)
    return conf

# ---------- Input filtering (robust anti-spike) ----------

class AxisFilter:
    """
    EMA + deadzone + hysteresis + spike guards with:
    - idle-aware soft start (clamped raw & output deltas)
    - median-of-N prefilter
    - edge-spike clamp
    - idle center lock
    """
    __slots__ = (
        "invert","a_s","a_f","jump","dz","hy","sg","center",
        "ema","last","prev","sp_th","armed",
        "sp_idle","sp_samples","sp_raw_step","sp_out_step","alpha_idle",
        "settle","last_ts","buf","edge_spike_dv","center_lock_window"
    )
    def __init__(
        self, invert: bool, a_s=0.35, a_f=0.85, jump=10.0, dz=3, hy=1, sg=2, center=128,
        sp_idle_ms=700, sp_samples=3, sp_raw_step=6, sp_out_step=8000, alpha_idle=0.20,
        median_window=3, edge_spike_dv=50, center_lock_window=2
    ):
        self.invert=invert; self.a_s=a_s; self.a_f=a_f; self.jump=jump
        self.dz=dz; self.hy=hy; self.sg=sg; self.center=center
        self.ema=None; self.last=0; self.prev=None; self.sp_th=18; self.armed=False

        self.sp_idle = sp_idle_ms/1000.0
        self.sp_samples = int(max(0, sp_samples))
        self.sp_raw_step = float(max(0, sp_raw_step))
        self.sp_out_step = int(max(0, sp_out_step))
        self.alpha_idle = float(alpha_idle)

        self.settle = 0
        self.last_ts = time.time()
        self.buf = deque(maxlen=max(1,int(median_window)))
        self.edge_spike_dv = max(0,int(edge_spike_dv))
        self.center_lock_window = max(0,int(center_lock_window))

    def _median(self, v: float) -> float:
        self.buf.append(float(v))
        if len(self.buf) == 1:
            return self.buf[0]
        try:
            return float(statistics.median(self.buf))
        except Exception:
            # fallback: middle of sorted
            b = sorted(self.buf)
            return float(b[len(b)//2])

    def map(self, raw_u8: int) -> int:
        now = time.time()
        idle_dt = now - self.last_ts
        if idle_dt > self.sp_idle:
            self.settle = self.sp_samples
        self.last_ts = now

        # median prefilter (and edge spike clamp)
        v_raw = float(max(0, min(255, raw_u8)))
        v_med = self._median(v_raw)

        # clamp huge single-sample jumps (raw side)
        if self.prev is not None and self.edge_spike_dv > 0:
            dv = v_med - self.prev
            if dv >  self.edge_spike_dv: v_med = self.prev + self.edge_spike_dv
            if dv < -self.edge_spike_dv: v_med = self.prev - self.edge_spike_dv

        # clamp raw after idle to avoid first-sample jump
        if self.prev is not None and self.settle > 0 and self.sp_raw_step > 0:
            dv = v_med - self.prev
            if dv >  self.sp_raw_step: v_med = self.prev + self.sp_raw_step
            if dv < -self.sp_raw_step: v_med = self.prev - self.sp_raw_step

        v = v_med

        # idle center lock: if we JUST came back from idle and we're near center, snap EMA to center
        if idle_dt > self.sp_idle and abs(v - self.center) <= self.center_lock_window:
            self.ema = float(self.center)
            self.last = 0
            self.prev = v
            # keep settle active for soft start
        else:
            # spike-guard (two-stage toggle) on raw changes
            if self.prev is None:
                self.prev = v
            else:
                dv = abs(v - self.prev)
                if dv > self.sp_th and not self.armed:
                    self.armed = True; v = self.prev
                elif dv > self.sp_th and self.armed:
                    self.armed = False
                else:
                    self.armed = False
                self.prev = v

            # smoothing (slower just after idle)
            if self.ema is None:
                self.ema = v
            else:
                a_fast = self.alpha_idle if self.settle>0 else self.a_f
                a = a_fast if abs(v-self.ema)>self.jump else self.a_s
                self.ema = a*v + (1-a)*self.ema

        # map to XInput range with deadzone/hysteresis/sign-guard
        raw = 255.0 - self.ema if self.invert else self.ema
        d = raw - self.center
        in_zero = (self.last == 0)
        leave = self.dz + (self.hy if in_zero else 0)
        if (in_zero and abs(d) <= leave) or (not in_zero and abs(d) <= self.dz):
            out = 0
        else:
            if (self.last>0 and d<0 and abs(d)<self.sg) or (self.last<0 and d>0 and abs(d)<self.sg):
                d = abs(d) * (1 if self.last>0 else -1)
            sign = 1 if d>=0 else -1
            mag = max(0.0, abs(d)-self.dz)
            span_pos = max(1.0, 255.0 - self.center - self.dz)
            span_neg = max(1.0, self.center - self.dz)
            full = 32767 if sign>0 else 32768
            out = int(round(mag * full / (span_pos if sign>0 else span_neg)))
            out = min(32767, out) if sign>0 else -min(32768, out)

        # clamp output delta during settle
        if self.settle>0 and self.sp_out_step>0:
            delta = out - self.last
            if delta >  self.sp_out_step: out = self.last + self.sp_out_step
            if delta < -self.sp_out_step: out = self.last - self.sp_out_step
            self.settle -= 1

        self.last = out
        return out

# ---------- BLE & bridge ----------

def is_target(d: BLEDevice, a: AdvertisementData, prefix: str) -> bool:
    return (d.name or "").startswith(prefix)

class Bridge:
    """Map BLE packet bytes to ViGEm Xbox 360 gamepad via vgamepad."""
    def __init__(self, conf: Dict[str, Any]):
        self.pad = VX360Gamepad()
        ax, cen, fl = conf["axes"], conf["centers"], conf["filters"]
        invy = conf["invert_y"]
        self.idx = (ax["lx"], ax["ly"], ax["rx"], ax["ry"])
        self.centers = (cen["lx"], cen["ly"], cen["rx"], cen["ry"])
        self.drop_all_zero = fl["drop_all_zero_frame"]
        self.f = [
            AxisFilter(False, fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["lx"],
                       fl["sp_idle_ms"], fl["sp_samples"], fl["sp_raw_step"], fl["sp_out_step"], fl["alpha_idle"],
                       fl["median_window"], fl["edge_spike_dv"], fl["center_lock_window"]),
            AxisFilter(invy,  fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["ly"],
                       fl["sp_idle_ms"], fl["sp_samples"], fl["sp_raw_step"], fl["sp_out_step"], fl["alpha_idle"],
                       fl["median_window"], fl["edge_spike_dv"], fl["center_lock_window"]),
            AxisFilter(False, fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["rx"],
                       fl["sp_idle_ms"], fl["sp_samples"], fl["sp_raw_step"], fl["sp_out_step"], fl["alpha_idle"],
                       fl["median_window"], fl["edge_spike_dv"], fl["center_lock_window"]),
            AxisFilter(invy,  fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["ry"],
                       fl["sp_idle_ms"], fl["sp_samples"], fl["sp_raw_step"], fl["sp_out_step"], fl["alpha_idle"],
                       fl["median_window"], fl["edge_spike_dv"], fl["center_lock_window"]),
        ]
        self.lsb = conf["lsb_idx"]
        self.msb = conf.get("msb_idx", -1)
        self.buttons = conf["buttons"]
        self.tr = conf["triggers"]

    def _btnbit(self, pkt: bytes, name: str) -> int:
        tup = self.buttons.get(name)
        if not tup: return 0
        bit, msb = tup
        if bit < 0: return 0
        idx = self.msb if msb else self.lsb
        if idx < 0 or idx >= len(pkt): return 0
        return (pkt[idx] >> bit) & 1

    def _trig(self, pkt: bytes, which: str) -> int:
        mode = self.tr.get(f"{which}_mode", "bit")
        if mode == "byte":
            idx = int(self.tr.get(f"{which}_byte", -1))
            inv = int(self.tr.get(f"{which}_invert", 0))
            if 0 <= idx < len(pkt):
                v = pkt[idx]; return (255 - v) if inv else v
            return 0
        bit = int(self.tr.get(f"{which}_bit", -1))
        msb = int(self.tr.get(f"{which}_msb", 0))
        idx = self.msb if msb else self.lsb
        if bit < 0 or idx < 0 or idx >= len(pkt): return 0
        return 255 if ((pkt[idx] >> bit) & 1) else 0

    def _get_axis_bytes(self, pkt: bytes) -> Tuple[int,int,int,int]:
        i0,i1,i2,i3 = self.idx
        return (pkt[i0], pkt[i1], pkt[i2], pkt[i3])

    def feed(self, pkt: bytes):
        # Drop frames where ALL axes are 0 (typical idle/glitch from some devices)
        if self.drop_all_zero:
            try:
                a0,a1,a2,a3 = self._get_axis_bytes(pkt)
                if (a0|a1|a2|a3) == 0:
                    return
            except Exception:
                # if packet too short or index error, just bail this frame
                return

        lx = self.f[0].map(pkt[self.idx[0]])
        ly = self.f[1].map(pkt[self.idx[1]])
        rx = self.f[2].map(pkt[self.idx[2]])
        ry = self.f[3].map(pkt[self.idx[3]])

        self.pad.left_joystick(x_value=lx, y_value=ly)
        self.pad.right_joystick(x_value=rx, y_value=ry)

        for n, b in (("A",XUSB_BUTTON.XUSB_GAMEPAD_A),
                     ("B",XUSB_BUTTON.XUSB_GAMEPAD_B),
                     ("1",XUSB_BUTTON.XUSB_GAMEPAD_X),
                     ("2",XUSB_BUTTON.XUSB_GAMEPAD_Y),
                     ("LB",XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER),
                     ("RB",XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER),
                     ("TAKEOFF",XUSB_BUTTON.XUSB_GAMEPAD_START)):
            (self.pad.press_button if self._btnbit(pkt, n) else self.pad.release_button)(b)

        self.pad.left_trigger(value=self._trig(pkt, "lt"))
        self.pad.right_trigger(value=self._trig(pkt, "rt"))
        self.pad.update()

# ---------- Keepalive ----------

async def heartbeat(cli: BleakClient, writable):
    if not HB_ENABLE or not writable: return
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
    evt = asyncio.Event(); found = None

    def on_adv(d: BLEDevice, a: AdvertisementData):
        nonlocal found
        if is_target(d, a, prefix): found = d; evt.set()

    sc = BleakScanner(on_adv)
    await sc.start()
    try:
        print(f"Scanning for '{prefix}' … Press Ctrl+C to stop.")
        while True:
            found = None; evt.clear()
            try: await asyncio.wait_for(evt.wait(), timeout=10.0)
            except asyncio.TimeoutError: continue
            if not found: continue

            bridge = Bridge(conf)
            try:
                async with BleakClient(found, timeout=10.0) as cli:
                    # Cache services so we can locate any writable char for keepalive
                    if not cli.services or len(list(cli.services)) == 0:
                        getter = getattr(cli, "get_services", None)
                        if callable(getter):
                            try: await getter()
                            except Exception: pass

                    # Collect writable char UUIDs for optional heartbeat
                    writable = []
                    try:
                        for s in cli.services:
                            for c in s.characteristics:
                                props = set(c.properties)
                                if "write_without_response" in props or "write" in props:
                                    writable.append(c.uuid)
                    except Exception:
                        pass

                    hb_task = asyncio.create_task(heartbeat(cli, writable)) if writable else None

                    # Subscribe and wait for first packet
                    latest = bytearray()
                    await cli.start_notify(uuid, lambda _, d: latest.__init__(bytearray(d)))
                    t0 = time#!/usr/bin/env python3
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
A = 3,0
B = 4,0
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
.time()
                    while len(latest) == 0 and time.time() - t0 < 3.0:
                        await asyncio.sleep(0.05)
                    if len(latest) < EXPECTED_MIN_LEN:
                        print("[!] Short packet; verify characteristic UUID in config.")
                        if hb_task:
                            hb_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await hb_task
                        continue

                    # Main feed loop
                    while getattr(cli, "is_connected", False):
                        if latest:
                            bridge.feed(bytes(latest))
                        await asyncio.sleep(0.01)

                    # Cleanup
                    if hb_task:
                        hb_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await hb_task

            except Exception as e:
                print(f"[!] Connect error: {e} — retrying…")
                await asyncio.sleep(2)
    finally:
        await sc.stop()

if __name__ == "__main__":
    try:
        conf = read_conf()
        asyncio.run(run(conf))
    except KeyboardInterrupt:
        print("\nBye")
