#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flypad2joy_fast.py
- Performance-tuned bridge: BLE -> ViGEm X360 (vgamepad).
- No calibration; reads flypad.conf (same format as before).
- If flypad.conf is missing, writes a template and exits.
"""

import asyncio, os, sys, time, configparser
from typing import Dict, Any, Optional, Tuple, Callable, List
from bleak import BleakScanner, BleakClient, BLEDevice, AdvertisementData
from vgamepad import VX360Gamepad, XUSB_BUTTON

CONF_PATH = os.path.join(os.path.dirname(__file__), "flypad.conf")
EXPECTED_MIN_LEN = 6
HB_ENABLE, HB_INTERVAL, HB_PAYLOAD = True, 0.5, b"\x01"

TEMPLATE = """[general]
target_name_prefix = FLYPAD_
ctrl_char_uuid = 9e35fa01-4344-44d4-a2e2-0c7f6046878b
invert_y = true
button_lsb_index = 1
button_msb_index = 2

[axes]
lx = -1
ly = -1
rx = -1
ry = -1

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
"""

# ---------- Config I/O ----------

def ensure_conf() -> None:
    if not os.path.exists(CONF_PATH):
        with open(CONF_PATH, "w", encoding="utf-8") as f: f.write(TEMPLATE)
        print(f"[i] Template written: {CONF_PATH}")
        print("    -> Fill [axes] (lx/ly/rx/ry) with your 0-based packet indices.")
        sys.exit(0)

def read_conf() -> Dict[str, Any]:
    ensure_conf()
    cp = configparser.ConfigParser()
    cp.read(CONF_PATH, encoding="utf-8")
    if not cp.has_section("general") or not cp.has_section("axes"):
        print(f"[!] Invalid config. Edit {CONF_PATH} (need [general] and [axes])."); sys.exit(1)

    g   = dict(cp.items("general"))
    ax  = dict(cp.items("axes"))
    cen = dict(cp.items("axis_centers")) if cp.has_section("axis_centers") else {}
    fl  = dict(cp.items("filters"))      if cp.has_section("filters")      else {}
    btn = dict(cp.items("buttons"))      if cp.has_section("buttons")      else {}
    tr  = dict(cp.items("triggers"))     if cp.has_section("triggers")     else {}

    def geti(d, k, dflt): 
        try: return int(d.get(k, dflt))
        except: return dflt
    def getf(d, k, dflt):
        try: return float(d.get(k, dflt))
        except: return dflt
    def getb(d, k, dflt=True):
        v = str(d.get(k, dflt)).lower()
        return v in ("1","true","yes","on")

    conf = {
        "prefix": g.get("target_name_prefix", "FLYPAD_"),
        "uuid": g.get("ctrl_char_uuid", "9e35fa01-4344-44d4-a2e2-0c7f6046878b"),
        "invert_y": getb(g, "invert_y", True),
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
            bit, msb = (int(x) for x in v.split(",", 1))
        except:
            bit, msb = (-1, 0)
        conf["buttons"][k.upper()] = (bit, msb)

    # sanity
    for k in ("lx","ly","rx","ry"):
        if conf["axes"][k] < 0:
            print(f"[!] Missing axis index: {k}. Edit {CONF_PATH} -> [axes]."); sys.exit(1)
    if conf["lsb_idx"] < 0:
        print(f"[!] Missing button_lsb_index in [general]. Edit {CONF_PATH}."); sys.exit(1)
    return conf

# ---------- Filters (micro-optimized) ----------

class AxisFilter:
    __slots__ = ("invert","a_s","a_f","jump","dz","hy","sg","center","ema","last","prev","sp_th","armed")
    def __init__(self, invert: bool, a_s=0.35, a_f=0.85, jump=10.0, dz=3, hy=1, sg=2, center=128):
        self.invert=invert; self.a_s=a_s; self.a_f=a_f; self.jump=jump
        self.dz=dz; self.hy=hy; self.sg=sg; self.center=center
        self.ema=None; self.last=0; self.prev=None; self.sp_th=18; self.armed=False

    def map(self, raw_u8: int) -> int:
        v = raw_u8
        if v < 0: v = 0
        elif v > 255: v = 255

        pv = self.prev
        if pv is None:
            self.prev = float(v); self.ema = float(v)
        else:
            dv = abs(v - pv)
            if dv > self.sp_th:
                if not self.armed: self.armed = True; v = int(pv)
                else: self.armed = False
            else:
                self.armed = False
            self.prev = float(v)
            a = self.a_f if abs(self.ema - v) > self.jump else self.a_s
            self.ema = a * v + (1.0 - a) * self.ema  # float

        raw = (255.0 - self.ema) if self.invert else self.ema
        d = raw - self.center

        last = self.last
        in_zero = (last == 0)
        leave = self.dz + (self.hy if in_zero else 0)
        ad = d if d >= 0 else -d
        if (in_zero and ad <= leave) or (not in_zero and ad <= self.dz):
            self.last = 0; return 0

        # sign guard near 0
        if (last > 0 and d < 0 and ad < self.sg) or (last < 0 and d > 0 and ad < self.sg):
            d = ad if last > 0 else -ad

        pos = d >= 0
        mag = ad - self.dz
        if mag < 0: mag = 0.0
        span_pos = (255.0 - self.center - self.dz);  span_pos = 1.0 if span_pos <= 1.0 else span_pos
        span_neg = (self.center - self.dz);           span_neg = 1.0 if span_neg <= 1.0 else span_neg
        full = 32767 if pos else 32768
        out = int(mag * full / (span_pos if pos else span_neg))
        out = (32767 if out > 32767 else out) if pos else -(32768 if out > 32768 else out)
        self.last = out
        return out

# ---------- Bridge ----------

def is_target(d: BLEDevice, a: AdvertisementData, prefix: str) -> bool:
    return (d.name or "").startswith(prefix)

class Bridge:
    """Precompiled, allocation-safe mapping loop."""
    __slots__ = ("pad","idx","f","lsb","msb","btn_masks","btn_targets","tr_l","tr_r")

    def __init__(self, conf: Dict[str, Any]):
        self.pad = VX360Gamepad()

        ax, cen, fl = conf["axes"], conf["centers"], conf["filters"]
        invy = conf["invert_y"]
        self.idx = (ax["lx"], ax["ly"], ax["rx"], ax["ry"])
        self.f = (
            AxisFilter(False, fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["lx"]),
            AxisFilter(invy,  fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["ly"]),
            AxisFilter(False, fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["rx"]),
            AxisFilter(invy,  fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["ry"]),
        )
        self.lsb = conf["lsb_idx"]
        self.msb = conf.get("msb_idx", -1)

        # Precompile buttons -> (mask_on_lsb, mask_on_msb, target_button)
        bmap = conf["buttons"]
        M = XUSB_BUTTON
        pairs = (
            ("A", M.XUSB_GAMEPAD_A),
            ("B", M.XUSB_GAMEPAD_B),
            ("1", M.XUSB_GAMEPAD_X),
            ("2", M.XUSB_GAMEPAD_Y),
            ("LB",M.XUSB_GAMEPAD_LEFT_SHOULDER),
            ("RB",M.XUSB_GAMEPAD_RIGHT_SHOULDER),
            ("TAKEOFF",M.XUSB_GAMEPAD_START),
        )
        masks_lsb: List[Tuple[int,int]] = []
        masks_msb: List[Tuple[int,int]] = []
        targets:   List[int] = []
        for name, tgt in pairs:
            bit, msb = bmap.get(name, (-1,0))
            if bit >= 0:
                if msb: masks_msb.append((1 << bit, tgt))
                else:   masks_lsb.append((1 << bit, tgt))
                targets.append(tgt)
        self.btn_masks = (tuple(masks_lsb), tuple(masks_msb))
        self.btn_targets = tuple(t for _, t in masks_lsb + masks_msb)  # for releases

        # Precompile triggers
        def prep_tr(side: str):
            mode = conf["triggers"].get(f"{side}_mode", "bit")
            if mode == "byte":
                idx = int(conf["triggers"].get(f"{side}_byte", -1))
                inv = int(conf["triggers"].get(f"{side}_invert", 0))
                return ("byte", idx, inv)
            else:
                bit = int(conf["triggers"].get(f"{side}_bit", -1))
                msb = int(conf["triggers"].get(f"{side}_msb", 0))
                sel = self.msb if msb else self.lsb
                return ("bit", bit, sel)
        self.tr_l = prep_tr("lt")
        self.tr_r = prep_tr("rt")

    def feed(self, pkt: bytes):
        # Axis
        f0,f1,f2,f3 = self.f
        i0,i1,i2,i3 = self.idx
        self.pad.left_joystick(  x_value=f0.map(pkt[i0]), y_value=f1.map(pkt[i1]) )
        self.pad.right_joystick( x_value=f2.map(pkt[i2]), y_value=f3.map(pkt[i3]) )

        # Buttons (mask test only on configured bytes)
        ml, mm = self.btn_masks
        lsb_idx, msb_idx = self.lsb, self.msb
        lsb = pkt[lsb_idx] if 0 <= lsb_idx < len(pkt) else 0
        msb = pkt[msb_idx] if 0 <= msb_idx < len(pkt) else 0

        # Press/release
        pad = self.pad
        for mask, tgt in ml:
            if lsb & mask: pad.press_button(tgt)
            else:          pad.release_button(tgt)
        for mask, tgt in mm:
            if msb & mask: pad.press_button(tgt)
            else:          pad.release_button(tgt)

        # Triggers
        tL = self.tr_l; tR = self.tr_r
        if tL[0] == "byte":
            idx, inv = tL[1], tL[2]
            val = pkt[idx] if 0 <= idx < len(pkt) else 0
            pad.left_trigger( value=(255 - val) if inv else val )
        else:
            bit, sel = tL[1], tL[2]
            val = (pkt[sel] >> bit) & 1 if (bit >= 0 and 0 <= sel < len(pkt)) else 0
            pad.left_trigger( value=255 if val else 0 )

        if tR[0] == "byte":
            idx, inv = tR[1], tR[2]
            val = pkt[idx] if 0 <= idx < len(pkt) else 0
            pad.right_trigger( value=(255 - val) if inv else val )
        else:
            bit, sel = tR[1], tR[2]
            val = (pkt[sel] >> bit) & 1 if (bit >= 0 and 0 <= sel < len(pkt)) else 0
            pad.right_trigger( value=255 if val else 0 )

        pad.update()

# ---------- Heartbeat ----------

async def heartbeat(cli: BleakClient, ch_uuid: Optional[str]):
    if not HB_ENABLE or not ch_uuid: return
    while getattr(cli, "is_connected", False):
        try:
            try: await cli.write_gatt_char(ch_uuid, HB_PAYLOAD, response=False)
            except: await cli.write_gatt_char(ch_uuid, HB_PAYLOAD, response=True)
        except: pass
        await asyncio.sleep(HB_INTERVAL)

# ---------- Main ----------

def is_target(d: BLEDevice, a: AdvertisementData, prefix: str) -> bool:
    return (d.name or "").startswith(prefix)

async def run(conf: Dict[str, Any]):
    prefix, uuid = conf["prefix"], conf["uuid"]
    evt = asyncio.Event(); dev: Optional[BLEDevice] = None

    def on_adv(d: BLEDevice, a: AdvertisementData):
        nonlocal dev
        if is_target(d, a, prefix):
            dev = d; evt.set()

    sc = BleakScanner(on_adv)
    await sc.start()
    try:
        print(f"Scanning for '{prefix}' … Ctrl+C to stop.")
        while True:
            dev = None; evt.clear()
            try: await asyncio.wait_for(evt.wait(), timeout=10.0)
            except asyncio.TimeoutError: continue
            if not dev: continue

            bridge = Bridge(conf)
            try:
                async with BleakClient(dev, timeout=10.0) as cli:
                    # Find one writeable char (optional heartbeat)
                    hb_char = None
                    try:
                        if not cli.services or len(list(cli.services)) == 0:
                            gs = getattr(cli, "get_services", None)
                            if callable(gs): await gs()
                        for s in cli.services:
                            for c in s.characteristics:
                                props = c.properties
                                if ("write_without_response" in props) or ("write" in props):
                                    hb_char = c.uuid; break
                            if hb_char: break
                    except: pass

                    volatile_pkt: Optional[bytes] = None
                    def _cb(_, data: bytearray):
                        # Avoid extra copy: store as bytes once per notify
                        nonlocal volatile_pkt
                        volatile_pkt = bytes(data)

                    await cli.start_notify(uuid, _cb)
                    # Wait first packet
                    t0 = time.time()
                    while volatile_pkt is None and time.time() - t0 < 3.0:
                        await asyncio.sleep(0.01)
                    if volatile_pkt is None or len(volatile_pkt) < EXPECTED_MIN_LEN:
                        print("[!] Short/no packet; check characteristic UUID."); continue

                    hb_task = asyncio.create_task(heartbeat(cli, hb_char)) if hb_char else None

                    # Hot loop — keep everything local
                    feed = bridge.feed
                    getpkt = lambda: volatile_pkt
                    sleep = asyncio.sleep
                    while getattr(cli, "is_connected", False):
                        pkt = getpkt()
                        if pkt is not None:
                            feed(pkt)
                        await sleep(0.005)  # ~200 Hz loop

                    if hb_task: hb_task.cancel()
            except Exception as e:
                print(f"[!] Connect error: {e} — retrying…"); await asyncio.sleep(1.5)
    finally:
        await sc.stop()

if __name__ == "__main__":
    try:
        conf = read_conf()
        asyncio.run(run(conf))
    except KeyboardInterrupt:
        print("\nBye")
