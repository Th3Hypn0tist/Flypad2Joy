#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flypad2joy.py
- Reads configuration only (no auto-calibration).
- If flypad.conf is missing, writes a template and exits.
- Bridges BLE packets -> ViGEm Xbox 360 virtual gamepad.
- Adds Battery Service (0x180F / 0x2A19) readout + periodic battery monitoring.
- Comments & prompts in English.
"""

import asyncio, os, time, sys, configparser, contextlib
from typing import Dict, Any
from bleak import BleakScanner, BleakClient, BLEDevice, AdvertisementData
from vgamepad import VX360Gamepad, XUSB_BUTTON

CONF_PATH = os.path.join(os.path.dirname(__file__), "flypad.conf")
EXPECTED_MIN_LEN = 6

# Optional BLE keepalive payload (if your device expects any write to stay awake)
HB_ENABLE, HB_INTERVAL, HB_PAYLOAD = True, 0.5, b"\x01"

# Battery Service
BAT_UUID      = "00002a19-0000-1000-8000-00805f9b34fb"  # Battery Level (uint8 0..100)
BAT_INTERVAL  = 60.0   # seconds
BAT_WARN_PCT  = 20     # warn under this
BAT_ENABLE    = True   # set False to disable battery reads

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
; Centers (0..255). 128 is usually fine.
lx = 128
ly = 128
rx = 128
ry = 128

[buttons]
; NAME = bit,msb  (msb: 0 = LSB byte, 1 = MSB byte)
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
; Bit-mode (digital) triggers from your mapping
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
; Motion feel (tune if you like)
alpha_slow = 0.35
alpha_fast = 0.85
jump_threshold = 10
deadzone = 3
hysteresis = 1
sign_guard = 2
"""

# ---------- Config ----------

def ensure_conf() -> None:
    if not os.path.exists(CONF_PATH):
        with open(CONF_PATH, "w", encoding="utf-8") as f:
            f.write(TEMPLATE)
        print(f"[i] Template written: {CONF_PATH}")
        print("    -> Open it and fill [axes] (lx/ly/rx/ry) with your 0-based packet indices.")
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

# ---------- Input filtering ----------

class AxisFilter:
    """Adaptive EMA + deadzone + hysteresis + small spike guard for 0..255 -> XInput range."""
    def __init__(self, invert: bool, a_s=0.35, a_f=0.85, jump=10.0, dz=3, hy=1, sg=2, center=128):
        self.invert=invert; self.a_s=a_s; self.a_f=a_f; self.jump=jump
        self.dz=dz; self.hy=hy; self.sg=sg; self.center=center
        self.ema=None; self.last=0; self.prev=None; self.sp_th=18; self.armed=False

    def map(self, raw_u8: int) -> int:
        v=float(max(0, min(255, raw_u8)))
        if self.prev is None: self.prev=v
        else:
            dv=abs(v-self.prev)
            if dv>self.sp_th and not self.armed: self.armed=True; v=self.prev
            elif dv>self.sp_th and self.armed: self.armed=False
            else: self.armed=False
            self.prev=v
        if self.ema is None: self.ema=v
        else:
            a=self.a_f if abs(v-self.ema)>self.jump else self.a_s
            self.ema=a*v+(1-a)*self.ema
        raw=255.0-self.ema if self.invert else self.ema
        d=raw-self.center
        in_zero=(self.last==0); leave=self.dz+(self.hy if in_zero else 0)
        if (in_zero and abs(d)<=leave) or (not in_zero and abs(d)<=self.dz):
            self.last=0; return 0
        if (self.last>0 and d<0 and abs(d)<self.sg) or (self.last<0 and d>0 and abs(d)<self.sg):
            d=abs(d)*(1 if self.last>0 else -1)
        sign=1 if d>=0 else -1; mag=max(0.0,abs(d)-self.dz)
        span_pos=max(1.0,255.0-self.center-self.dz); span_neg=max(1.0,self.center-self.dz)
        full=32767 if sign>0 else 32768
        out=int(round(mag*full/(span_pos if sign>0 else span_neg)))
        out=min(32767,out) if sign>0 else -min(32768,out)
        self.last=out; return out

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
        self.f = [
            AxisFilter(False, fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["lx"]),
            AxisFilter(invy,  fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["ly"]),
            AxisFilter(False, fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["rx"]),
            AxisFilter(invy,  fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"], cen["ry"]),
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

    def feed(self, pkt: bytes):
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

# ---------- Keepalive & battery ----------

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

async def read_battery(cli: BleakClient) -> int | None:
    """Return battery percent 0..100 or None if not available."""
    if not BAT_ENABLE: return None
    try:
        data = await cli.read_gatt_char(BAT_UUID)
        if data and len(data) >= 1:
            return int(data[0])  # spec: 0..100
    except Exception:
        return None
    return None

async def battery_monitor(cli: BleakClient):
    """Poll battery level periodically and print warnings."""
    if not BAT_ENABLE: return
    while getattr(cli, "is_connected", False):
        try:
            pct = await read_battery(cli)
            if pct is not None:
                print(f"[i] Battery: {pct}%")
                if pct <= BAT_WARN_PCT:
                    print(f"[w] Battery low ({pct}%). Charge soon to avoid random disconnects.")
        except Exception:
            pass
        await asyncio.sleep(BAT_INTERVAL)

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
                    # Ensure services are cached so we can probe characteristics (battery/write)
                    if not cli.services or len(list(cli.services)) == 0:
                        getter = getattr(cli, "get_services", None)
                        if callable(getter):
                            try: await getter()
                            except Exception: pass

                    # Discover writeable characteristics for optional heartbeat
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

                    # Try reading battery once at connect and start monitor if present
                    bat_task = None
                    try:
                        pct = await read_battery(cli)
                        if pct is not None:
                            print(f"[i] Battery: {pct}%")
                            if pct <= BAT_WARN_PCT:
                                print(f"[w] Battery low ({pct}%).")
                            bat_task = asyncio.create_task(battery_monitor(cli))
                    except Exception:
                        pass

                    # Subscribe and wait for first packet
                    latest = bytearray()
                    await cli.start_notify(uuid, lambda _, d: latest.__init__(bytearray(d)))
                    t0 = time.time()
                    while len(latest) == 0 and time.time() - t0 < 3.0:
                        await asyncio.sleep(0.05)
                    if len(latest) < EXPECTED_MIN_LEN:
                        print("[!] Short packet; verify characteristic UUID in config.")
                        for t in (hb_task, bat_task):
                            if t: t.cancel()
                        continue

                    # Main feed loop
                    while getattr(cli, "is_connected", False):
                        if latest:
                            bridge.feed(bytes(latest))
                        await asyncio.sleep(0.01)

                    # Cleanup tasks
                    for t in (hb_task, bat_task):
                        if t:
                            t.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await t

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
