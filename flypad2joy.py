#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flypad2joy_rollback.py
- Minimal, stable BLE → ViGEm bridge.
- Uses config only; no auto-calibration.
- Direct notify on the configured characteristic; if it fails once, refresh services and retry.
- English prompts & comments only.
"""

import asyncio, os, sys, time, configparser
from typing import Dict, Any, Optional
from bleak import BleakScanner, BleakClient, BLEDevice, AdvertisementData
from vgamepad import VX360Gamepad, XUSB_BUTTON

CONF_PATH = os.path.join(os.path.dirname(__file__), "flypad.conf")
EXPECTED_MIN_LEN = 6  # adjust if your packet is longer/shorter

# ---------------- Config ----------------

def read_conf() -> Dict[str, Any]:
    if not os.path.exists(CONF_PATH):
        print(f"[!] {CONF_PATH} missing. Run your generator once to create the template.")
        sys.exit(1)
    cp = configparser.ConfigParser()
    cp.read(CONF_PATH, encoding="utf-8")
    if not cp.has_section("general") or not cp.has_section("axes"):
        print(f"[!] Invalid config. Edit {CONF_PATH}.")
        sys.exit(1)

    g   = dict(cp.items("general"))
    ax  = dict(cp.items("axes"))
    cen = dict(cp.items("axis_centers")) if cp.has_section("axis_centers") else {}
    flt = dict(cp.items("filters"))      if cp.has_section("filters")      else {}
    btn = dict(cp.items("buttons"))      if cp.has_section("buttons")      else {}
    trig= dict(cp.items("triggers"))     if cp.has_section("triggers")     else {}

    def geti(d,k,v): 
        try: return int(d.get(k,v))
        except: return v
    def getf(d,k,v):
        try: return float(d.get(k,v))
        except: return v
    def getb(d,k,v=True):
        return str(d.get(k,v)).lower() in ("1","true","yes","on")

    conf = {
        "prefix": g.get("target_name_prefix","FLYPAD_"),
        "uuid":   g.get("ctrl_char_uuid","9e35fa01-4344-44d4-a2e2-0c7f6046878b"),
        "invert_y": getb(g,"invert_y",True),
        "lsb": geti(g,"button_lsb_index",-1),
        "msb": geti(g,"button_msb_index",-1),
        "axes": {k:geti(ax,k,-1) for k in ("lx","ly","rx","ry")},
        "centers": {k:geti(cen,k,128) for k in ("lx","ly","rx","ry")},
        "filters": {
            "a_s": getf(flt,"alpha_slow",0.35), "a_f": getf(flt,"alpha_fast",0.85),
            "jump": getf(flt,"jump_threshold",10.0),
            "dz": geti(flt,"deadzone",3), "hy": geti(flt,"hysteresis",1), "sg": geti(flt,"sign_guard",2)
        },
        "buttons": {},
        "triggers": {
            "lt_mode": trig.get("lt_mode","bit"), "lt_bit": trig.get("lt_bit","-1"),
            "lt_msb": trig.get("lt_msb","0"), "lt_byte": trig.get("lt_byte","-1"), "lt_invert": trig.get("lt_invert","0"),
            "rt_mode": trig.get("rt_mode","bit"), "rt_bit": trig.get("rt_bit","-1"),
            "rt_msb": trig.get("rt_msb","0"), "rt_byte": trig.get("rt_byte","-1"), "rt_invert": trig.get("rt_invert","0"),
        }
    }
    for k,v in btn.items():
        try: bit,msb = (int(x) for x in v.split(",",1))
        except: bit,msb = (-1,0)
        conf["buttons"][k.upper()] = (bit,msb)

    for k in ("lx","ly","rx","ry"):
        if conf["axes"][k] < 0:
            print(f"[!] Missing axis index: {k}. Edit {CONF_PATH} -> [axes]."); sys.exit(1)
    if conf["lsb"] < 0:
        print(f"[!] Missing button_lsb_index in [general]. Edit {CONF_PATH}."); sys.exit(1)
    return conf

# ---------------- Filtering ----------------

class AxisFilter:
    """Simple stable filter: adaptive EMA + deadzone + small sign-guard."""
    __slots__=("invert","a_s","a_f","jump","dz","hy","sg","center","ema","last","prev","sp_th","armed")
    def __init__(self, invert,a_s,a_f,jump,dz,hy,sg,center):
        self.invert=invert; self.a_s=a_s; self.a_f=a_f; self.jump=jump
        self.dz=dz; self.hy=hy; self.sg=sg; self.center=center
        self.ema=None; self.last=0; self.prev=None; self.sp_th=18; self.armed=False
    def map(self, raw:int)->int:
        v = 0 if raw<0 else (255 if raw>255 else raw)
        pv=self.prev
        if pv is None:
            self.prev=float(v); self.ema=float(v)
        else:
            dv=abs(v-pv)
            if dv>self.sp_th:
                if not self.armed: self.armed=True; v=int(pv)
                else: self.armed=False
            else: self.armed=False
            self.prev=float(v)
            a=self.a_f if abs(self.ema-v)>self.jump else self.a_s
            self.ema=a*v+(1-a)*self.ema
        rawf=(255.0-self.ema) if self.invert else self.ema
        d=rawf-self.center; ad = d if d>=0 else -d
        last=self.last; in_zero=(last==0); leave=self.dz+(self.hy if in_zero else 0)
        if (in_zero and ad<=leave) or (not in_zero and ad<=self.dz): self.last=0; return 0
        if (last>0 and d<0 and ad<self.sg) or (last<0 and d>0 and ad<self.sg): d=ad if last>0 else -ad
        pos=d>=0; mag=ad-self.dz; mag=0.0 if mag<0 else mag
        sp=(255.0-self.center-self.dz); sp=1.0 if sp<=1.0 else sp
        sn=(self.center-self.dz);        sn=1.0 if sn<=1.0 else sn
        full=32767 if pos else 32768
        out=int(mag*full/(sp if pos else sn)); out=(32767 if out>32767 else out) if pos else -(32768 if out>32768 else out)
        self.last=out; return out

# ---------------- Bridge ----------------

def is_target(d:BLEDevice,a:AdvertisementData,prefix:str)->bool:
    return (d.name or "").startswith(prefix)

class Bridge:
    """Map one BLE packet to ViGEm inputs."""
    __slots__=("pad","idx","f","lsb","msb","btnL","btnM","trL","trR")
    def __init__(self, conf:Dict[str,Any]):
        self.pad = VX360Gamepad()
        ax,cen,fl = conf["axes"], conf["centers"], conf["filters"]
        invy = conf["invert_y"]
        self.idx = (ax["lx"],ax["ly"],ax["rx"],ax["ry"])
        self.f = (
            AxisFilter(False,fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"],cen["lx"]),
            AxisFilter(invy, fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"],cen["ly"]),
            AxisFilter(False,fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"],cen["rx"]),
            AxisFilter(invy, fl["a_s"],fl["a_f"],fl["jump"],fl["dz"],fl["hy"],fl["sg"],cen["ry"]),
        )
        self.lsb, self.msb = conf["lsb"], conf.get("msb",-1)
        M = XUSB_BUTTON
        mapping = (("A",M.XUSB_GAMEPAD_A),("B",M.XUSB_GAMEPAD_B),("1",M.XUSB_GAMEPAD_X),
                   ("2",M.XUSB_GAMEPAD_Y),("LB",M.XUSB_GAMEPAD_LEFT_SHOULDER),
                   ("RB",M.XUSB_GAMEPAD_RIGHT_SHOULDER),("TAKEOFF",M.XUSB_GAMEPAD_START))
        l,m=[],[]
        for name,tgt in mapping:
            bit,msb = conf["buttons"].get(name,(-1,0))
            if bit>=0: (m if msb else l).append((1<<bit,tgt))
        self.btnL, self.btnM = tuple(l), tuple(m)

        def prep(side:str):
            t=conf["triggers"]; mode=t.get(f"{side}_mode","bit")
            if mode=="byte": return ("byte", int(t.get(f"{side}_byte",-1)), int(t.get(f"{side}_invert",0)))
            else:            return ("bit", int(t.get(f"{side}_bit",-1)), self.msb if int(t.get(f"{side}_msb",0)) else self.lsb)
        self.trL, self.trR = prep("lt"), prep("rt")

    def feed(self, pkt:bytes):
        f0,f1,f2,f3=self.f; i0,i1,i2,i3=self.idx; pad=self.pad
        pad.left_joystick( x_value=f0.map(pkt[i0]), y_value=f1.map(pkt[i1]) )
        pad.right_joystick(x_value=f2.map(pkt[i2]), y_value=f3.map(pkt[i3]) )

        lsb = pkt[self.lsb] if 0<=self.lsb<len(pkt) else 0
        msb = pkt[self.msb] if 0<=self.msb<len(pkt) else 0
        for mask,tgt in self.btnL: (pad.press_button if (lsb & mask) else pad.release_button)(tgt)
        for mask,tgt in self.btnM: (pad.press_button if (msb & mask) else pad.release_button)(tgt)

        kind,a,b = self.trL
        if kind=="byte": val = pkt[a] if 0<=a<len(pkt) else 0; pad.left_trigger(value=(255-val) if b else val)
        else:            val = ((pkt[b]>>a)&1) if (a>=0 and 0<=b<len(pkt)) else 0; pad.left_trigger(value=255 if val else 0)
        kind,a,b = self.trR
        if kind=="byte": val = pkt[a] if 0<=a<len(pkt) else 0; pad.right_trigger(value=(255-val) if b else val)
        else:            val = ((pkt[b]>>a)&1) if (a>=0 and 0<=b<len(pkt)) else 0; pad.right_trigger(value=255 if val else 0)

        pad.update()

# ---------------- BLE main ----------------

async def run(conf:Dict[str,Any]):
    prefix, uuid = conf["prefix"], conf["uuid"]
    evt = asyncio.Event(); dev:Optional[BLEDevice]=None
    def on_adv(d:BLEDevice,a:AdvertisementData):
        nonlocal dev
        if is_target(d,a,prefix): dev=d; evt.set()

    print(f"[i] Scanning for '{prefix}' … Ctrl+C to stop.")
    sc = BleakScanner(on_adv); await sc.start()
    try:
        while True:
            dev=None; evt.clear()
            try: await asyncio.wait_for(evt.wait(), timeout=10.0)
            except asyncio.TimeoutError: continue
            if not dev: continue

            bridge = Bridge(conf)
            try:
                async with BleakClient(dev, timeout=10.0) as cli:
                    # Try notify directly; if it fails once, refresh services then retry once.
                    first_try = True
                    while True:
                        try:
                            if not first_try:
                                getter=getattr(cli,"get_services",None)
                                if callable(getter): await getter()
                            latest = bytearray()
                            await cli.start_notify(uuid, lambda _,d: latest.__init__(bytearray(d)))
                            break
                        except Exception as e:
                            if first_try:
                                first_try = False
                                continue
                            raise e

                    t0=time.time()
                    while len(latest)==0 and time.time()-t0<3.0:
                        await asyncio.sleep(0.05)
                    if len(latest) < EXPECTED_MIN_LEN:
                        print("[!] Short/empty packet; verify characteristic UUID."); continue

                    # Simple, steady feed loop
                    while getattr(cli,"is_connected",False):
                        if latest:
                            bridge.feed(bytes(latest))
                        await asyncio.sleep(0.01)

            except Exception as e:
                print(f"[!] Connect error: {e} — retrying…")
                await asyncio.sleep(1.5)
    finally:
        await sc.stop()

if __name__=="__main__":
    try:
        conf=read_conf()
        asyncio.run(run(conf))
    except KeyboardInterrupt:
        print("\nBye")
