#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio, time, os, sys, configparser, platform, select
from typing import Optional, Dict, Any, List, Tuple
from bleak import BleakScanner, BleakClient, BLEDevice, AdvertisementData

DEFAULT_PREFIX="FLYPAD_"
DEFAULT_CHAR="9e35fa01-4344-44d4-a2e2-0c7f6046878b"
CONF_DEFAULT=os.path.join(os.path.dirname(__file__), "flypad.conf")
SAMPLE_DT=0.02

def is_target(d:BLEDevice,a:AdvertisementData,prefix:str)->bool:
    return (d.name or "").startswith(prefix)

# ---------- Config I/O ----------
def write_conf(path:str, conf:Dict[str,Any]):
    cp=configparser.ConfigParser()
    cp["general"]={k:str(v) for k,v in conf["general"].items()}
    cp["axes"]={k:str(v) for k,v in conf["axes"].items()}
    cp["axis_centers"]={k:str(v) for k,v in conf["axis_centers"].items()}
    cp["buttons"]={n:"-1,0" for n in ["a","b","1","2","lb","rb","lt","rt","takeoff"]}
    cp["triggers"]={"lt_mode":"bit","lt_bit":"-1","lt_msb":"0","lt_byte":"-1","lt_invert":"0",
                    "rt_mode":"bit","rt_bit":"-1","rt_msb":"0","rt_byte":"-1","rt_invert":"0"}
    cp["filters"]={"alpha_slow":"0.35","alpha_fast":"0.85","jump_threshold":"10",
                   "deadzone":"3","hysteresis":"1","sign_guard":"2"}
    with open(path,"w",encoding="utf-8") as f: cp.write(f)

def read_conf(path:str)->Optional[Dict[str,Any]]:
    if not os.path.exists(path): return None
    cp=configparser.ConfigParser(); cp.read(path,encoding="utf-8")
    try:
        g={"target_name_prefix":cp.get("general","target_name_prefix",fallback=DEFAULT_PREFIX),
           "ctrl_char_uuid":cp.get("general","ctrl_char_uuid",fallback=DEFAULT_CHAR),
           "invert_y":cp.getboolean("general","invert_y",fallback=True)}
        ax={k:cp.getint("axes",k,fallback=i) for i,k in enumerate(("lx","ly","rx","ry"))}
        centers={k:int(v) for k,v in cp.items("axis_centers")} if cp.has_section("axis_centers") else {}
        return {"general":g,"axes":ax,"axis_centers":centers}
    except: return None

async def load_or_bind(path:str, rebind_prompt_seconds:int=5)->Dict[str,Any]:
    conf=read_conf(path)
    if conf is None:
        print("[i] No config → starting binding."); return await run_binding(path)
    # keep backward compatibility with the arg; simple prompt
    if rebind_prompt_seconds:
        if prompt_yn("Rebind (remap) now?", rebind_prompt_seconds):
            return await run_binding(path, conf["general"]["target_name_prefix"], conf["general"]["ctrl_char_uuid"])
    print(f"[i] Using existing config: {path}")
    return conf

def prompt_yn(msg:str, timeout:int)->bool:
    print(f"{msg} (y/r, {timeout}s)…"); sys.stdout.flush()
    end=time.time()+timeout
    if platform.system().lower().startswith("win"):
        try:
            import msvcrt
            while time.time()<end:
                if msvcrt.kbhit(): return msvcrt.getwch().lower() in ("y","r")
                time.sleep(0.05)
        except: pass
    else:
        import select
        while time.time()<end:
            r,_,_=select.select([sys.stdin],[],[],0.1)
            if r: return sys.stdin.readline().strip().lower() in ("y","r")
    print("(continue)"); return False

# ---------- Sampling & simple metrics ----------
def collect_stats(idx:int, seq:List[List[int]])->Tuple[int,int,int,int]:
    vals=[s[idx] for s in seq]
    vmin=min(vals); vmax=max(vals); center=int(sum(vals)/len(vals))
    changes=sum(1 for i in range(1,len(vals)) if vals[i]!=vals[i-1])
    return vmin, vmax, center, changes

def pick_two_bytes(seq:List[List[int]], exclude:set)->Tuple[int,int,Dict[int,Tuple[int,int,int,int]]]:
    m=len(seq[0]); stats={}
    for j in range(m):
        if j in exclude: continue
        vmin,vmax,center,changes=collect_stats(j,seq)
        stats[j]=(vmin,vmax,center,changes)
    # rank by (range, then changes)
    ranked=sorted(stats.keys(),
                  key=lambda j: ((stats[j][1]-stats[j][0]), stats[j][3]),
                  reverse=True)
    if len(ranked)<2:
        # fallbacks if stream short
        a=ranked[0] if ranked else 0
        b=(a+1) if m>1 else a
        return a,b,stats
    return ranked[0], ranked[1], stats

async def sample_for(seconds:float, getv)->List[List[int]]:
    seq=[]; t=time.time()
    while time.time()-t<seconds:
        seq.append(getv()); await asyncio.sleep(SAMPLE_DT)
    if not seq: seq=[getv()]
    return seq

def bar(val:int)->str:
    pos=int((val/255.0)*20)
    return "[" + "#"*pos + "-"*(20-pos) + "]"

def preview_axes(getv, Lx:int,Ly:int,Rx:int,Ry:int, seconds:float=3.0):
    print("\n== Preview (move sticks) ==")
    t=time.time()
    while time.time()-t<seconds:
        pkt=getv()
        lx,ly,rx,ry=(pkt[Lx],pkt[Ly],pkt[Rx],pkt[Ry])
        line=f"LX {lx:3d} {bar(lx)}  LY {ly:3d} {bar(ly)} | RX {rx:3d} {bar(rx)}  RY {ry:3d} {bar(ry)}"
        print("\r"+line+" "*4, end="", flush=True)
        time.sleep(0.05)
    print("\nLooks right? If not, rebind.")

# ---------- Binding core ----------
async def bind_stick(name:str, getv, exclude:set)->Tuple[int,int,Dict[int,Tuple[int,int,int,int]]]:
    print(f"\n== {name} stick ==")
    print("Rotate full range for ~3 s, then leave centered…")
    seq=await sample_for(3.0, getv)
    # grab another short centered slice to improve center average
    seq += await sample_for(0.8, getv)
    x,y,stats = pick_two_bytes(seq, exclude)
    # As requested: assume the latter byte is Y
    # We keep the order (first picked -> X, second -> Y)
    vx=stats[x]; vy=stats[y]
    print(f"{name}: pick X=byte {x} (rng {vx[0]}..{vx[1]} ctr≈{vx[2]})  "
          f"Y=byte {y} (rng {vy[0]}..{vy[1]} ctr≈{vy[2]})")
    return x,y,stats

# ---------- Main binding ----------
async def run_binding(path:str, prefix:str=DEFAULT_PREFIX, uuid:str=DEFAULT_CHAR)->Dict[str,Any]:
    print(f"Binding → scanning {prefix}")
    evt=asyncio.Event(); dev:Optional[BLEDevice]=None
    def on_adv(d,a):
        nonlocal dev
        if is_target(d,a,prefix): dev=d; evt.set()
    sc=BleakScanner(on_adv); await sc.start()
    try:
        while True:
            dev=None; evt.clear()
            try: await asyncio.wait_for(evt.wait(), timeout=10.0)
            except asyncio.TimeoutError: continue
            if not dev: continue
            print(f"Connecting {dev.name} @ {dev.address}")
            async with BleakClient(dev, timeout=10.0) as cli:
                latest=bytearray()
                await cli.start_notify(uuid, lambda _,d: latest.__init__(bytearray(d)))
                t=time.time()
                while len(latest)==0 and time.time()-t<5.0: await asyncio.sleep(0.05)
                if len(latest)<6: print("Packet too short."); continue
                getv=lambda: list(latest)

                # Left stick
                Lx,Ly,statsL = await bind_stick("Left", getv, exclude=set())
                # Right stick (exclude Left bytes so we don't pick same again)
                Rx,Ry,statsR = await bind_stick("Right", getv, exclude={Lx,Ly})

                centers={
                    "lx": statsL[Lx][2],
                    "ly": statsL[Ly][2],
                    "rx": statsR[Rx][2],
                    "ry": statsR[Ry][2],
                }
                axes={"lx":Lx,"ly":Ly,"rx":Rx,"ry":Ry}
                invert_y=True  # common for many pads; change in conf if needed

                # Preview (visual indicator)
                preview_axes(getv, Lx,Ly,Rx,Ry, seconds=3.0)

                conf={"general":{"target_name_prefix":prefix,"ctrl_char_uuid":uuid,"invert_y":invert_y},
                      "axes":axes,"axis_centers":centers}
                write_conf(path, conf); print(f"\n[OK] Saved: {path}")
                return conf
    finally:
        await sc.stop()

# ---------- CLI ----------
if __name__=="__main__":
    try:
        import argparse
        ap=argparse.ArgumentParser(description="Flypad simple axis binder")
        ap.add_argument("--prefix", default=DEFAULT_PREFIX)
        ap.add_argument("--char", default=DEFAULT_CHAR)
        ap.add_argument("--out", default=CONF_DEFAULT)
        args=ap.parse_args()
        asyncio.run(run_binding(args.out, args.prefix, args.char))
    except KeyboardInterrupt:
        print("\nBye")
