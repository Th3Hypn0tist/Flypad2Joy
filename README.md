My Parrot Mambo drone died by crash, so I had extra controller with no use.  

It didn't have ready driver, so I decided to make one.  

Have fun!  

---

# Flypad2Joy (BLE → ViGEm, Windows)

Ultra-light bridge that maps BLE notification packets to a **virtual Xbox 360 controller** via **ViGEm**.  
Includes **idle-aware anti-spike filtering** so the first nudge after a pause won’t slam to extremes — perfect for FPV sims.

> Daily driver script: **`flypad2joy_anti_spike_nobatt.py`**  
> Baseline (for rollback): **`flypad2joy_baseline_ok.py`**

---

## Table of contents

- [Quick start](#quick-start)
- [Example `flypad.conf`](#example-flypadconf)
- [Configuration reference](#configuration-reference)
  - [`[general]`](#general)
  - [`[axes]`](#axes)
  - [`[axis_centers]`](#axis_centers)
  - [`[buttons]`](#buttons)
  - [`[triggers]`](#triggers)
  - [`[filters]` — motion feel & anti-spike](#filters--motion-feel--anti-spike)
- [Deadzone tuning guide](#deadzone-tuning-guide)
- [Keep-alive (optional)](#keep-alive-optional)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Quick start

1. **Install prerequisites**
   - Windows 10/11
   - Python 3.10+ (3.11 recommended)
   - ViGEmBus driver (required by `vgamepad`)
   - Install Python deps:
     ```bash
     pip install bleak vgamepad
     ```

2. **Run once** to generate a config template:
   ```bash
   py -3.11 flypad2joy_anti_spike_nobatt.py
   ```
   The script writes **`flypad.conf`** next to it and exits.

3. **Edit `flypad.conf`**, set your axis byte indices, and run again:
   ```bash
   py -3.11 flypad2joy_anti_spike_nobatt.py
   ```

---

## Example `flypad.conf`

```ini
[general]
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
; anti-spike (soft start after idle)
spike_idle_ms = 700
spike_samples = 3
spike_raw_step = 6
spike_out_step = 8000
alpha_idle = 0.20
```

---

## Configuration reference

### `[general]`
- **`target_name_prefix`** — BLE name prefix for scanning (devices whose *Bluetooth name* starts with this are accepted).
- **`ctrl_char_uuid`** — GATT characteristic UUID that sends control packets (notifications).
- **`invert_y`** (`true/false`) — Invert LY & RY axes.
- **`button_lsb_index` / `button_msb_index`** — Byte indices in the incoming packet that hold button bits.  
  *Indices are 0-based, e.g. `0` means `pkt[0]`.*

### `[axes]`
- **`lx, ly, rx, ry`** — 0-based byte indices in the packet that hold each 0..255 axis value.

### `[axis_centers]`
- Per-axis neutral center in **0..255** (default `128`).  
  Use this to trim a constant offset (stick not perfectly centered) **before** increasing `deadzone`.

### `[buttons]`
- Format: `NAME = bit,msb`
  - `bit`: 0..7  
  - `msb`: `0` → use `button_lsb_index`; `1` → use `button_msb_index`
- Pre-wired names: `A, B, 1, 2, LB, RB, TAKEOFF`  
  (Triggers are configured in `[triggers]`.)

### `[triggers]`
Two modes:

**Bit mode (digital 0/255)**
```ini
lt_mode = bit
lt_bit  = 0          ; bit position 0..7
lt_msb  = 1          ; 0 = LSB byte, 1 = MSB byte
lt_byte = -1         ; ignored in bit mode
lt_invert = 0        ; ignored in bit mode
```

**Byte mode (analog 0..255 from a packet byte)**
```ini
lt_mode = byte
lt_byte = 7          ; packet byte index
lt_invert = 0        ; 1 = use 255 - value
```

Mirror for right trigger using `rt_*` fields.

### `[filters]` — motion feel & anti-spike

Core smoothing:
- **`alpha_slow`** *(0..1, default 0.35)* — EMA when movement is small → damps micro-jitter.  
- **`alpha_fast`** *(0..1, default 0.85)* — EMA when change > `jump_threshold`. Higher = snappier.  
- **`jump_threshold`** *(raw units, default 10)* — Switch threshold between slow/fast EMAs.  
- **`deadzone`** *(raw units, default 3)* — Center deadzone in **raw** 0..255 units.  
- **`hysteresis`** *(raw units, default 1)* — Extra margin to avoid chatter around zero.  
- **`sign_guard`** *(raw units, default 2)* — Avoids tiny immediate direction flips near center.

Anti-spike (soft start after idle):
- **`spike_idle_ms`** *(ms, default 700)* — If an axis is idle this long, the next samples enter “settle” mode.  
- **`spike_samples`** *(count, default 3)* — Number of post-idle samples to soften.  
- **`spike_raw_step`** *(raw units/sample, default 6)* — Max allowed raw (0..255) step per sample during settle.  
- **`spike_out_step`** *(XInput units/sample, default 8000)* — Max allowed output delta (−32768..32767) per sample during settle.  
- **`alpha_idle`** *(0..1, default 0.20)* — Temporarily slower “fast” alpha during settle.

---

## Deadzone tuning guide

**What’s what**
- `axis_centers` fix a **constant** offset (e.g., stick rests at 131 instead of 128).
- `deadzone` hides **small noise** around center **after** center is set correctly.
- `hysteresis` stops tiny flicker when crossing in/out of the deadzone.

**Workflow**
1. **Observe neutral** → Let the stick rest and inspect raw bytes per axis.
2. **Trim centers** → If an axis sits at, say, 130–131, set its `axis_centers` to the median (130/131).
3. **Measure noise** → Note the wiggle (e.g., 127–129 = ±1).
4. **Set deadzone** → Choose `deadzone` ≈ noise amplitude + 1.  
   Example: noise ±1 → `deadzone = 2` or `3`. Keep it small for precision.
5. **Add hysteresis** → Keep `hysteresis = 1` (or `2` if near-zero flicker remains).
6. **First-nudge feel** → If the first movement after a pause still feels punchy, increase `spike_samples` (4–5) or lower `spike_raw_step` (4).

**Visual intuition**

```
 raw (0..255)
         center
           |
  ---------+---------             deadzone (±dz)
           |<--dz-->|             ┌──── zero output zone ────┐
           |         \            │                         │
left side  |          \           │  output (−32768..32767) │  right side
           |           \__________│__________/              |
           |                      0                         |
           +----------------------------------------------->
```

- Inside ±`deadzone` (+ tiny `hysteresis` when leaving zero) → output **0**.  
- Outside → linear mapping to full XInput range, symmetric left/right.

---

## Keep-alive (optional)

Some devices need a periodic write to stay awake. The script can ping any writable characteristic every ~0.5 s.

At the top of `flypad2joy_anti_spike_nobatt.py`:
```python
HB_ENABLE = True      # set False if your device doesn't need keep-alive
HB_INTERVAL = 0.5
HB_PAYLOAD = b"\x01"
```

---

## Troubleshooting

- **No movement / wrong axes** → verify `[axes]` indices and `ctrl_char_uuid`.  
- **Buttons don’t register** → check `[buttons]` bit & `msb`, and `[general]` `button_lsb_index` / `button_msb_index`.  
- **First nudge after a pause is too strong** → raise `spike_samples` (4–5), lower `spike_raw_step` (4), or lower `alpha_idle` (0.15).  
- **Minor drift** → adjust the specific `axis_centers` first; only then bump `deadzone` by 1–2 if needed.  
- **Bluetooth power saving** (Windows) → In Device Manager, disable “Allow the computer to turn off this device to save power” for your BT adapter if you see random drops.

---

## License

GPL3
