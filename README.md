My Parrot Mambo drone died by crash, so I had extra controller with no use.  

It didn't have ready driver, so I decided to make one.

Have fun!  

---

# Flypad2Joy (BLE → ViGEm, Windows)

Ultra-light bridge that maps BLE notification packets to a **virtual Xbox 360 controller** via **ViGEm**.  
Includes **idle-aware anti-spike filtering** so the first nudge after a pause won’t slam to extremes — perfect for FPV sims.

## What this does
This project turns the **Parrot Flypad controller** into a fully working  
**Xbox 360 gamepad** on Windows.  
That means you can use it in games, emulators, or any software  
that expects an XInput controller.

## How it works
- The Flypad only talks over Bluetooth Low Energy (BLE).  
- Windows has no native driver for it.  
- This bridge listens to Flypad BLE packets → maps them to ViGEm virtual gamepad.  
- The result: Windows thinks you have a normal Xbox 360 pad.

## Requirements
- Windows 10/11  
- Python 3.11+  
- [ViGEmBus driver](https://vigem.org/)  
- Packages:  
  ```bash
  pip install bleak vgamepad

---

## Quick start

1. **Run**
   ```bash
   py -3.11 flypad2joy.py

