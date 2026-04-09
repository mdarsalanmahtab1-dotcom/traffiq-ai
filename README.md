# TRAFFIQ – Autonomous Lane Following AI
**Team:** AXON  
**Event:** TRAFFIQ | IET On-Campus JU × JUMPER  

## Overview
CNN + PID hybrid approach for real-time white lane following on a black surface.

## Architecture
- **Vision**: OpenCV histogram-based white lane centroid detection
- **Control**: PID controller for smooth, low-latency steering
- **Safety**: Safe-stop mechanism (halts on vision loss / NaN outputs)
- **Obstacle**: Edge-density detector to slow down near objects

## How to Run
```bash
pip install -r requirements.txt
python model.py
```

## Output Variables
| Variable | Range | Description |
|---|---|---|
| `speed` | [-1.0, 1.0] | -1 = full reverse, 1 = full forward |
| `direction` | [-1.0, 1.0] | -1 = full left, 1 = full right |

## Files
- `model.py` — Main AI controller (run this on Raspberry Pi 4B)
- `requirements.txt` — Python dependencies
