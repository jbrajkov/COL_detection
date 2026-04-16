# Cut-Off Low Detection Algorithm

This repository provides a Python implementation of a cut-off low (COL)
detection algorithm based on geopotential height (z500) fields.

The algorithm:
- Detects closed low-pressure systems using histogram-based thresholds
- Identifies spatially coherent structures (COLs)
- Attributes extreme precipitation events to detected COLs

---

## 📦 Installation

```bash
pip install -r requirements.txt

## Main class
col_detection within the script COL_detection.py

## Main method
col()

## Requirements
See requirements.txt

## Usage
Run the example script run_example.py

## Notes
- Paths must be adapted to the user environment
- Input data: NetCDF (z500 + precipitation)
