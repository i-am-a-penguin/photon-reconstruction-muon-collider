# BIB Energy Density
This directory contains scripts to quantify the impact of BIB on photon reconstruction.

### Using `EnergyContainmentDeltaR.py`
This script studies the retained photon signal fraction versus the cone half-angle and finds the cone half-angle ∆R that contains 90% of the photon's energy (R90).

Run with:
```
python3 EnergyContainmentDeltaR.py
```

### Using `BIBEnergyDensity.py`
This script calculates the median BIB energy density from all events in an annulus that is at a large angle from the photon signal direction. It exports its data into a ROOT file.

Run with:
```
python3 BIBEnergyDensity.py
```