# Photon Conversion

This directory contains analysis scripts to identify and study photon conversion events (where a photon converts into an electron-positron pair) in the MAIA simulation data.

### Using `ConversionVertex.py`
This script determines whether an event underwent photon conversion and plot their production vertices in a (z, r) plane. It also calculates and prints the number and fraction of photon conversion events.

Run with:
```
python3 ConversionVertex.py
```

### Using `ConversionFractionVSTheta.py` and `ConversionFractionVSEnergy.py`
The former script makes a plot of photon conversion fraction vs true polar angle. The latter script makes a plot of photon conversion fraction vs true energy.

Run with:
```
python3 ConversionFractionVSTheta.py
python3 ConversionFractionVSEnergy.py
```

### Using `DeltaRHistogram.py`
This script makes a ΔR histogram between the true photon and the leading electron/positron from photon conversions.

Run with:
```
python3 DeltaRHistogram.py
```