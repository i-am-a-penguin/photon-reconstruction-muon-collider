# Time Studies (for individual events)

### Using `EventHitTimeDisplay.py`
This script makes a 2D scatter plor of ECAL hits in (z, r) plane and colors each hits by its hit time.

Run with:
```
python3 EventHitTimeDisplay.py
```

### Using `EnergyRetentionTimingCut.py`
This script compares energy retention as a function of a timing cut. It plots 3 curves: retained fraction made with noBIB samples, retained fraction with BIB samples, and their difference.

Run with:
```
python3 EnergyRetentionTimingCut.py <event_index>
```