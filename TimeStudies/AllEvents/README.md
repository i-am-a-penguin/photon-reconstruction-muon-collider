# Time Studies (averaging across all events)

### Using `AverageEnergyRetentionTimingCut.py`
This script plots the energy retention vs timing cut curve across different photon energies. You can choose the photon energy ranges by modifying `energy_ranges` in the script.

Run with:
```
python3 AverageEnergyRetentionTimingCut.py
```

### Using `TimeDistribution.py`
This script plots a 2D map of average ECAL hit times across all events in the (z, r) plane.

Run with:
```
python3 TimeDistribution.py
```

### Using `TimeDistributionBIBOnly.py` and `TimeDistributionBIBOnlyRemoved.py`
`TimeDistributionBIBOnly.py` plots the same thing as `TimeDistribution.py` using only hits that appear in the BIB sample but not in the noBIB sample (BIB-only hits). `TimeDistributionBIBOnlyRemoved.py` does the same but using only hits that appear in both the BIB and noBIB samples (shared hits).

Run with:
```
python3 TimeDistributionBIBOnly.py
python3 TimeDistributionBIBOnlyRemoved.py
```

### Using 'HitCountDistribution.py'
This script plots a 2D map of the number of hits in the (z, r) plane.

Run with:
```
python3 HitCountDistribution.py
```
