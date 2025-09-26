# Calibration
### Using `TrueEnergyCalibration.py`
This script compares reconstructed photon energy to the true photon energy within a chosen polar angle region. It evaluates how much the reconstructed energy deviates from the truth and thus how much calibration would be needed to bring them into agreement for photons in that region. You can modify the angular selection to study different detector regions (central barrel, transition region, or endcaps).

*Note: This script is exploratory and was not used in the final reconstruction algorithm.*

Run with:
```
python3 TrueEnergyCalibration.py
```

### Using `ThetaCalibration.py`
This script compares reconstructed photon energy to the true photon energy within a chosen true energy range. It evaluates how much the reconstructed energy deviates from the truth and thus how much calibration would be needed to bring them into agreement for photons in that region. You can modify the energy selection to study different energy regions.

*Note: This script is exploratory and was not used in the final reconstruction algorithm.*

Run with:
```
python3 ThetaCalibration.py
```

### Using `Calibration.py` and `TestingCalibrationOutput.py`
`Calibration.py` builds a 2D calibration map that tells you how much to scale reconstructed photon energies to match the true photon energies, as a function of both polar angle and reconstructed photon energy. It then saves the map into a ROOT file (`calibration_entries.root`). `TestingCalibrationOutput.py` shows how you can access the data.

Run with:
```
python3 Calibration.py
python3 TestingCalibrationOutput.py
```