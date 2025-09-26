# Resolution Plots For Photon Reconstruction Algorithm
A reconstruction algorithm's precision is evaluated with resolution plots. The resolution is defined as the standard deviation (σ) of a Gaussian fit to the distribution of the relative difference between reconstructed and true quantities.

First create a directory for storing the Gaussian fits in each resolution plot, using
```
mkdir GaussianCheck
```

### Using `OptimizingTimeCuts.py`
This script makes resolution plots of photon energy reconstruction performance under different timing cuts to check which improves energy resolution the most.

Run with:
```
mkdir -p GaussianCheck/OptimizingTimeCuts
python3 OptimizingTimeCuts.py
```
The Gaussian fits from each point in `OptimizingTimeCuts.py` will be stored into the directory `GaussianCheck/OptimizingTimeCuts/`.

### Using `ResolutionPlotVSTheta.py`
This script plots the photon energy resolution versus true polar angle for a chosen energy range. It applies a simple cone reconstruction algorithm.  
*Note: this script is more basic and does not include helper functions for variable binning or systematic resolution extraction.*

Run with:
```
python3 ResolutionPlotVSTheta.py
```

### Using `ResolutionPlotVSEnergy.py`
This script plots the photon energy resolution versus true energy for a chosen polar angle region. Like `ResolutionPlotVSTheta.py`, it uses a simple cone reconstruction algorithm, but it also implements functions for variable binning and standardized systematic resolution calculation and plotting.

Run with:
```
mkdir -p GaussianCheck/ResolutionPlotVSEnergy
python3 ResolutionPlotVSEnergy.py
```
The Gaussian fits from each point in `ResolutionPlotVSEnergy.py` will be stored into the directory `GaussianCheck/ResolutionPlotVSEnergy/`.

### Using `OptimizingConeSize.py`
This script tries to find the optimal cone size by plotting resolution plots applied with calibration data from `../Calibration/calibration_entries.root`.

Run with:
```
python3 OptimizingConeSize.py
```

### Using `ReconstructionAlgorithm.py`
This script accounts for timing cut, spatial cut, cone clustering, calibration, and BIB energy density.

Run with:
```
python3 ReconstructionAlgorithm.py
```