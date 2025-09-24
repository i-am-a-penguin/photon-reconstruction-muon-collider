# Photon Reconstruction in the Muon Collider

### Overview
This repository contains analysis code for studying photon reconstruction in the MAIA detector concept for a 10 TeV muon collider.

The repository is organized (by workflow stage) into the following directories:
- `ReadingData/`: scripts for inspecting branches and making histograms
- `ShowerProfileStudies/`: scripts for analyzing electromagnetic shower profile
- `TimeStudies/`: scripts for studying effects of time cuts in improving photon reconstruction
- `ReconstructionAlgorithm/`:
- `Calibration/`:
- `BIBEnergyDensity/`:
- `PhotonConversion/`:

### Setup
```
git clone git@github.com:i-am-a-penguin/photon-reconstruction-muon-collider.git
cd photon-reconstruction-muon-collider
```

### Requirements

These scripts require Python 3.12.2 and the following packages:

- [uproot](https://github.com/scikit-hep/uproot) (reading and making ROOT files)
- [awkward](https://awkward-array.org/) (handling jagged arrays)
- [numpy](https://numpy.org/) (numerical operations)
- [matplotlib](https://matplotlib.org/) (plotting)
- [scipy](https://scipy.org/) (for Gaussian fits in resolution scripts)

To install the dependencies, run:
```
pip install -r requirements.txt
```
 Modify file paths (marked by `# MODIFY`) to the MAIA photon gun sample ROOT files you are working with in the Python scripts.
