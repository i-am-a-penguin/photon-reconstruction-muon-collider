# Reading and Visualizing Data

This folder contains basic tools for reading, inspecting, and visualizing photon gun samples.

### Using `ReadingBranches.py`
This script is a quick inspection tool for looking inside your ROOT file. It prints event-by-event details for any branch you want to peek at.

Usage:
```
python3 ReadingBranches.py <branch> <starting_index> <ending_index> (inclusive)
```
        
Example:
```
python3 ReadingBranches.py mcp_pt 0 1
```

### Using `Histogram.py`

This script is a generic histogram plotter for branches inside your photon gun ROOT ntuples; shows a histogram (and optionally can be modified to save as PNG).

Usage:
```
python3 Histogram.py <branch> <number_of_bins>
```
Example:
```
python3 Histogram.py mcp_pt 100
```

### Using `Verify.py`
This script checks the consistency of the kinematic relation for photons in the ntuples: transverse momentum = energy * sin(polar angle theta)

Usage:
```
python3 Verify.py <starting_index> <ending_index>
```

It prints slices of the calculated transverse momentum and the stored transverse momentum for comparison.

### Using `VisualizingHits.py`
This script is a single-event visualizer for ECAL hits. It takes an event index and produces a 2D cross-sectional view of the ECAL in the (z, r) plane.

Usage:
```
python3 VisualizeEvent.py <event_index>
```