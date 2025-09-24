import uproot
import numpy as np

f = uproot.open("calibration_entries.root")
tree = f["calib_bins"]

theta_low  = tree["theta_low"].array(library="np")
theta_high = tree["theta_high"].array(library="np")
energy_low  = tree["energy_low"].array(library="np")
energy_high = tree["energy_high"].array(library="np")
values     = tree["value"].array(library="np")  # E_true/E_reco

def calib_lookup(theta_reco, energy_reco):
    """Return E_true/E_reco for the bin containing (theta_reco, energy_reco). NaN if out-of-range or empty."""
    in_theta = (theta_reco >= theta_low) & (theta_reco < theta_high)
    in_energy = (energy_reco >= energy_low) & (energy_reco < energy_high)
    idx = np.where(in_theta & in_energy)[0]
    if idx.size == 0:
        return np.nan
    return values[idx[0]]

# example usage
theta_test = 0.7   # rad
energy_test = 40  # GeV
cf = calib_lookup(theta_test, energy_test)
print("Calibration factor (E_true/E_reco):", cf)