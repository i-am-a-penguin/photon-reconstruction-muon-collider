import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

yn = "noBIB"
file_nobib = uproot.open(f"../ntuple_photonGun_{yn}_MAIAv5.root") # MODIFY
tree_nobib = file_nobib["Events"]

ecal_hit_energy_nobib = tree_nobib["ecal_hit_energy"].array()
ecal_hit_time_nobib = tree_nobib["ecal_hit_time"].array()
ecal_hit_theta_nobib = tree_nobib["ecal_hit_theta"].array()
ecal_hit_phi_nobib = tree_nobib["ecal_hit_phi"].array()
ecal_hit_depth_nobib = tree_nobib["ecal_hit_depth"].array()
ecal_hit_z_nobib = tree_nobib["ecal_hit_z"].array()
mcp_phi_nobib = tree_nobib["mcp_phi"].array()
mcp_theta_nobib = tree_nobib["mcp_theta"].array()
mcp_energy = tree_nobib["mcp_energy"].array()

# select certain energies
photon_theta = ak.firsts(mcp_theta_nobib)
energy_mask = (photon_theta > 0.2) & (photon_theta < 2.94)

ecal_hit_energy_nobib = ecal_hit_energy_nobib[energy_mask]
ecal_hit_time_nobib = ecal_hit_time_nobib[energy_mask]
ecal_hit_theta_nobib = ecal_hit_theta_nobib[energy_mask]
ecal_hit_phi_nobib = ecal_hit_phi_nobib[energy_mask]
ecal_hit_depth_nobib = ecal_hit_depth_nobib[energy_mask]
ecal_hit_z_nobib = ecal_hit_z_nobib[energy_mask]
mcp_phi_nobib = mcp_phi_nobib[energy_mask]
mcp_theta_nobib = mcp_theta_nobib[energy_mask]
mcp_energy = mcp_energy[energy_mask]

# defining cone shape
true_phi = ak.firsts(mcp_phi_nobib)
true_phi_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_phi), ecal_hit_phi_nobib) ])
dphi = ecal_hit_phi_nobib - true_phi_broadcasted

true_theta = ak.firsts(mcp_theta_nobib)
true_theta_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_theta), ecal_hit_theta_nobib) ])
dtheta = ecal_hit_theta_nobib - true_theta_broadcasted

half_angle = 0.15
dR = np.sqrt(dphi ** 2 + dtheta **2)
cone_mask = dR < half_angle

# defining timing cut
time_mask_200 = (ecal_hit_time_nobib < 0.2) & (ecal_hit_time_nobib > -0.2)

mask = cone_mask & time_mask_200

energy_true = np.asarray(ak.firsts(mcp_energy), dtype=float)
energy_reco = np.asarray(ak.sum(ecal_hit_energy_nobib[mask], axis=1), dtype=float)
theta_true = np.asarray(ak.firsts(mcp_theta_nobib), dtype=float)
# reconstructed theta per event: energy-weighted mean of hit thetas
num = ak.sum(ecal_hit_energy_nobib[mask] * ecal_hit_theta_nobib[mask], axis=1)
den = ak.sum(ecal_hit_energy_nobib[mask], axis=1)

num_np = np.asarray(num, dtype=float)
den_np = np.asarray(den, dtype=float)

theta_reco = np.divide(
    num_np, den_np,
    out=np.full_like(den_np, np.nan),
    where=den_np > 0
)

ratio = np.divide(energy_true, energy_reco, out=np.full_like(energy_true, np.nan), where=energy_reco > 0)

# choose binning
theta_min, theta_max = 0.2, 2.94
E_min, E_max = 0.0, 1000.0

theta_bins = np.linspace(theta_min, theta_max, 31)
E_bins = np.linspace(E_min, E_max, 31)

stat, xedges, yedges, _ = binned_statistic_2d(
    theta_reco, energy_reco, ratio, statistic='mean', bins=[theta_bins, E_bins]
)

# mask empty bins for nicer plotting
stat = np.ma.array(stat.T, mask=~np.isfinite(stat.T))  # transpose so y is vertical

# save calibration map to ROOT
calib_map = stat.T  # shape: (len(xedges)-1, len(yedges)-1)

# build bin-edge tables (low/high per bin) and flatten
theta_low  = xedges[:-1]
theta_high = xedges[1:]
energy_low  = yedges[:-1]
energy_high = yedges[1:]

# make a grid of bin edges
tlo, elo = np.meshgrid(theta_low,  energy_low,  indexing="ij")   # (nth, nen)
thi, ehi = np.meshgrid(theta_high, energy_high, indexing="ij")   # (nth, nen)

# flatten everything to 1D for a clean table
arr_theta_low  = tlo.ravel()
arr_theta_high = thi.ravel()
arr_energy_low  = elo.ravel()
arr_energy_high = ehi.ravel()
arr_value = np.asarray(calib_map).ravel()
arr_value = np.where(np.isfinite(arr_value), arr_value, np.nan)  # keep NaNs

with uproot.recreate(f"calibration_entries.root") as fout:
    fout["calib_bins"] = {
        "theta_low":  arr_theta_low,
        "theta_high": arr_theta_high,
        "energy_low":  arr_energy_low,
        "energy_high": arr_energy_high,
        "value":     arr_value, # E_true/E_reco for that bin
    }

plt.figure(figsize=(7, 6))
im = plt.pcolormesh(xedges, yedges, stat, shading='auto', vmin=1.0, vmax=2)  # adjust vmin=1.0, vmax=1.7
cbar = plt.colorbar(im)
cbar.set_label(r'$E_{\mathrm{true}}/E_{\mathrm{reco}}$')

plt.xlabel(r'Reconstructed $\theta$ [rad]')
plt.ylabel('Reconstructed Photon Energy [GeV]')
plt.title('Calibration')
plt.tight_layout()
plt.show()