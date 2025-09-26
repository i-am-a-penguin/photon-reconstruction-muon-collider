import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file = uproot.open("../ntuple_photonGun_noBIB_MAIAv5.root") # MODIFY
tree = file["Events"]

ecal_hit_energy = tree["ecal_hit_energy"].array()
ecal_hit_time = tree["ecal_hit_time"].array()
ecal_hit_theta = tree["ecal_hit_theta"].array()
ecal_hit_phi = tree["ecal_hit_phi"].array()
mcp_phi = tree["mcp_phi"].array()
mcp_theta = tree["mcp_theta"].array()
mcp_energy = tree["mcp_energy"].array()

# MODIFY energy range
energy = 50
primary_energy = ak.firsts(mcp_energy)
energy_mask = (primary_energy > 45) & (primary_energy < 55)

ecal_hit_energy = ecal_hit_energy[energy_mask]
ecal_hit_time = ecal_hit_time[energy_mask]
ecal_hit_theta = ecal_hit_theta[energy_mask]
ecal_hit_phi = ecal_hit_phi[energy_mask]
mcp_phi = mcp_phi[energy_mask]
mcp_theta = mcp_theta[energy_mask]
mcp_energy = mcp_energy[energy_mask]

# fit gaussian
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# defining cone shape
true_phi = ak.ravel(ak.firsts(mcp_phi))
true_phi_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(true_phi, ecal_hit_phi) ])
dphi = ecal_hit_phi - true_phi_broadcasted

true_theta = ak.ravel(ak.firsts(mcp_theta))
true_theta_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(true_theta, ecal_hit_theta) ])
dtheta = ecal_hit_theta - true_theta_broadcasted

dR = np.sqrt(dphi ** 2 + dtheta **2)
spatial_mask = dR < 0.2

# build reco_energy and true_energy and dE
E_reco = ak.sum(ecal_hit_energy[spatial_mask], axis=1)
E_truth = ak.ravel(ak.firsts(mcp_energy))
dE = (E_reco - E_truth)/E_truth

# binning
E_bins = np.linspace(0, np.pi, 16)
bin_centers = 0.5 * (E_bins[:-1] + E_bins[1:])
bin_widths = E_bins[1:] - E_bins[:-1]
x_errors = bin_widths / 2

resolutions = []
res_errors = []

for i in range(len(E_bins) - 1):
    mask = (true_theta >= E_bins[i]) & (true_theta < E_bins[i+1])
    dE_bin = ak.to_numpy(dE[mask])
    print(f"Bin {i}: {len(dE_bin)} events") 

    if len(dE_bin) > 20:
        hist, edges = np.histogram(dE_bin, bins=50, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        try:
            p0 = [np.max(hist), 0, np.std(dE_bin)]
            print(f"Fitting bin {i}: p0 = {p0}, std = {np.std(dE_bin)}, max(hist) = {np.max(hist)}")

            popt, pcov = curve_fit(gaussian, centers, hist, p0=p0)
            sigma = abs(popt[2])
            sigma_err = np.sqrt(pcov[2][2])

        except Exception as e:
            print(f"Fit failed in bin {i}: {e}")
            sigma, sigma_err = np.nan, np.nan
    else:
        sigma, sigma_err = np.nan, np.nan

    resolutions.append(sigma)
    res_errors.append(sigma_err)

resolutions = np.array(resolutions)
res_errors = np.array(res_errors)

plt.figure(figsize=(10,6))
plt.errorbar(bin_centers, resolutions, yerr=res_errors, xerr=x_errors, fmt='o', color='black')
plt.xlabel("Truth theta [rad]")
plt.ylabel(r"$\sigma_E / E$")
plt.title(f"Photon Energy Resolution vs Theta (True Energy: {energy} GeV)")
plt.grid(True)
plt.tight_layout()
plt.show()