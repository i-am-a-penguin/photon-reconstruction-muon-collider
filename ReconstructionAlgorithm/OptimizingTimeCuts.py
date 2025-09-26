import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

yn = "BIB"
file = uproot.open(f"../ntuple_photonGun_{yn}_MAIAv5.root") # MODIFY
tree = file["Events"]

ecal_hit_energy = tree["ecal_hit_energy"].array()
ecal_hit_time = tree["ecal_hit_time"].array()
ecal_hit_theta = tree["ecal_hit_theta"].array()
ecal_hit_phi = tree["ecal_hit_phi"].array()
ecal_hit_depth = tree["ecal_hit_depth"].array()
ecal_hit_z = tree["ecal_hit_z"].array()
mcp_phi = tree["mcp_phi"].array()
mcp_theta = tree["mcp_theta"].array()
mcp_energy = tree["mcp_energy"].array()

# MODIFY theta range
region = "central barrel" # central barrel, transition, endcaps
primary_theta = ak.firsts(mcp_theta)
#theta_mask = (primary_theta > 0.99) & (primary_theta < 2.15)
theta_mask = ((primary_theta > 0.7) & (primary_theta < 0.99)) | ((primary_theta > 2.15) & (primary_theta < 2.44))
#theta_mask = (primary_theta < 0.7) | (primary_theta > 2.44) 0.5

ecal_hit_energy = ecal_hit_energy[theta_mask]
ecal_hit_time = ecal_hit_time[theta_mask]
ecal_hit_theta = ecal_hit_theta[theta_mask]
ecal_hit_phi = ecal_hit_phi[theta_mask]
ecal_hit_depth = ecal_hit_depth[theta_mask]
ecal_hit_z = ecal_hit_z[theta_mask]
mcp_phi = mcp_phi[theta_mask]
mcp_theta = mcp_theta[theta_mask]
mcp_energy = mcp_energy[theta_mask]

# fit gaussian
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# defining cone shape
true_phi = ak.firsts(mcp_phi)
true_phi_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_phi), ecal_hit_phi) ])
dphi = ecal_hit_phi - true_phi_broadcasted

true_theta = ak.firsts(mcp_theta)
true_theta_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_theta), ecal_hit_theta) ])
dtheta = ecal_hit_theta - true_theta_broadcasted

dR = np.sqrt(dphi ** 2 + dtheta **2)
cone_mask = dR < 0.2

# defining timing cut (100 ps to a couple of nanoseconds)
time_mask_1 = (ecal_hit_time < 0.001) & (ecal_hit_time > -0.001)
time_mask_5 = (ecal_hit_time < 0.005) & (ecal_hit_time > -0.005)
time_mask_100 = (ecal_hit_time < 0.1) & (ecal_hit_time > -0.1)
time_mask_10 = (ecal_hit_time < 0.01) & (ecal_hit_time > -0.01)
time_mask_50 = (ecal_hit_time < 0.05) & (ecal_hit_time > -0.05)
time_mask_300 = (ecal_hit_time < 0.3) & (ecal_hit_time > -0.3)
time_mask_500 = (ecal_hit_time < 0.5) & (ecal_hit_time > -0.5)
time_mask_700 = (ecal_hit_time < 0.7) & (ecal_hit_time > -0.7)
time_mask_1000 = (ecal_hit_time < 1) & (ecal_hit_time > -1)

# build reco_energy and true_energy and dE
E_reco_0 = ak.sum(ecal_hit_energy[cone_mask], axis=1)
E_reco_1 = ak.sum(ecal_hit_energy[cone_mask & time_mask_100], axis=1)
E_reco_2 = ak.sum(ecal_hit_energy[cone_mask & time_mask_50], axis=1)
E_reco_3 = ak.sum(ecal_hit_energy[cone_mask & time_mask_10], axis=1)
E_reco_4 = ak.sum(ecal_hit_energy[cone_mask & time_mask_1], axis=1)
E_reco_5 = ak.sum(ecal_hit_energy[cone_mask & time_mask_5], axis=1)
E_truth = ak.ravel(ak.firsts(mcp_energy))
dE_0 = (E_reco_0 - E_truth) / E_truth
dE_1 = (E_reco_1 - E_truth) / E_truth
dE_2 = (E_reco_2 - E_truth) / E_truth
dE_3 = (E_reco_3 - E_truth) / E_truth
dE_4 = (E_reco_4 - E_truth) / E_truth
dE_5 = (E_reco_5 - E_truth) / E_truth

# binning
E_bins = np.linspace(0, 1000, 11) # log spaced bins np.logspace(np.log10(1), np.log10(5000), num=13)
bin_centers = 0.5 * (E_bins[:-1] + E_bins[1:])
bin_widths = E_bins[1:] - E_bins[:-1]
x_errors = bin_widths / 2

def compute_resolution(dE, E_truth, E_bins, version, bin_min_entries=20): # saves the gaussian fits
    resolutions = []
    res_errors = []
    for i in range(len(E_bins) - 1):
        # Bin mask
        mask = (E_truth >= E_bins[i]) & (E_truth < E_bins[i+1])
        dE_bin = ak.to_numpy(dE[mask])

        nbins = 30 if len(dE_bin) >= 30 else max(10, len(dE_bin)//2 or 10)

        fig, ax = plt.subplots(figsize=(6,4.2))
        ax.hist(dE_bin, bins=nbins, density=True, histtype='step', linewidth=1.5, label=f"{version} data")
        
        if len(dE_bin) > bin_min_entries:
            hist, edges = np.histogram(dE_bin, bins=nbins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            try:
                p0 = [np.max(hist), 0, np.std(dE_bin)]
                popt, pcov = curve_fit(gaussian, centers, hist, p0=p0)
                A_fit, mu_fit, sigma = popt
                sigma = abs(popt[2])
                sigma_err = np.sqrt(pcov[2][2])

                ax.plot(centers, gaussian(centers, A_fit, mu_fit, sigma), label=f"{version} fit")
            except Exception as e:
                print(f"Fit failed in bin {i}: {e}")
                sigma, sigma_err = np.nan, np.nan
        else:
            sigma, sigma_err = np.nan, np.nan

        ax.legend()
        ax.set_ylabel("density")
        ax.set_xlabel(r"$(E_{reco}-E_{truth})/E_{truth}$")
        ax.set_title(f"Fit in bin {i+1}/{len(E_bins) - 1}: "
                        rf"E_truth $\in$ [{E_bins[i]:.0f}, {E_bins[i+1]:.0f}) GeV")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.savefig(f"GaussianCheck/OptimizingTimeCuts/{version} bin_{i+1}_of_{len(E_bins) - 1}.png")
        plt.close(fig)

        resolutions.append(sigma)
        res_errors.append(sigma_err)
    
    return np.array(resolutions), np.array(res_errors)

# compute resolutions
res_0, err_0 = compute_resolution(dE_0, E_truth, E_bins, "cone only")
res_1, err_1 = compute_resolution(dE_1, E_truth, E_bins, "cone and time (100 ps)")
res_2, err_2 = compute_resolution(dE_2, E_truth, E_bins, "cone and time (50 ps)")
res_3, err_3 = compute_resolution(dE_3, E_truth, E_bins, "cone and time (10 ps)")
res_4, err_4 = compute_resolution(dE_4, E_truth, E_bins, "cone and time (1 ps)")
res_5, err_5 = compute_resolution(dE_5, E_truth, E_bins, "cone and time (5 ps)")

plt.figure(figsize=(10,6))
plt.errorbar(bin_centers, res_0, yerr= err_0, xerr=x_errors, fmt='o', color='black', label=r"Cone only")
plt.errorbar(bin_centers, res_1, yerr= err_1, xerr=x_errors, fmt='o', color='blue', label=r"Cone + 100 ps timing cut") 
plt.errorbar(bin_centers, res_2, yerr= err_2, xerr=x_errors, fmt='o', color='red', label=r"Cone + 50 ps timing cut")
plt.errorbar(bin_centers, res_3, yerr= err_3, xerr=x_errors, fmt='o', color='green', label=r"Cone + 10 ps timing cut")
plt.errorbar(bin_centers, res_4, yerr= err_4, xerr=x_errors, fmt='o', color='purple', label=r"Cone + 1 ps timing cut")
plt.errorbar(bin_centers, res_5, yerr= err_5, xerr=x_errors, fmt='o', color='orange', label=r"Cone + 5 ps timing cut")
plt.xlabel("Truth photon E [GeV]")
plt.ylabel(r"Photon energy resolution")
plt.title(r"Photon Energy Resolution vs Energy (Transition region: $0.7 < \theta < 0.99, 2.15 < \theta < 2.44$)")
# Central barrel region: $0.99 < \theta < 2.15$, Transition region: $0.7 < \theta < 0.99, 2.15 < \theta < 2.44$, Endcap region: $\theta < 0.7, \theta > 2.44$
plt.legend()
plt.grid(True)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(f"time optimization/{region} time comparison {yn} short time cuts 0-1000")
plt.show()