import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
loading files, needed branches, selecting certain theta ranges
'''
file_nobib = uproot.open("../ntuple_photonGun_noBIB_MAIAv2.root")
tree_nobib = file_nobib["Events"]

file_bib = uproot.open(f"../ntuple_photonGun_BIB_MAIAv2.root")
tree_bib = file_bib["Events"]

# loading nobib branches
ecal_hit_energy_bib = tree_bib["ecal_hit_energy"].array()
ecal_hit_time_bib = tree_bib["ecal_hit_time"].array()
ecal_hit_theta_bib = tree_bib["ecal_hit_theta"].array()
ecal_hit_phi_bib = tree_bib["ecal_hit_phi"].array()
ecal_hit_depth_bib = tree_bib["ecal_hit_depth"].array()
ecal_hit_z_bib = tree_bib["ecal_hit_z"].array()
mcp_phi_bib = tree_bib["mcp_phi"].array()
mcp_theta_bib = tree_bib["mcp_theta"].array()
mcp_energy = tree_bib["mcp_energy"].array()

'''
# loading bib branches
mcp_theta_nobib = tree_nobib["mcp_theta"].array()
ecal_hit_z_nobib = tree_nobib["ecal_hit_z"].array()
ecal_hit_depth_nobib = tree_nobib["ecal_hit_depth"].array()
ecal_hit_theta_nobib = tree_nobib["ecal_hit_theta"].array()
'''

# select certain theta
region = "endcaps" # central barrel, transition, endcaps
primary_theta_bib = ak.firsts(mcp_theta_bib)
#theta_mask_bib = (primary_theta_bib > 0.99) & (primary_theta_bib < 2.15)
#theta_mask_bib = ((primary_theta_bib > 0.7) & (primary_theta_bib < 0.99)) | ((primary_theta_bib > 2.15) & (primary_theta_bib < 2.44))
theta_mask_bib = ((0.2 < primary_theta_bib) & (primary_theta_bib < 0.7)) | ((primary_theta_bib > 2.44) & (primary_theta_bib < 2.94))

'''
primary_theta_nobib = ak.firsts(mcp_theta_nobib)
theta_mask_nobib = (primary_theta_nobib > 0.99) & (primary_theta_nobib < 2.15)
#theta_mask_nobib = ((primary_theta_nobib > 0.7) & (primary_theta_nobib < 0.99)) | ((primary_theta_nobib > 2.15) & (primary_theta_nobib < 2.44))
#theta_mask_nobib = (primary_theta_nobib < 0.7) | (primary_theta_nobib > 2.44)
'''

ecal_hit_energy_bib = ecal_hit_energy_bib[theta_mask_bib]
ecal_hit_time_bib = ecal_hit_time_bib[theta_mask_bib]
ecal_hit_theta_bib = ecal_hit_theta_bib[theta_mask_bib]
ecal_hit_phi_bib = ecal_hit_phi_bib[theta_mask_bib]
ecal_hit_depth_bib = ecal_hit_depth_bib[theta_mask_bib]
ecal_hit_z_bib = ecal_hit_z_bib[theta_mask_bib]
mcp_phi_bib = mcp_phi_bib[theta_mask_bib]
mcp_theta_bib = mcp_theta_bib[theta_mask_bib]
mcp_energy = mcp_energy[theta_mask_bib]

'''
ecal_hit_z_nobib = ecal_hit_z_nobib[theta_mask_nobib]
ecal_hit_depth_nobib = ecal_hit_depth_nobib[theta_mask_nobib]
ecal_hit_theta_nobib = ecal_hit_theta_nobib[theta_mask_nobib]
'''

'''
create BIB only and BIB only removed data

coords_nobib = list(zip(np.array(ak.flatten(ecal_hit_z_nobib)), np.array(ak.flatten(ecal_hit_depth_nobib)), np.array(ak.flatten(ecal_hit_theta_nobib))))
coords_bib = list(zip(np.array(ak.flatten(ecal_hit_z_bib)), np.array(ak.flatten(ecal_hit_depth_bib)), np.array(ak.flatten(ecal_hit_theta_bib))))

coords_bib = np.array(coords_bib, dtype=[('z', 'f4'), ('r', 'f4'), ('theta', 'f4')])
coords_nobib = np.array(list(coords_nobib), dtype=[('z', 'f4'), ('r', 'f4'), ('theta', 'f4')])

# create BIB only data
mask_BIBonly_flat = ~np.isin(coords_bib, coords_nobib)
counts = ak.num(ecal_hit_z_bib)
mask_BIBonly = ak.unflatten(mask_BIBonly_flat, counts) # unflatten the mask
ecal_hit_z_bib_only = ecal_hit_z_bib[mask_BIBonly]
ecal_hit_depth_bib_only = ecal_hit_depth_bib[mask_BIBonly]
'''

"""
# create BIB only removed data
mask_BIBonly_removed = ~mask_BIBonly
ecal_hit_z_bib_only_removed = ecal_hit_z_bib[mask_BIBonly_removed]
ecal_hit_depth_bib_only_removed = ecal_hit_depth_bib[mask_BIBonly_removed]
"""

'''
techniques for BIB mitigation
'''
# cone clustering
true_phi = ak.firsts(mcp_phi_bib)
true_phi_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_phi), ecal_hit_phi_bib) ])
dphi = ecal_hit_phi_bib - true_phi_broadcasted

true_theta = ak.firsts(mcp_theta_bib)
true_theta_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_theta), ecal_hit_theta_bib) ])
dtheta = ecal_hit_theta_bib - true_theta_broadcasted

dR = np.sqrt(dphi ** 2 + dtheta **2)
cone_mask = dR < 0.2

# timing cut
time_mask = (ecal_hit_time_bib < 0.2) & (ecal_hit_time_bib > -0.2)

# spatial cut: rejecting earlier layers of BIB deposits (only at the endcaps)
endcap_inner = 2307
r_bins = np.linspace(310, 2000, 5) # change the binning, only consider the endcaps
print("r_bins:", r_bins)

def thresholds_per_bin(ecal_z, ecal_r, r_bins, keep_frac=0.20):
    """
    For each r-bin:
      left_threshold  = quantile(z<0, 1-keep_frac)  (e.g. 0.30 if keep_frac=0.70)
      right_threshold = quantile(z>0, keep_frac)    (e.g. 0.70)
    Return two length-(len(r_bins)-1) numpy arrays storing the calculated threshold.
    """

    left = []
    right = []
    for i in range(len(r_bins)-1):
        in_bin = (np.abs(ecal_z) > endcap_inner) & (ecal_r >= r_bins[i]) & (ecal_r < r_bins[i+1])
        z_bin = np.asarray(ecal_z[in_bin])  # forces into 1D numpy

        z_left  = z_bin[z_bin < 0]
        z_right = z_bin[z_bin > 0]

        if z_left.size > 0:
            left.append(np.quantile(z_left, 1 - keep_frac))
        else:
            left.append(-np.inf)

        if z_right.size > 0:
            right.append(np.quantile(z_right, keep_frac))
        else:
            right.append(+np.inf)

    return np.array(left), np.array(right)

# learn thresholds from BIB samples
ecal_z_bib = ak.flatten(ecal_hit_z_bib)
ecal_r_bib = ak.flatten(ecal_hit_depth_bib)
left_threshold, right_threshold = thresholds_per_bin(ecal_z_bib, ecal_r_bib, r_bins)
print("left thresholds: ", left_threshold)
print("right thresholds: ", right_threshold)

def apply_threshold(ecal_r_bib, ecal_z_bib, left_th, right_th, r_bins):
    result_mask = ak.full_like(ecal_r_bib, False, dtype=bool)

    for i in range(len(r_bins)-1):
        in_bin = (ecal_r_bib >= r_bins[i]) & (ecal_r_bib < r_bins[i+1])
        left_ok  = (ecal_z_bib < 0) & (ecal_z_bib >= left_th[i])
        right_ok = (ecal_z_bib > 0) & (ecal_z_bib <= right_th[i])

        mask_this_bin = in_bin & (left_ok | right_ok)
        result_mask = result_mask | mask_this_bin

    return result_mask

spatial_mask = apply_threshold(ecal_hit_depth_bib, ecal_hit_z_bib, left_threshold, right_threshold, r_bins)
# spatial_mask = (ecal_hit_depth_nobib < 1630) & (ecal_hit_z_nobib > 0) & (ecal_hit_z_nobib > left_threshold) | (ecal_hit_depth_nobib < 1630) & (ecal_hit_z_nobib < 0) & (ecal_hit_z_nobib > right_threshold) 

'''
for checking/debugging purposes
'''
# Fractions removed by the spatial cut (endcaps only sample)
total_hits  = ak.count(ak.flatten(ecal_hit_z_bib))
rej_hits    = ak.count(ak.flatten(ecal_hit_z_bib[spatial_mask]))
print(f"Spatial cut removes hits: {rej_hits}/{total_hits} = {100*rej_hits/max(total_hits,1):.1f}%")

# Energy removed by the spatial cut (much more telling)
E_in_cone_time = ak.sum(ecal_hit_energy_bib[cone_mask & time_mask], axis=1)
E_after_spa    = ak.sum(ecal_hit_energy_bib[cone_mask & time_mask & ~spatial_mask], axis=1)
kept_frac = ak.to_numpy(E_after_spa / ak.where(E_in_cone_time>0, E_in_cone_time, np.nan))
print(f"Median energy kept after spatial cut: {np.nanmedian(kept_frac):.3f}")

'''
computing and graphing resolution vs true energy
'''
# fit gaussian
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# build reco_energy and true_energy and dE
E_reco_0 = ak.sum(ecal_hit_energy_bib[cone_mask], axis=1)
E_reco_1 = ak.sum(ecal_hit_energy_bib[cone_mask & time_mask], axis=1)
E_reco_2 = ak.sum(ecal_hit_energy_bib[cone_mask & ~spatial_mask], axis=1)
E_reco_3 = ak.sum(ecal_hit_energy_bib[cone_mask & time_mask & ~spatial_mask], axis=1)

print("mean E cone:", float(ak.mean(E_reco_0)))
print("mean E cone+time:", float(ak.mean(E_reco_1)))
print("mean E cone+spatial:", float(ak.mean(E_reco_2)))
print("mean E cone+time+spatial:", float(ak.mean(E_reco_3)))

E_truth = ak.ravel(ak.firsts(mcp_energy))
dE_0 = (E_reco_0 - E_truth) / E_truth
dE_1 = (E_reco_1 - E_truth) / E_truth
dE_2 = (E_reco_2 - E_truth) / E_truth
dE_3 = (E_reco_3 - E_truth) / E_truth

# binning
E_bins = np.linspace(0, 1000, 11) # log spaced bins np.logspace(np.log10(1), np.log10(5000), num=13)
bin_centers = 0.5 * (E_bins[:-1] + E_bins[1:])
bin_widths = E_bins[1:] - E_bins[:-1]
x_errors = bin_widths / 2

# How many events fall in each truth-energy bin?
E_truth_np = np.asarray(ak.to_numpy(E_truth), dtype=float)
counts_per_bin, _ = np.histogram(E_truth_np, bins=E_bins)

print("\nEvents per energy bin:")
for i in range(len(E_bins)-1):
    print(f"[{E_bins[i]:.0f}, {E_bins[i+1]:.0f}) GeV : {counts_per_bin[i]}")

def compute_resolution(dE, E_truth, E_bins, bin_min_entries=20, n_hist_bins=25): # the most basic version of this function
    resolutions = []
    res_errors = []
    for i in range(len(E_bins) - 1):
        # Bin mask
        mask = (E_truth >= E_bins[i]) & (E_truth < E_bins[i+1])
        dE_bin = ak.to_numpy(dE[mask])
        
        if len(dE_bin) > bin_min_entries:
            hist, edges = np.histogram(dE_bin, bins=n_hist_bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            try:
                p0 = [np.max(hist), 0, np.std(dE_bin)]
                popt, pcov = curve_fit(gaussian, centers, hist, p0=p0)
                sigma = abs(popt[2])
                sigma_err = np.sqrt(pcov[2][2])
            except Exception as e:
                print(f"Fit failed in bin {i}: {e}")
                sigma, sigma_err = np.nan, np.nan
        else:
            sigma, sigma_err = np.nan, np.nan
            print(f"Not enough data in bin {i}: {len(dE_bin)}")

        resolutions.append(sigma)
        res_errors.append(sigma_err)
    
    return np.array(resolutions), np.array(res_errors)

# compute resolutions
res_0, err_0 = compute_resolution(dE_0, E_truth, E_bins)
res_1, err_1 = compute_resolution(dE_1, E_truth, E_bins)
res_2, err_2 = compute_resolution(dE_2, E_truth, E_bins)
res_3, err_3 = compute_resolution(dE_3, E_truth, E_bins)

plt.figure(figsize=(10,6))
plt.errorbar(bin_centers, res_0, yerr= err_0, xerr=x_errors, fmt='o', color='black', label=r"Cone only") 
plt.errorbar(bin_centers, res_1, yerr= err_1, xerr=x_errors, fmt='o', color='blue', label=r"Cone + 200 ps timing cut")
plt.errorbar(bin_centers, res_2, yerr= err_2, xerr=x_errors, fmt='o', color='red', label=r"Cone + 20% spatial cut")
plt.errorbar(bin_centers, res_3, yerr= err_3, xerr=x_errors, fmt='o', color='green', label=r"Cone + 200 ps timing cut + 20% spatial cut")
plt.xlabel("Truth photon E [GeV]")
plt.ylabel(r"Photon energy resolution")
plt.title(r"Photon Energy Resolution vs Energy (BIB, Endcap region: $0.2 < \theta < 0.7, 2.44 < \theta < 2.94$)")
#Central barrel region: $0.99 < \theta < 2.15$, Transition region: $0.7 < \theta < 0.99, 2.15 < \theta < 2.44$, Endcap region: $0.2 < \theta < 0.7, 2.44 < \theta < 2.94$
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"resolution plots/resolution vs energy {region} comparison v5")
plt.show()