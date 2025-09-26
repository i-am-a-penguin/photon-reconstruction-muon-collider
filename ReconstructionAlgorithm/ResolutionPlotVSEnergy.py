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
mcp_phi = tree["mcp_phi"].array()
mcp_theta = tree["mcp_theta"].array()
mcp_energy = tree["mcp_energy"].array()

# MODIFY theta ranges
region = "central barrel" # central barrel, transition, endcaps
primary_theta = ak.firsts(mcp_theta)
#theta_mask = (primary_theta > 0.99) & (primary_theta < 2.15)
#theta_mask = ((primary_theta > 0.7) & (primary_theta < 0.99)) | ((primary_theta > 2.15) & (primary_theta < 2.44))
theta_mask = (primary_theta < 0.7) | (primary_theta > 2.44)

ecal_hit_energy = ecal_hit_energy[theta_mask]
ecal_hit_time = ecal_hit_time[theta_mask]
ecal_hit_theta = ecal_hit_theta[theta_mask]
ecal_hit_phi = ecal_hit_phi[theta_mask]
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

# build reco_energy and true_energy and dE
E_reco_0 = ak.sum(ecal_hit_energy[cone_mask], axis=1)
E_truth = ak.ravel(ak.firsts(mcp_energy))
dE_0 = (E_reco_0 - E_truth) / E_truth

# variable-binning helper
def make_energy_bins(E_truth_array, scheme="quantile", *, n_bins=12,
                     E_min=None, E_max=None, custom_edges=None):
    """
    Return strictly increasing energy bin edges.

    scheme:
      - "quantile": ~equal-count bins using quantiles of E_truth (good for fit stability)
      - "log": log-spaced bins from E_min to E_max
      - "custom": use custom_edges exactly
    """
    E = np.asarray(ak.to_numpy(E_truth_array), dtype=float)
    E = E[np.isfinite(E)]
    if E.size == 0:
        raise ValueError("No finite energies in E_truth_array.")
    if E_min is None: E_min = np.min(E)
    if E_max is None: E_max = np.max(E)

    if scheme == "custom":
        edges = np.array(custom_edges, dtype=float)
        if not np.all(np.diff(edges) > 0):
            raise ValueError("custom_edges must be strictly increasing.")
        return edges

    if scheme == "log":
        if E_min <= 0:
            E_min = max(E_min, 1e-3)
        return np.logspace(np.log10(E_min), np.log10(E_max), n_bins + 1)

    if scheme == "quantile":
        # try requested n_bins; back off if edges collide
        k = n_bins
        while k >= 2:
            q = np.linspace(0, 1, k + 1)
            edges = np.quantile(E, q, method="linear")
            if np.all(np.diff(edges) > 0):
                return edges
            k -= 1
        # fallback to 2 bins if many identical energies
        return np.quantile(E, [0, 0.5, 1.0], method="linear")

    raise ValueError(f"Unknown scheme: {scheme}")

E_bins = make_energy_bins(E_truth, scheme="quantile", n_bins=12) # MODIFY

bin_centers = 0.5 * (E_bins[:-1] + E_bins[1:])
bin_widths  = E_bins[1:] - E_bins[:-1]
x_errors    = bin_widths / 2

def compute_resolution(dE, E_truth, E_bins, version, bin_min_entries=20):
    num_bins = len(E_bins) - 1
    resolutions = []
    res_errors = []
    for i in range(num_bins):
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
        ax.set_title(f"Fit in bin {i+1}/{num_bins}: "
                        rf"E_truth $\in$ [{E_bins[i]:.0f}, {E_bins[i+1]:.0f}) GeV")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.savefig(f"GaussianCheck/ResolutionPlotVSEnergy/{version} bin_{i+1}_of_{num_bins}.png")
        plt.close(fig)

        resolutions.append(sigma)
        res_errors.append(sigma_err)
    
    return np.array(resolutions), np.array(res_errors)

res_0, err_0 = compute_resolution(dE_0, E_truth, E_bins, "cone only")

plt.figure(figsize=(10,6))
plt.errorbar(bin_centers, res_0, yerr=err_0, xerr=x_errors, fmt='o', color='black') 
plt.xlabel("Truth energy [GeV]")
plt.ylabel(r"Photon energy resolution")
plt.title(rf"Photon Energy Resolution vs Energy ({yn}, #Central barrel region: $0.99 < \theta < 2.15$)")
#Central barrel region: $0.99 < \theta < 2.15$, Transition region: $0.7 < \theta < 0.99, 2.15 < \theta < 2.44$, Endcap region: $\theta < 0.7, \theta > 2.44$
plt.grid(True)
plt.tight_layout()
plt.show()