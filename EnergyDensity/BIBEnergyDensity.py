import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt

yn = "BIB"
version = "v5"
file = uproot.open(f"../ntuple_photonGun_{yn}_MAIA{version}.root") # MODIFY
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

true_phi = ak.firsts(mcp_phi)
true_phi_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_phi), ecal_hit_phi) ])
dphi = ecal_hit_phi - true_phi_broadcasted

true_theta = ak.firsts(mcp_theta)
true_theta_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_theta), ecal_hit_theta) ])
dtheta = ecal_hit_theta - true_theta_broadcasted

dR = np.sqrt(dphi ** 2 + dtheta **2)

def energy_density_annulus(hit_energy, Rin, Rout):
    # select hits in annulus
    mask = (dR > Rin) & (dR < Rout)
    E_annulus = ak.sum(hit_energy[mask], axis=1)
    
    # area in ΔR metric
    area = np.pi * (Rout**2 - Rin**2)
    return E_annulus / area  # energy density (per unit ΔR^2)

Rin, Rout = 0.4, 0.5
energy_density = ak.to_numpy(energy_density_annulus(ecal_hit_energy, Rin, Rout))
theta = ak.to_numpy(true_theta)

plt.figure(figsize=(7,5))
plt.scatter(theta, energy_density, s=5, alpha=0.3, label="Events")

# bin in theta and compute medians
bins = np.linspace(0, np.pi, 30)
bin_centers = 0.5*(bins[1:] + bins[:-1])
medians = []
for i in range(len(bins)-1):
    mask = (theta >= bins[i]) & (theta < bins[i+1])
    if np.any(mask):
        medians.append(np.median(energy_density[mask]))
    else:
        medians.append(np.nan)
medians = np.array(medians)

with uproot.recreate(f"BIB_energy_density_040-050_{version}.root") as fout:
    fout["medians"] = {
        "bin_centers": bin_centers,
        "median_energy_density": medians,
    }

plt.plot(bin_centers, medians, color="red", lw=2, label="Median")
plt.xlabel(r"True polar angle $\theta$ [rad]")
plt.ylabel(r"BIB energy density [GeV/$\text{rad}^2$]")
plt.title(f"BIB energy density vs polar angle ({Rin}-{Rout} rad, {version})")
plt.legend()
plt.tight_layout()
plt.show()