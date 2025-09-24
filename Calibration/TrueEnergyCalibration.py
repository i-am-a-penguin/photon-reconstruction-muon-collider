import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

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

# MODIFY
region = "endcaps" # central barrel, transition, endcaps
primary_theta = ak.firsts(mcp_theta_nobib)
#theta_mask = (primary_theta > 0.99) & (primary_theta < 2.15)
#theta_mask = ((primary_theta > 0.7) & (primary_theta < 0.99)) | ((primary_theta > 2.15) & (primary_theta < 2.44))
theta_mask = ((primary_theta < 0.7) & (primary_theta > 0.2)) | ((primary_theta > 2.44) & (primary_theta < 2.94))

ecal_hit_energy_nobib = ecal_hit_energy_nobib[theta_mask]
ecal_hit_time_nobib = ecal_hit_time_nobib[theta_mask]
ecal_hit_theta_nobib = ecal_hit_theta_nobib[theta_mask]
ecal_hit_phi_nobib = ecal_hit_phi_nobib[theta_mask]
ecal_hit_depth_nobib = ecal_hit_depth_nobib[theta_mask]
ecal_hit_z_nobib = ecal_hit_z_nobib[theta_mask]
mcp_phi_nobib = mcp_phi_nobib[theta_mask]
mcp_theta_nobib = mcp_theta_nobib[theta_mask]
mcp_energy = mcp_energy[theta_mask]

# defining cone shape
true_phi = ak.firsts(mcp_phi_nobib)
true_phi_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_phi), ecal_hit_phi_nobib) ])
dphi = ecal_hit_phi_nobib - true_phi_broadcasted

true_theta = ak.firsts(mcp_theta_nobib)
true_theta_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_theta), ecal_hit_theta_nobib) ])
dtheta = ecal_hit_theta_nobib - true_theta_broadcasted

dR = np.sqrt(dphi ** 2 + dtheta **2)
cone_mask = dR < 0.2

# defining timing cut
time_mask_200 = (ecal_hit_time_nobib < 0.2) & (ecal_hit_time_nobib > -0.2)

true_energy = np.asarray(ak.firsts(mcp_energy), dtype=float)
measured_energy = np.asarray(ak.sum(ecal_hit_energy_nobib[cone_mask & time_mask_200], axis=1), dtype=float)

# create figure with 2 vertically stacked axes (shared x-axis)
fig, (ax1, ax2) = plt.subplots(
    2, 1,              # 2 rows, 1 column
    figsize=(10, 7),   # overall figure size
    gridspec_kw={'height_ratios': [3, 1]},  # top taller than bottom
    sharex=True        # share x-axis
)

# mask: points below the line
#m = 0.215
#below = measured_energy < m * true_energy
#above = ~below

ax1.scatter(true_energy, measured_energy, s=10, alpha=0.6, label=f"{yn} data (cone only + 200 ps cut)")
#ax1.scatter(true_energy[below], measured_energy[below], s=10, color="green", alpha=0.6, label=f"{yn} data, below the boundary line (cone only + 200 ps cut)")

# linear fit to the data
a, b = np.polyfit(true_energy, measured_energy, 1)  # degree 1 = linear
x_fit = np.linspace(true_energy.min(), true_energy.max(), 200)
y_fit = a * x_fit + b
ax1.plot(x_fit, y_fit, color="red", lw=1, label=f"Fit: y = {a:.2f}x + {b:.2f}")

# boundary line
#line = m * x_fit
#ax1.plot(x_fit, line, color="black", lw=1, label=f"Boundary: y = {m}x")

# reference line y = x
max_val = max(np.max(true_energy), np.max(measured_energy))
ax1.plot([0, max_val], [0, max_val], 'r--', label="y = x")

ax1.set_ylabel("Reconstructed Energy [GeV]")
ax1.legend()

# residual
fit_at_data = a * true_energy + b
rel_rate = 100.0 * (measured_energy - fit_at_data) / fit_at_data

ax2.axhline(0, color="black", lw=1)
ax2.plot(true_energy, rel_rate, ".", alpha=0.5)
ax2.set_ylim(-50, 200)
ax2.set_ylabel("Relative Rate (w.r.t. fit) [%]")
ax2.set_xlabel("True Energy [GeV]")

# MODIFY
fig.suptitle(r"True vs Reconstructed Energy (Endcap region: $0.2 < \theta < 0.7, 2.44 < \theta < 2.94$)", y=0.98)
# Central barrel region: $0.99 < \theta < 2.15$, Transition region: $0.7 < \theta < 0.99, 2.15 < \theta < 2.44$, Endcap region: $0.12 < \theta < 0.7, 2.44 < \theta < 3.02$

plt.tight_layout()
plt.show()