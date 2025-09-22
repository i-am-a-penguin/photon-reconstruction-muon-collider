import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

yn = 'noBIB'
file_path = f"../ntuple_photonGun_{yn}_MAIAv5.root" # MODIFY
file = uproot.open(file_path)
tree = file["Events"]

ecal_hit_depth = tree["ecal_hit_depth"].array()
ecal_hit_phi = tree["ecal_hit_phi"].array()
ecal_hit_theta = tree["ecal_hit_theta"].array()
ecal_hit_energy = tree["ecal_hit_energy"].array()
ecal_hit_time = tree["ecal_hit_time"].array()
ecal_hit_z = tree["ecal_hit_z"].array()
mcp_energy = tree["mcp_energy"].array()

z = np.array(ak.flatten(ecal_hit_z))
depth = np.array(ak.flatten(ecal_hit_depth))
hit_times = np.array(ak.flatten(ecal_hit_time))

fig, ax = plt.subplots(figsize=(10, 6))

# Define bin edges
z_edges = np.linspace(-3000, 3000, 250)
r_edges = np.linspace(0, 2500, 250)

# 2D histograms: sum of times and count of hits
sum_time, _, _ = np.histogram2d(z, depth, bins=[z_edges, r_edges], weights=hit_times)
count_hits, _, _ = np.histogram2d(z, depth, bins=[z_edges, r_edges])

# Avoid division by 0
with np.errstate(divide='ignore', invalid='ignore'):
    avg_time = np.where(count_hits == 0, np.nan, sum_time / count_hits)

img = plt.imshow(avg_time.T, origin='lower',
                 extent=[z_edges[0], z_edges[-1], r_edges[0], r_edges[-1]],
                 aspect='auto', cmap='RdBu', vmin=-0.2, vmax=0.2, zorder=2) # lower the vmin and vmax to better resolve timing structure
cbar = plt.colorbar(img, ax=ax, label="Average Hit Time [ns]")

ecal_barrel = plt.Rectangle((-2307, 1857), 4614, 268,
                         facecolor='green', alpha=0.1,
                         linewidth=2, zorder=1)
ecal_left_end = plt.Rectangle((-2575, 310), 268, 1815,
                         facecolor='green', alpha=0.1,
                         linewidth=2, zorder=1)
ecal_right_end = plt.Rectangle((2307, 310), 268, 1815,
                         facecolor='green', alpha=0.1,
                         linewidth=2, zorder=1)
ax.add_patch(ecal_barrel)
ax.add_patch(ecal_left_end)
ax.add_patch(ecal_right_end)
plt.xlabel("z [mm]")
plt.ylabel("r [mm]")
plt.xlim(-3000, 3000)
plt.ylim(0, 2500)
plt.title(f"Hit time distribution for all events ({yn})")
plt.tight_layout()
plt.show()
