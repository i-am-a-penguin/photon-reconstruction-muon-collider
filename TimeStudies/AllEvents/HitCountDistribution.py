import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

yn = 'BIB'
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

z_edges = np.linspace(-3000, 3000, 500)
r_edges = np.linspace(0, 2500, 250)

count_hits, _, _ = np.histogram2d(z, depth, bins=[z_edges, r_edges])

z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
Z, R = np.meshgrid(z_centers, r_centers, indexing='xy')  # shape: (ny, nx)

barrel_mask = (R > 1857) & (R < 2125) & (Z > -2307) & (Z < 2307)
left_endcap_mask = (R > 310) & (R < 2125) & (Z > -2575) & (Z < -2307)
right_endcap_mask = (R > 310) & (R < 2125) & (Z > 2307) & (Z < 2575)
ecal_bin_mask = barrel_mask | left_endcap_mask | right_endcap_mask

masked_count_hits = np.where(ecal_bin_mask, count_hits.T, np.nan)

img = plt.imshow(masked_count_hits,
                 origin='lower',
                 extent=[z_edges[0], z_edges[-1], r_edges[0], r_edges[-1]],
                 aspect='auto', cmap='viridis', vmin=0, vmax=300000, zorder=2)
cbar = plt.colorbar(img, ax=ax, label="Number of entries")

ecal_barrel = plt.Rectangle((-2307, 1857), 4614, 268,
                         facecolor='black', alpha=0.4,
                         linewidth=2, zorder=1)
ecal_left_end = plt.Rectangle((-2575, 310), 268, 1815,
                         facecolor='black', alpha=0.4,
                         linewidth=2, zorder=1)
ecal_right_end = plt.Rectangle((2307, 310), 268, 1815,
                         facecolor='black', alpha=0.4,
                         linewidth=2, zorder=1)
ax.add_patch(ecal_barrel)
ax.add_patch(ecal_left_end)
ax.add_patch(ecal_right_end)
plt.xlabel("z [mm]")
plt.ylabel("r [mm]")
plt.xlim(-3000, 3000)
plt.ylim(0, 2500)
plt.title(f"Number of entries in hit time distribution for all events ({yn})")
plt.tight_layout()
fig.subplots_adjust(right=1.04)
plt.show()
