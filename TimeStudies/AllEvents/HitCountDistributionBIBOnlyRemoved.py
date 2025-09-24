import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

nobib_file = uproot.open(f"../ntuple_photonGun_nobib_MAIAv5.root") # MODIFY
nobib_tree = nobib_file["Events"]

bib_file = uproot.open(f"../ntuple_photonGun_bib_MAIAv5.root") # MODIFY
bib_tree = bib_file["Events"]

bib_z = np.array(ak.flatten(bib_tree["ecal_hit_z"].array()))
bib_depth = np.array(ak.flatten(bib_tree["ecal_hit_depth"].array()))
bib_time = np.array(ak.flatten(bib_tree["ecal_hit_time"].array()))
bib_coords = list(zip(bib_z, bib_depth))

nobib_z = np.array(ak.flatten(nobib_tree["ecal_hit_z"].array()))
nobib_depth = np.array(ak.flatten(nobib_tree["ecal_hit_depth"].array()))
nobib_coords = set(zip(nobib_z, nobib_depth))

bib_coords = np.array(bib_coords, dtype=[('z', 'f4'), ('depth', 'f4')])
nobib_coords = np.array(list(nobib_coords), dtype=[('z', 'f4'), ('depth', 'f4')])

mask = np.isin(bib_coords, nobib_coords)
bib_only_z = bib_z[mask]
bib_only_depth = bib_depth[mask]

fig, ax = plt.subplots(figsize=(10, 6))

z_edges = np.linspace(-3000, 3000, 250)
r_edges = np.linspace(0, 2500, 250)
count_hits, _, _ = np.histogram2d(bib_only_z, bib_only_depth, bins=[z_edges, r_edges])
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
                 aspect='auto', cmap='viridis', vmin=0, vmax=2500, zorder=2)
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
plt.title(f"Number of entries in hit time distribution for all events (BIB only removed)")
plt.tight_layout()
fig.subplots_adjust(right=1.04)
plt.show()
