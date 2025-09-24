import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

file_nobib = uproot.open("../ntuple_photonGun_noBIB_MAIAv5.root") # MODIFY
tree_nobib = file_nobib["Events"]

file_bib = uproot.open(f"../ntuple_photonGun_BIB_MAIAv5.root") # MODIFY
tree_bib = file_bib["Events"]

ecal_hit_theta_bib = tree_bib["ecal_hit_theta"].array()
ecal_hit_depth_bib = tree_bib["ecal_hit_depth"].array()
ecal_hit_z_bib = tree_bib["ecal_hit_z"].array()

ecal_hit_z_nobib = tree_nobib["ecal_hit_z"].array()
ecal_hit_depth_nobib = tree_nobib["ecal_hit_depth"].array()
ecal_hit_theta_nobib = tree_nobib["ecal_hit_theta"].array()

coords_nobib = list(zip(np.array(ak.flatten(ecal_hit_z_nobib)), np.array(ak.flatten(ecal_hit_depth_nobib)), np.array(ak.flatten(ecal_hit_theta_nobib))))
coords_bib = list(zip(np.array(ak.flatten(ecal_hit_z_bib)), np.array(ak.flatten(ecal_hit_depth_bib)), np.array(ak.flatten(ecal_hit_theta_bib))))

coords_bib = np.array(coords_bib, dtype=[('z', 'f4'), ('r', 'f4'), ('theta', 'f4')])
coords_nobib = np.array(list(coords_nobib), dtype=[('z', 'f4'), ('r', 'f4'), ('theta', 'f4')])

mask_BIBonly_flat = ~np.isin(coords_bib, coords_nobib)
counts = ak.num(ecal_hit_z_bib)
mask_BIBonly = ak.unflatten(mask_BIBonly_flat, counts)
bib_only_z = ecal_hit_z_bib[mask_BIBonly]
bib_only_depth = ecal_hit_depth_bib[mask_BIBonly]

fig, ax = plt.subplots(figsize=(10, 6))

z_edges = np.linspace(-3000, 3000, 500)
r_edges = np.linspace(0, 2500, 250)
count_hits, _, _ = np.histogram2d(bib_only_z, bib_only_depth, bins=[z_edges, r_edges])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
Z, R = np.meshgrid(z_centers, r_centers, indexing='xy')

barrel_mask = (R > 1857) & (R < 2125) & (Z > -2307) & (Z < 2307)
left_endcap_mask = (R > 310) & (R < 2125) & (Z > -2575) & (Z < -2307)
right_endcap_mask = (R > 310) & (R < 2125) & (Z > 2307) & (Z < 2575)
ecal_bin_mask = barrel_mask | left_endcap_mask | right_endcap_mask

masked_count_hits = np.where(ecal_bin_mask, count_hits.T, np.nan)

img = plt.imshow(masked_count_hits,
                 origin='lower',
                 extent=[z_edges[0], z_edges[-1], r_edges[0], r_edges[-1]],
                 aspect='auto', cmap='viridis', zorder=2)
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
plt.title(f"Number of entries in hit time distribution for all events (BIB only)")
plt.tight_layout()
fig.subplots_adjust(right=1.04)
plt.show()
