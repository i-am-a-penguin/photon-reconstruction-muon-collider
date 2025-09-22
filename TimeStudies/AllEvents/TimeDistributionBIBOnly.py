import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak 

nobib_file_path = f"../ntuple_photonGun_nobib_MAIAv5.root"
nobib_file = uproot.open(nobib_file_path)
nobib_tree = nobib_file["Events"]

bib_file_path = f"../ntuple_photonGun_bib_MAIAv5.root"
bib_file = uproot.open(bib_file_path)
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
mask = ~np.isin(bib_coords, nobib_coords)
bib_only_z = bib_z[mask]
bib_only_depth = bib_depth[mask]
bib_only_time = bib_time[mask]

fig, ax = plt.subplots(figsize=(10, 6))

z_edges = np.linspace(-3000, 3000, 250)
r_edges = np.linspace(0, 2500, 250)

sum_time, _, _ = np.histogram2d(bib_only_z, bib_only_depth, bins=[z_edges, r_edges], weights= bib_only_time)
count_hits, _, _ = np.histogram2d(bib_only_z, bib_only_depth, bins=[z_edges, r_edges])

with np.errstate(divide='ignore', invalid='ignore'):
    avg_time = np.where(count_hits == 0, np.nan, sum_time / count_hits)

img = plt.imshow(avg_time.T, origin='lower',
                 extent=[z_edges[0], z_edges[-1], r_edges[0], r_edges[-1]],
                 aspect='auto', cmap='RdBu', vmin=-10, vmax=10, zorder=2)
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
plt.title(f"Hit time distribution (BIB only)")
plt.tight_layout()
plt.show()


