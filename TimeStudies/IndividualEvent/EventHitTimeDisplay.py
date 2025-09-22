import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import sys 

yn = 'BIB'
file_path = f"../ntuple_photonGun_{yn}_MAIAv0_v2.root" # MODIFY
file = uproot.open(file_path)
tree = file["Events"]
branches = tree.arrays()

ecal_hit_depth = tree["ecal_hit_depth"].array()
ecal_hit_phi = tree["ecal_hit_phi"].array()
ecal_hit_theta = tree["ecal_hit_theta"].array()
ecal_hit_energy = tree["ecal_hit_energy"].array()
ecal_hit_time = tree["ecal_hit_time"].array()
ecal_hit_z = tree["ecal_hit_z"].array()
mcp_phi = tree["mcp_phi"].array()
mcp_theta = tree["mcp_theta"].array()
mcp_energy = tree["mcp_energy"].array()

idx = int(sys.argv[1])
true_energy = mcp_energy[idx][0]
mcp_phi = mcp_phi[idx][0]
mcp_theta = mcp_theta[idx][0]
ecal_hit_depth = ecal_hit_depth[idx]
ecal_hit_z = ecal_hit_z[idx]
ecal_hit_energy = ecal_hit_energy[idx]
ecal_hit_time = ecal_hit_time[idx]

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(ecal_hit_z, ecal_hit_depth, c=ecal_hit_time, cmap='viridis', s=5, vmin=0, vmax=np.max(ecal_hit_time), zorder=2)
cbar = plt.colorbar(sc, ax=ax, label="Hit Time [ns]")

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

# Optional: draw the true track (adjust coordinates)
ax.plot([0, 8000*np.cos(mcp_theta)], [0, 8000*np.sin(mcp_theta)], color='green', linewidth=1, zorder=2)

plt.xlabel("z [mm]")
plt.ylabel("r [mm]")
plt.xlim(-3000, 3000)
plt.ylim(0, 2500)
plt.title(f"Hit time distribution for individual events ({yn}, True Energy: {true_energy:.1f})")
plt.tight_layout()
plt.show()
