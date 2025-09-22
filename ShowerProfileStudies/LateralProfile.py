import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import sys 

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
    "figure.titlesize": 20
})

file_path = "../ntuple_photonGun_noBIB_MAIAv4.root" # MODIFY
file = uproot.open(file_path)
tree = file["Events"]
branches = tree.arrays()

ecal_hit_depth = tree["ecal_hit_depth"].array()
ecal_hit_x = tree["ecal_hit_x"].array()
ecal_hit_y = tree["ecal_hit_y"].array()
ecal_hit_z = tree["ecal_hit_z"].array()
ecal_hit_energy = tree["ecal_hit_energy"].array()
mcp_vx = tree["mcp_vx"].array()
mcp_vy = tree["mcp_vy"].array()
mcp_vz = tree["mcp_vz"].array()
mcp_phi = tree["mcp_phi"].array()
mcp_theta = tree["mcp_theta"].array()
mcp_energy = tree["mcp_energy"].array()

energy = 200
energy_mask = (ak.firsts(mcp_energy) >= 195) & (ak.firsts(mcp_energy) <= 205)
is_perp = np.abs(ak.firsts(mcp_theta) - (np.pi/2)) < 0.2

def get_radial_distance(phi, theta, x, y, z):
    results = []
    for i in range(len(phi)):
        xi, yi, zi = x[i], y[i], z[i]
        t = theta[i]
        p = phi[i]
        vi = np.array([
            np.sin(t) * np.cos(p),
            np.sin(t) * np.sin(p),
            np.cos(t)
        ])

        distances = []
        for j in range(len(xi)):
            P = np.array([xi[j], yi[j], zi[j]])
            r = np.linalg.norm(np.cross(P, vi)) / np.linalg.norm(vi)
            distances.append(r)
        results.append(distances)
    return ak.Array(results)

def get_profile(radial_distance, energies, bin_width=10, radial_min=0, radial_max=150):
    bins = np.arange(radial_min, radial_max + bin_width, bin_width)
    values, _ = np.histogram(ak.to_numpy(radial_distance), bins=bins, weights=ak.to_numpy(energies), density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, values

mask = energy_mask & is_perp
true_phi = ak.firsts(mcp_phi[mask])
true_theta = ak.firsts(mcp_theta[mask])
hit_energy = ecal_hit_energy[mask]
hit_depth = ecal_hit_depth[mask]
hit_x = ecal_hit_x[mask]
hit_y = ecal_hit_y[mask]
hit_z = ecal_hit_z[mask]

depth_ranges = {
    "20 mm": (hit_depth > 1870) & (hit_depth < 1880),
    "60 mm": (hit_depth > 1910) & (hit_depth < 1920),
    "150 mm": (hit_depth > 2000) & (hit_depth < 2010),
}
colors = ["red", "black", "blue"]

plt.figure(figsize=(10,6))
for label, color in zip(depth_ranges, colors):
    new_mask = depth_ranges[label]
    selected_hit_energy = hit_energy[new_mask]
    selected_hit_x = hit_x[new_mask]
    selected_hit_y = hit_y[new_mask]
    selected_hit_z = hit_z[new_mask]

    #print("Num selected events:", len(selected_true_phi))
    #print("Total hits after cut:", ak.sum(ak.num(selected_hit_energy)))

    radial_distance = get_radial_distance(true_phi, true_theta, selected_hit_x, selected_hit_y, selected_hit_z)

    #print("Radial distance min/max:", ak.min(ak.flatten(radial_distance)), ak.max(ak.flatten(radial_distance)))
    #print("Energy min/max:", ak.min(ak.flatten(selected_hit_energy)), ak.max(ak.flatten(selected_hit_energy)))
    #print("Total energy deposited:", ak.sum(ak.flatten(selected_hit_energy)))

    x, y = get_profile(ak.flatten(radial_distance), ak.flatten(selected_hit_energy))
    plt.plot(x, y, label=label, color=color)

plt.xlabel("Distance from shower axis [mm]")
plt.ylabel("Energy deposit [arbitrary unit]")
#plt.yscale('log')
plt.title(f"True energy: {energy} GeV")
plt.legend(title="Depth", loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/lateral profile {energy}GeV new.png")
plt.show()