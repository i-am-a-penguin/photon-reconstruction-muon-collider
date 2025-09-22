import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import sys 

file_path = "../ntuple_photonGun_noBIB_MAIAv4.root" # MODIFY
file = uproot.open(file_path)
tree = file["Events"]
branches = tree.arrays()

ecal_hit_depth = tree["ecal_hit_depth"].array()
ecal_hit_phi = tree["ecal_hit_phi"].array()
ecal_hit_theta = tree["ecal_hit_theta"].array()
ecal_hit_energy = tree["ecal_hit_energy"].array()
ecal_hit_z = tree["ecal_hit_z"].array()
mcp_theta = tree["mcp_theta"].array()
mcp_phi = tree["mcp_phi"].array()
mcp_energy = tree["mcp_energy"].array()

def get_profile(depths, energies, num_events, bin_width=30, depth_min=1800, depth_max=2150):
    bins = np.arange(depth_min, depth_max + bin_width, bin_width)
    values, _ = np.histogram(ak.to_numpy(depths), bins=bins, weights=ak.to_numpy(energies))
    profile = values / num_events / bin_width # GeV/mm per event
    area = np.sum(profile * bin_width)
    normalized_profile = profile / area # normalize the profile
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, normalized_profile

# is_barrel = (np.abs(ecal_hit_z) < 2307)
# is_endcap = (np.abs(ecal_hit_z) > 2307) & (np.abs(ecal_hit_depth) < 2125)

is_perp = np.abs(ak.firsts(mcp_theta) - (np.pi/2)) < 0.2
e = ak.firsts(mcp_energy)
energy_ranges = {
    "1 GeV": (e >= 0) & (e < 3),
    "10 GeV": (e >= 5) & (e < 15),
    "100 GeV": (e >= 95) & (e < 105),
    "200 GeV": (e >= 195) & (e < 205),
}
colors = ["purple", "blue", "green", "orange"]

plt.figure(figsize=(10, 6))
for label, color in zip(energy_ranges, colors):
    cond = energy_ranges[label] & is_perp
    selected_depths = ak.flatten(ecal_hit_depth[cond])
    selected_energies = ak.flatten(ecal_hit_energy[cond])
    depth_in_x0 = (selected_depths - 1857) / 3.5
    num_events = ak.sum(cond) # counts the number of sublist

    if num_events > 0:
        x, y = get_profile(selected_depths, selected_energies, num_events)
        plt.plot(x, y, label=label, color=color)

plt.xlabel("Depth [mm]")
#plt.xlabel('Depth [X₀]')
plt.ylabel("Energy deposited per mm [arbitrary units]")
#plt.yscale('log') 
plt.title("Normalized Longitudinal Shower Profile vs Photon Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
