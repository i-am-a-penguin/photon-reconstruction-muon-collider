import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import sys 

file_path = "../ntuple_photonGun_noBIB_MAIAv0_v2.root" # MODIFY
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

def get_profile(depths, energies, bin_width=5, depth_min=1850, depth_max=(2150)):
    bins = np.arange(depth_min, depth_max + bin_width, bin_width)
    values, _ = np.histogram(ak.to_numpy(depths), bins=bins, weights=ak.to_numpy(energies))
    cumulative_energy = np.cumsum(values)
    fractional_containment = cumulative_energy / cumulative_energy[-1] # maybe divide by the true energy instead? the curve should go to 1 at small energies but not anymore when energy goes up
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, fractional_containment

is_perp = np.abs(ak.firsts(mcp_theta) - (np.pi/2)) < 0.2
e = ak.firsts(mcp_energy)
energy_ranges = {
    "1 GeV": (e >= 0) & (e < 3),
    "10 GeV": (e >= 5) & (e < 15),
    "100 GeV": (e >= 95) & (e < 105),
    "200 GeV": (e >= 195) & (e < 205),
}
styles = [['s',"purple"], ['o',"blue"], ['^',"green"], ['v',"orange"]]

plt.figure(figsize=(10, 6))
for label, style in zip(energy_ranges, styles):
    cond = energy_ranges[label] & is_perp
    selected_depths = ak.flatten(ecal_hit_depth[cond])
    selected_energies = ak.flatten(ecal_hit_energy[cond])
    true_energies = ak.firsts(mcp_energy[cond])
    depth_in_x0 = (selected_depths - 1857) / 10
    num_events = ak.sum(cond) # counts the number of sublist

    if num_events > 0:
        x, y = get_profile(selected_depths, selected_energies)
        plt.plot(x, y, label=label, marker=style[0], color=style[1])

plt.xlabel("Depth [mm]")
#plt.xlabel('Depth [X₀]')
plt.ylabel("Shower energy fraction contained")
plt.ylim(0.90, 1.01)
plt.title("Longitudinal Shower Containment For Different Photon Energies")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
