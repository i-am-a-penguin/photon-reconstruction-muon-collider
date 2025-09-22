import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import sys 

nobib = "../ntuple_photonGun_noBIB_MAIAv0_v2.root" # MODIFY
file_nobib = uproot.open(nobib)
tree_nobib = file_nobib["Events"]

bib = "../ntuple_photonGun_BIB_MAIAv0_v2.root" # MODIFY
file_bib = uproot.open(bib)
tree_bib = file_bib["Events"]

ecal_hit_energy_nobib = tree_nobib["ecal_hit_energy"].array()
ecal_hit_time_nobib = tree_nobib["ecal_hit_time"].array()
mcp_energy_nobib = tree_nobib["mcp_energy"].array()

ecal_hit_energy_bib = tree_bib["ecal_hit_energy"].array()
ecal_hit_time_bib = tree_bib["ecal_hit_time"].array()
mcp_energy_bib = tree_bib["mcp_energy"].array()

idx = sys.argv[1]
energies_nobib = ecal_hit_energy_nobib[idx]
hit_times_nobib = ecal_hit_time_nobib[idx]
true_energy_nobib = mcp_energy_nobib[idx][0]

energies_bib = ecal_hit_energy_bib[idx]
hit_times_bib = ecal_hit_time_bib[idx]
true_energy_bib = mcp_energy_bib[idx][0]

timing_cuts = np.linspace(0, 0.04, 100)
fractions_nobib = []
fractions_bib = []
fractions_diff = []

for cut in timing_cuts:
    mask_nobib = hit_times_nobib < cut
    retained_energy_nobib = energies_nobib[mask_nobib]
    fraction_nobib = ak.sum(retained_energy_nobib) / true_energy_nobib
    fractions_nobib.append(fraction_nobib)

    mask_bib = hit_times_bib < cut
    retained_energy_bib = energies_bib[mask_bib]
    fraction_bib = ak.sum(retained_energy_bib) / true_energy_bib
    fractions_bib.append(fraction_bib)

    fraction_diff = fraction_bib - fraction_nobib
    fractions_diff.append(fraction_diff)

plt.figure(figsize=(10, 6))
plt.plot(timing_cuts, fractions_nobib, label="No BIB")
plt.plot(timing_cuts, fractions_bib, label="With BIB")
plt.plot(timing_cuts, fractions_diff, label="Difference")
plt.xlabel("Timing Cut [ns]")
plt.ylabel("Fraction of Total Energy Retained")
plt.title(f"Energy Retention vs Timing Cut (True Energy: {true_energy_bib:.2f} GeV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()