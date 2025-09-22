import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

yn = "noBIB"
file_nobib = uproot.open(f"../ntuple_photonGun_{yn}_MAIAv5.root") # MODIFY
tree_nobib = file_nobib["Events"]

ecal_hit_energy_nobib = tree_nobib["ecal_hit_energy"].array()
ecal_hit_time_nobib = tree_nobib["ecal_hit_time"].array()
mcp_energy_nobib = tree_nobib["mcp_energy"].array()

e = ak.firsts(mcp_energy_nobib)

energy_ranges = {
    "5 GeV": (e >= 4) & (e < 6),
    "10 GeV": (e >= 9) & (e < 11),
    "50 GeV": (e >= 48) & (e < 52),
    "100 GeV": (e >= 98) & (e < 102),
    "500 GeV": (e >= 495) & (e < 505),
    "1000 GeV": (e >= 995) & (e < 1000)
    #"0-200 GeV": (e >= 0) & (e < 200),
    #"200-400 GeV": (e >= 200) & (e < 400),
    #"400-600 GeV": (e >= 400) & (e < 600),
    #"600-800 GeV": (e >= 600) & (e < 800),
    #"800-1000 GeV": (e >= 800) & (e < 1000)
}

def apply_time_cuts(energies, times, true_energies, timing_cuts):
    all_fractions = []
    for i in range(len(energies)):
        e_hits = np.array(energies[i])
        t_hits = np.array(times[i])
        e_true = true_energies[i]

        if e_true == 0 or len(e_hits) == 0:
            continue # skip bad events

        fractions = [np.sum(e_hits[t_hits<cut])/e_true for cut in timing_cuts]
        all_fractions.append(fractions)
        if i % 500 == 0:
            print(f"Processed {i} events")
    return np.mean(np.array(all_fractions), axis=0)

plt.figure(figsize=(10, 6))

for label in energy_ranges:
    timing_cuts = np.linspace(-0.02, 0.4, 100) # time can be negative
    mask = energy_ranges[label]

    energies_nobib = ecal_hit_energy_nobib[mask]
    hit_times_nobib = ecal_hit_time_nobib[mask]
    true_energy_nobib = ak.firsts(mcp_energy_nobib[mask])
    assert len(energies_nobib) == len(true_energy_nobib), "noBIB length doesn't match"
    
    print(f"Computing for energy range: {label}")
    fractions_nobib = apply_time_cuts(energies_nobib, hit_times_nobib, true_energy_nobib, timing_cuts)
    plt.plot(timing_cuts, fractions_nobib, label=label)

plt.xlabel("Timing Cut [ns]")
plt.ylabel("Average Fraction of Total Energy Retained")
plt.title(f"Energy Retention vs Timing Cut ({yn})")
plt.grid(True)
plt.legend(title="True Energy Range")
plt.tight_layout()
plt.show()