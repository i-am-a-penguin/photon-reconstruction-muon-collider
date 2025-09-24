import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt

yn = "noBIB"
file = uproot.open(f"../ntuple_photonGun_{yn}_MAIAv5.root") # MODIFY
tree = file["Events"]

ecal_hit_energy = tree["ecal_hit_energy"].array()
ecal_hit_time = tree["ecal_hit_time"].array()
ecal_hit_theta = tree["ecal_hit_theta"].array()
ecal_hit_phi = tree["ecal_hit_phi"].array()
ecal_hit_depth = tree["ecal_hit_depth"].array()
ecal_hit_z = tree["ecal_hit_z"].array()
mcp_phi = tree["mcp_phi"].array()
mcp_theta = tree["mcp_theta"].array()
mcp_energy = tree["mcp_energy"].array()

# defining cone shape
true_phi = ak.firsts(mcp_phi)
true_phi_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_phi), ecal_hit_phi) ])
dphi = ecal_hit_phi - true_phi_broadcasted

true_theta = ak.firsts(mcp_theta)
true_theta_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_theta), ecal_hit_theta) ])
dtheta = ecal_hit_theta - true_theta_broadcasted

dR = np.sqrt(dphi ** 2 + dtheta **2)

Rmax   = 0.5
nsteps = 51
Rgrid  = np.linspace(0.0, Rmax, nsteps)

# total energy per event
Etot = ak.firsts(mcp_energy)
Etot = ak.where(Etot == 0.0, 1e-12, Etot) # avoid division by zero

fractions = []
for R in Rgrid:
    frac_R = ak.sum( ecal_hit_energy * (dR <= R), axis=1 ) / Etot
    fractions.append(ak.to_numpy(frac_R))
fractions = np.vstack(fractions).T   # shape: (nevents, nR)

median_frac = np.nanmedian(fractions, axis=0)
lo_frac     = np.nanpercentile(fractions, 16, axis=0)
hi_frac     = np.nanpercentile(fractions, 84, axis=0)

fractions = np.clip(fractions, 0, 1)

target = 0.9
R90 = []
for ev in range(fractions.shape[0]):
    f_ev = fractions[ev]
    # If the event is empty or never reaches 0.9, return NaN
    if not np.isfinite(f_ev).any() or f_ev.max() < target:
        R90.append(np.nan)
        continue
    # np.interp expects ascending x; fractions vs Rgrid is ascending.
    R90.append(np.interp(target, f_ev, Rgrid))
R90 = np.array(R90)

R90_median = np.nanmedian(R90)
R90_lo     = np.nanpercentile(R90, 16)
R90_hi     = np.nanpercentile(R90, 84)
print(f"R90: median={R90_median:.4f}, 16-84% band = [{R90_lo:.4f}, {R90_hi:.4f}]")

plt.figure(figsize=(6,4))
plt.plot(Rgrid, median_frac, label="Median")
plt.fill_between(Rgrid, lo_frac, hi_frac, alpha=0.25, label="16-84%")
plt.axhline(0.9, linestyle="--", linewidth=1)
plt.axvline(R90_median, linestyle="--", linewidth=1, label=rf"R90 $\approx$ {R90_median:.3f}")
plt.xlabel(r"$\Delta R$")
plt.ylabel("Fraction of signal energy retained")
plt.title(r"Retained signal energy fraction vs $\Delta R$")
plt.legend()
plt.tight_layout()
plt.show()