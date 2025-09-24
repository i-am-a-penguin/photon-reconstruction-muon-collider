import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

# Known geometry
Z0     = 150.0   # distance along z from IP to solenoid start (cm)
R0     = 150.0   # inner radius at solenoid (cm)
Rf     = 185.7   # outer radius at solenoid (cm)
X0     = 8.89   # effective radiation length of solenoid stack (cm)
theta1 = math.atan(150.0/230.7)   # boundary angle between "no material" and solenoid (rad)
theta2 = math.atan(185.7/230.7)   # boundary angle between solenoid and central barrel (rad)

yn = "noBIB"
file_nobib = uproot.open(f"../ntuple_photonGun_{yn}_MAIAv5.root") # MODIFY
tree_nobib = file_nobib["Events"]

ecal_hit_energy_nobib = tree_nobib["ecal_hit_energy"].array()
ecal_hit_time_nobib = tree_nobib["ecal_hit_time"].array()
ecal_hit_theta_nobib = tree_nobib["ecal_hit_theta"].array()
ecal_hit_phi_nobib = tree_nobib["ecal_hit_phi"].array()
ecal_hit_depth_nobib = tree_nobib["ecal_hit_depth"].array()
ecal_hit_z_nobib = tree_nobib["ecal_hit_z"].array()
mcp_phi_nobib = tree_nobib["mcp_phi"].array()
mcp_theta_nobib = tree_nobib["mcp_theta"].array()
mcp_energy = tree_nobib["mcp_energy"].array()

# select certain energies
photon_energy = ak.firsts(mcp_energy)
photon_theta = ak.firsts(mcp_theta_nobib)
energy_mask = (photon_energy > 100) & (photon_theta > 0.2) & (photon_theta < 2.94) & (photon_energy < 150)

ecal_hit_energy_nobib = ecal_hit_energy_nobib[energy_mask]
ecal_hit_time_nobib = ecal_hit_time_nobib[energy_mask]
ecal_hit_theta_nobib = ecal_hit_theta_nobib[energy_mask]
ecal_hit_phi_nobib = ecal_hit_phi_nobib[energy_mask]
ecal_hit_depth_nobib = ecal_hit_depth_nobib[energy_mask]
ecal_hit_z_nobib = ecal_hit_z_nobib[energy_mask]
mcp_phi_nobib = mcp_phi_nobib[energy_mask]
mcp_theta_nobib = mcp_theta_nobib[energy_mask]
mcp_energy = mcp_energy[energy_mask]

# N(theta): radiation-lengths before ECAL
def N_of_theta(theta):
    theta = np.asarray(theta, dtype=float)
    theta = np.minimum(theta, np.pi - theta)
    N = np.zeros_like(theta) # make an array of 0 that has the same shape as theta

    # regions
    m1 = (theta >= theta1) & (theta <= theta2)
    m2 = (theta > theta2)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    N[m1] = (Z0/np.abs(cos_t[m1]) - R0/np.abs(sin_t[m1])) / X0
    N[m2] = ((Rf - R0) / np.abs(sin_t[m2])) / X0
    # theta < theta1 -> N = 0 already

    N = np.clip(N, 0, None) # no negative N
    return N

def ratio_model(theta, C, k):
    return C * np.exp(np.log(2.0) * k * N_of_theta(theta))

# defining cone shape
true_phi = ak.firsts(mcp_phi_nobib)
true_phi_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_phi), ecal_hit_phi_nobib) ])
dphi = ecal_hit_phi_nobib - true_phi_broadcasted

true_theta = ak.firsts(mcp_theta_nobib)
true_theta_broadcasted = ak.Array([ [a]*len(b) for a, b in zip(ak.ravel(true_theta), ecal_hit_theta_nobib) ])
dtheta = ecal_hit_theta_nobib - true_theta_broadcasted

dR = np.sqrt(dphi ** 2 + dtheta **2)
cone_mask = dR < 0.2

# defining timing cut
time_mask_200 = (ecal_hit_time_nobib < 0.2) & (ecal_hit_time_nobib > -0.2)

true_energy = np.asarray(ak.firsts(mcp_energy), dtype=float)
true_theta = np.asarray(ak.firsts(mcp_theta_nobib), dtype=float)
measured_energy = np.asarray(ak.sum(ecal_hit_energy_nobib[cone_mask & time_mask_200], axis=1), dtype=float)

good = np.isfinite(true_energy) & np.isfinite(measured_energy) & (measured_energy > 1e-6)
true_energy = true_energy[good]
true_theta = true_theta[good]
measured_energy = measured_energy[good]
ratio = true_energy / measured_energy

# binning
theta_bins = np.linspace(0.2, 2.94, 40)
bin_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
bin_widths = theta_bins[1:] - theta_bins[:-1]
x_errors = bin_widths / 2

average_ratio = []

for i in range(len(theta_bins) - 1):
    mask = (true_theta >= theta_bins[i]) & (true_theta < theta_bins[i+1])
    bin_average_ratio = np.average(ratio[mask])

    average_ratio.append(bin_average_ratio) 

average_ratio = np.array(average_ratio)

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 7),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=True # share x-axis
)
ax1.scatter(true_theta, ratio, s=10, alpha=0.6, label=f"Data (cone only + 200 ps cut)")
ax1.errorbar(bin_centers, average_ratio, xerr=x_errors, fmt='o', color='black', label="Averaged data in each bin")

p0 = (1.0, 1.0) # start near the Heitler expectation
#bounds = ([0.1, 0.1], [10.0, 5.0])

popt, pcov = curve_fit(ratio_model, true_theta, ratio, p0=p0, maxfev=20000)
C_fit, k_fit = popt
C_err, k_err = np.sqrt(np.diag(pcov))

thetas = np.linspace(true_theta.min(), true_theta.max(), 400)
y_fit  = ratio_model(thetas, C_fit, k_fit)
ax1.plot(thetas, y_fit, lw=2, color='red', label=fr"Fit: $C={C_fit:.3f}\pm{C_err:.3f}$, $k={k_fit:.3f}\pm{k_err:.3f}$")

ax1.set_ylabel(r"$E_0/E_f$")
ax1.legend()

# residual
#fit_at_data = a * true_energy + b
#rel_rate = 100.0 * (measured_energy - fit_at_data) / fit_at_data

#ax2.axhline(0, color="black", lw=1)
#ax2.plot(true_energy, rel_rate, ".", alpha=0.5)
#ax2.set_ylabel("Relative Rate (w.r.t. fit) [%]")
ax2.set_xlabel("Theta [rad]")

fig.suptitle(rf"Theta calibrations ({yn}, $100<E_0<150$ GeV)", y=0.98)
# Central barrel region: $0.99 < \theta < 2.15$, Transition region: $0.7 < \theta < 0.99, 2.15 < \theta < 2.44$, Endcap region: $0.12 < \theta < 0.7, 2.44 < \theta < 3.02$

plt.tight_layout()
plt.show()