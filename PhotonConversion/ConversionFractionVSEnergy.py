import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt

yn = "noBIB"
version = "v5"
file = uproot.open(f"../ntuple_photonGun_{yn}_MAIA{version}.root") # MODIFY
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
mcp_pt = tree["mcp_pt"].array()
mcp_pdg = tree["mcp_pdg"].array()
mcp_vx = tree["mcp_vx"].array()
mcp_vy = tree["mcp_vy"].array()
mcp_vz = tree["mcp_vz"].array()

def wrap_dphi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2*np.pi) - np.pi

# configuration (tune as needed)
PT_MIN     = 1
DR_MAX     = 0.2
E_RATIO_LO = 0.6
E_RATIO_HI = 1.1
VERTEX_MIN = 0
VERTEX_MAX = 1857

def photon_conversion_check(mcp_pdg, mcp_energy, mcp_pt, mcp_theta, mcp_phi, mcp_vx, mcp_vy, mcp_vz):
    pdg = mcp_pdg
    E   = mcp_energy
    pt  = mcp_pt
    th  = mcp_theta
    ph  = mcp_phi
    vx  = mcp_vx
    vy  = mcp_vy
    vz  = mcp_vz

    pdg_primary = ak.firsts(pdg)
    is_primary_photon = (pdg_primary == 22) # outputs [True, False, False, ...]
    
    E_gamma_primary = ak.where(is_primary_photon, ak.firsts(E), np.nan)
    th_gamma = ak.where(is_primary_photon, ak.firsts(th), np.nan)
    ph_gamma = ak.where(is_primary_photon, ak.firsts(ph), np.nan)

    # masks for particle types
    is_ep    = (pdg == +11)
    is_em    = (pdg == -11)

    # apply pT selection to leptons
    ep_sel = is_ep & (pt > PT_MIN) # outputs [[False, True], [True, True, False], ...]
    em_sel = is_em & (pt > PT_MIN)

    # count selected e+ and e- per event
    n_ep = ak.sum(ep_sel, axis=1) # outputs [3, 0, 2, ...]
    n_em = ak.sum(em_sel, axis=1)
    has_ep = (n_ep > 0)
    has_em = (n_em > 0)
    has_both = has_ep & has_em

    # pick the HIGHEST-ENERGY selected e+ and e- per event (indices)
    idx_ep_max = ak.argmax(ak.where(ep_sel, E, -np.inf), axis=1, keepdims=True)
    idx_em_max = ak.argmax(ak.where(em_sel, E, -np.inf), axis=1, keepdims=True)

    # grab their (theta, phi, energy, vx, vy); if none exists, fill with NaN
    th_ep = ak.firsts(th[idx_ep_max])
    ph_ep = ak.firsts(ph[idx_ep_max])
    E_ep_max = ak.firsts(E[idx_ep_max])
    vx_ep = ak.firsts(vx[idx_ep_max])
    vy_ep = ak.firsts(vy[idx_ep_max])
    vz_ep = ak.firsts(vz[idx_ep_max])

    th_em = ak.firsts(th[idx_em_max])
    ph_em = ak.firsts(ph[idx_em_max])
    E_em_max = ak.firsts(E[idx_em_max])
    vx_em = ak.firsts(vx[idx_em_max])
    vy_em = ak.firsts(vy[idx_em_max])
    vz_em = ak.firsts(vz[idx_em_max])

    # angular consistency requirement (both within DR_MAX)
    dphi_ep = wrap_dphi(ph_ep, ph_gamma)
    dphi_em = wrap_dphi(ph_em, ph_gamma)
    dR_ep = np.sqrt((th_ep - th_gamma)**2 + dphi_ep**2)
    dR_em = np.sqrt((th_em - th_gamma)**2 + dphi_em**2)
    dR_ok = (dR_ep < DR_MAX) & (dR_em < DR_MAX)

    # energy consistency
    E_pair = E_ep_max + E_em_max
    fE = E_pair / E_gamma_primary
    fE_ok = (fE > E_RATIO_LO) & (fE < E_RATIO_HI)

    # vertex location check
    v_ep = np.sqrt(vx_ep**2 + vy_ep**2)
    v_em = np.sqrt(vx_em**2 + vy_em**2)
    location_ok = (v_ep > VERTEX_MIN) & (v_ep < VERTEX_MAX) & (v_em > VERTEX_MIN) & (v_em < VERTEX_MAX)

    # positron and electron vertices coincide
    v_ok = (vx_ep == vx_em) & (vy_ep == vy_em) & (vz_ep == vz_em)

    is_converted = is_primary_photon & has_both & dR_ok & fE_ok & location_ok & v_ok
    is_converted = ak.fill_none(is_converted, False)

    for name, arr in [("is_primary_photon", is_primary_photon), ("has_both", has_both), ("dR_ok", dR_ok), ("fE_ok", fE_ok), ("v_ok", v_ok)]:
        print(name, ak.type(arr))
    
    return is_converted

def get_conversion_vertex(mcp_pdg, mcp_energy, mcp_pt, mcp_vx, mcp_vy, mcp_vz):
    pdg = mcp_pdg
    E   = mcp_energy
    pt  = mcp_pt
    vx  = mcp_vx
    vy  = mcp_vy
    vz  = mcp_vz

    is_ep = (pdg == +11)
    is_em = (pdg == -11)
    ep_sel = is_ep & (pt > PT_MIN)
    em_sel = is_em & (pt > PT_MIN)

    idx_ep_max = ak.argmax(ak.where(ep_sel, E, -np.inf), axis=1, keepdims=True)
    idx_em_max = ak.argmax(ak.where(em_sel, E, -np.inf), axis=1, keepdims=True)

    vx_ep = ak.firsts(vx[idx_ep_max])
    vy_ep = ak.firsts(vy[idx_ep_max])
    vz_ep = ak.firsts(vz[idx_ep_max])
    vx_em = ak.firsts(vx[idx_em_max])
    vy_em = ak.firsts(vy[idx_em_max])
    vz_em = ak.firsts(vz[idx_em_max])

    vx = 0.5 * (vx_ep + vx_em) # vx should be the same as vx_ep and vx_em since vx_ep = vx_em
    vy = 0.5 * (vy_ep + vy_em)
    vz = 0.5 * (vz_ep + vz_em)
    vr = np.sqrt(vx ** 2 + vy ** 2)

    return {"x": vx, "y": vy, "r": vr, "z": vz}

is_conv = photon_conversion_check(mcp_pdg, mcp_energy, mcp_pt, mcp_theta, mcp_phi, mcp_vx, mcp_vy, mcp_vz)
print(f"Number of photon conversion events: {int(ak.sum(is_conv))}")
print(f"Fraction of photon conversion events: {float(ak.mean(is_conv)):.4f}")
energy = ak.firsts(mcp_energy)

nbins = 20
edges = np.linspace(0, 1000, nbins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
bin_widths = edges[1:] - edges[:-1]
x_errors = bin_widths / 2

fracs = []
errors = []
for i in range(nbins):
    lo, hi = edges[i], edges[i+1]
    sel = (energy >= lo) & ((energy < hi) if i < nbins - 1 else (energy <= hi))

    n_tot  = int(ak.sum(sel))
    n_conv = int(ak.sum(is_conv & sel))
    err_conv = np.sqrt(n_conv)

    fracs.append(np.nan if n_tot == 0 else n_conv / n_tot)
    errors.append(np.nan if n_tot == 0 else err_conv / n_tot)

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(centers, fracs, yerr=errors, xerr=x_errors, fmt='o', color='black',
            markersize=4, capsize=3)
ax.set_xlabel(r"Energy [GeV]")
ax.set_ylabel("Conversion fraction")
ax.set_xlim(0, 1000)
ax.set_ylim(0, 0.25)
ax.set_title(rf"Photon conversion fraction vs true energy ({yn})")
fig.tight_layout()
plt.show()