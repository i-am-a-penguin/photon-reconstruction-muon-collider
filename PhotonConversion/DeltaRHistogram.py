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

    is_converted = is_primary_photon & has_both & fE_ok & location_ok & dR_ok & v_ok
    is_converted = ak.fill_none(is_converted, False)

    for name, arr in [("is_primary_photon", is_primary_photon), ("has_both", has_both), ("dR_ok", dR_ok), ("fE_ok", fE_ok), ("v_ok", v_ok)]:
        print(name, ak.type(arr))
    
    return is_converted

def get_conversion_direction(mcp_pdg, mcp_energy, mcp_pt, mcp_theta, mcp_phi):
    pdg = mcp_pdg
    E   = mcp_energy
    pt  = mcp_pt
    th = mcp_theta
    phi = mcp_phi

    is_ep = (pdg == +11)
    is_em = (pdg == -11)
    pt_mask = (pt > 200) # MODIFY transverse momentum cuts
    ep_sel = is_ep & (pt > PT_MIN) & pt_mask
    em_sel = is_em & (pt > PT_MIN) & pt_mask

    idx_ep_max = ak.argmax(ak.where(ep_sel, E, -np.inf), axis=1, keepdims=True)
    idx_em_max = ak.argmax(ak.where(em_sel, E, -np.inf), axis=1, keepdims=True)

    th_ep = ak.firsts(th[idx_ep_max])
    phi_ep = ak.firsts(phi[idx_ep_max])

    th_em = ak.firsts(th[idx_em_max])
    phi_em = ak.firsts(phi[idx_em_max])

    return {"electron_theta": th_em, "electron_phi": phi_em, "positron_theta": th_ep, "positron_phi": phi_ep}

is_conv = photon_conversion_check(mcp_pdg, mcp_energy, mcp_pt, mcp_theta, mcp_phi, mcp_vx, mcp_vy, mcp_vz)

new_vx = mcp_vx[is_conv]
new_vy = mcp_vy[is_conv]
new_vz = mcp_vz[is_conv]
new_theta = mcp_theta[is_conv]
new_phi = mcp_phi[is_conv]
new_pdg = mcp_pdg[is_conv]
new_energy = mcp_energy[is_conv]
new_pt = mcp_pt[is_conv]

photon_theta = ak.firsts(new_theta)
photon_phi = ak.firsts(new_phi)

direction = get_conversion_direction(new_pdg, new_energy, new_pt, new_theta, new_phi)
electron_theta = ak.to_numpy(direction["electron_theta"])
electron_phi = ak.to_numpy(direction["electron_phi"])
positron_theta = ak.to_numpy(direction["positron_theta"])
positron_phi = ak.to_numpy(direction["positron_phi"])

electron_dphi = wrap_dphi(electron_phi, photon_phi)
positron_dphi = wrap_dphi(positron_phi, photon_phi)
electron_dR = np.sqrt((electron_theta - photon_theta) ** 2 + electron_dphi ** 2)
positron_dR = np.sqrt((positron_theta - photon_theta) ** 2 + positron_dphi ** 2)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(electron_dR, bins=50, range=(0,0.0005), histtype="step", label='electron', color="red")
ax.hist(positron_dR, bins=50, range=(0,0.0005), histtype="step", label='positron', color="blue")
ax.set_xlabel(r"$\Delta R$ [rad]")
ax.set_ylabel("Counts")
ax.set_title(rf"$\Delta R$ between photon and leptons in photon conversion events ({yn}, $p_T>200$)")
fig.tight_layout()
ax.legend()
plt.show()