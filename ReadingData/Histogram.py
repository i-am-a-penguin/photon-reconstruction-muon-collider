import uproot
import matplotlib.pyplot as plt
import awkward as ak
import sys
import numpy as np 

file = uproot.open('../ntuple_photonGun_noBIB_MAIAv0_v2.root') # MODIFY
tree = file["Events"]
branches = tree.arrays()

branch_labels = {
    'mcp_pt': [r'True Transverse Momentum $p_{\mathrm{T}}$', '[GeV]'], 
    'mcp_energy': ['True Energy', '[GeV]'], 
    'mcp_theta': [r'True Polar Angle $\theta$', '[rad]'], 
    'mcp_phi': [r'True Azimuthal Angle $\phi$', '[rad]'], 
    'ecal_hit_energy': ['ECAL Hit Energy', '[GeV]'], 
    'ecal_hit_time': ['ECAL Hit Time', '[ns]'], 
    'ecal_hit_theta': [r'ECAL Hit Polar Angle $\theta$', '[rad]'], 
    'ecal_hit_phi': [r'ECAL Hit Azimuthal Angle $\phi$', '[rad]'], 
    'ecal_hit_depth': ['ECAL Hit Depth', '[mm]']
}

def plot(branch_name, label):
    data = ak.flatten(branches[branch_name])
    plt.hist(data, bins=int(sys.argv[2]), range=(0, np.percentile(data, 95)), histtype='step')
    plt.xlabel(label[0] + ' ' + label[1])
    plt.ylabel('Count')
    plt.title(f"Photon {label[0]} Distribution (no BIB)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    branch = sys.argv[1]

    if branch in branch_labels:
        plot(branch, branch_labels[branch])
    else:
        print(f"Branch {branch} not recognized.")
