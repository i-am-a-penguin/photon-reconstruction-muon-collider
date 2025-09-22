import uproot
import sys
import numpy as np

file = uproot.open('../ntuple_photonGun_noBIB_MAIAv0_v2.root') # MODIFY
tree = file["Events"]
branches = tree.arrays()

photon_mask = branches['mcp_pdg'] == 22

if __name__ == "__main__":
    calculated_pt = branches['mcp_energy'][photon_mask] * np.sin(branches['mcp_theta'][photon_mask])
    print(calculated_pt[int(sys.argv[1]):int(sys.argv[2])])
    
    recorded_pt = branches['mcp_pt'][photon_mask]
    print(recorded_pt[int(sys.argv[1]):int(sys.argv[2])])
