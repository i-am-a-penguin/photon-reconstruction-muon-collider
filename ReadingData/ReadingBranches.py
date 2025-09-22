import uproot
import matplotlib.pyplot as plt
import awkward as ak
import sys

file = uproot.open('../ntuple_photonGun_noBIB_MAIAv5.root') # MODIFY
tree = file["Events"]
branches = tree.arrays()

def look(branch_name, start, end):
    """
    Print basic statistics about a branch.
    """
    data = branches[branch_name][start:end+1]
    print(data)
    print("length of each sublist:", ak.num(data))
    print("maximum in each sublist:", ak.max(data, axis=1))
    print("minimum in each sublist:", ak.min(data, axis=1))
    print("mean:", ak.mean(data))
    print("sum:", ak.sum(data))

if __name__ == "__main__":
    branch = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    look(branch, start, end)