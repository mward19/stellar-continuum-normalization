from pathlib import Path
from sklearn.decomposition import NMF
import numpy as np
from tqdm.auto import trange

from load_spectra import BOSZSpectra

# class BigNMF:
#     def __init__(self, row_loader):
#         self.loader = row_loader

#     def calc_nmf(self, k: int):
#         """Perform NMF using coordinate descent in batches. Number of basis vectors is `k`."""


def construct_S(spectra: BOSZSpectra):
    N = len(spectra)
    D = spectra.D
    S = np.empty((N, D), dtype=np.float32)

    

if __name__ == "__main__":
    root = Path("/home/mward19/nobackup/autodelete/bosz-spectra/data-test")
    
    spectra = BOSZSpectra(root)
    spectral_matrix = np.vstack([spectra[i] for i in trange(len(spectra))])
    nmf = NMF()