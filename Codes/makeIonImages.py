from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib as mtl
# mtl.use("TKAgg")
from pyimzml.ImzMLParser import _bisect_spectrum
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler as SS
from Codes.Utilities import saveimages, find_nearest, _smooth_spectrum, umap_it, hdbscan_it, chart
from scipy.signal import argrelextrema
import pandas as pd
import shap
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
rng = np.random.RandomState(31337)

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'

savepath = os.path.join(fileDir, 'SavGolwl55po2BaseCorrdeg6max1000tol5XlocYlocRegID.h5')
print(savepath)
with h5py.File(savepath, 'r') as pfile:
    print(pfile.keys())
    processedSpectra = np.array(pfile['processedspectra'])
    mzs = np.array(pfile['rawmzs'])
    xCor = np.array(pfile['xCor'])
    yCor = np.array(pfile['yCor'])
    corRegID = np.array(pfile['corRegID'])
meanSpec = np.mean(processedSpectra, axis=0)
peakPositions = argrelextrema(meanSpec, np.greater)[0]
# processedSpectra = processedSpectra[:, peakPositions]
print("peak selected spectra:", processedSpectra.shape)
peakmzs = mzs[peakPositions]

ionImageSpectra = np.zeros([len(xCor), len(peakPositions)])
print(ionImageSpectra.shape)
ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
#
tol = 50
for idx, spectrum in enumerate(processedSpectra):
    for jdx, mz_value in enumerate(peakmzs):
        min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
        ionImageSpectra[idx, jdx] = sum(spectrum[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
    ionImage3D[xCor[idx], yCor[idx], :] = ionImageSpectra[idx]
# saveimages(ionImage3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, :], peakmzs, r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/IonImagesSC/')
savepath = os.path.join(fileDir, 'IonImageSpecTol{}.h5'.format(tol))
with h5py.File(savepath, 'w') as pfile:
    pfile['spectra'] = np.array(ionImageSpectra)
    pfile['peakmzs'] = np.array(peakmzs, dtype=peakmzs[0].dtype)
    pfile['xloc'] = np.array(xCor, dtype=type(xCor[0]))
    pfile['yloc'] = np.array(yCor, dtype=type(xCor[0]))