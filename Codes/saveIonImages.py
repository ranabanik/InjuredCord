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

fileDir1 = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'

savepath = os.path.join(fileDir1, 'SavGolwl55po2BaseCorrdeg6max1000tol5XlocYlocRegID.h5')
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
processedSpectra = processedSpectra[:, peakPositions]
print("peak selected spectra:", processedSpectra.shape)
peakmzs = mzs[peakPositions]

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/GM_injured_cord1/'

csv1 = glob(os.path.join(fileDir, '*1_spots.csv'))
csv2 = glob(os.path.join(fileDir, '*2_spots.csv'))
csv3 = glob(os.path.join(fileDir, '*3_spots.csv'))
csv4 = glob(os.path.join(fileDir, '*4_spots.csv'))
csv5 = glob(os.path.join(fileDir, '*5_spots.csv'))
csv6 = glob(os.path.join(fileDir, '*6_spots.csv'))
print(csv1)
wm1df = pd.read_csv(csv1[0])
wm2df = pd.read_csv(csv2[0])
wm3df = pd.read_csv(csv3[0])
wm4df = pd.read_csv(csv4[0])
wm5df = pd.read_csv(csv5[0])
wm6df = pd.read_csv(csv6[0])

wmdflist = [wm1df, wm2df, wm3df, wm4df, wm5df, wm6df]
# dfinjured = [gm4df, gm5df, gm6df]

specIdx = []
for wmdf in wmdflist:
    specIdx.extend(wmdf['Spot index'].values)

spectra = processedSpectra[specIdx]
print(spectra.shape)
regID = corRegID[specIdx]

ionImageSpec = np.zeros([len(specIdx), len(peakPositions)])
ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
# spectra3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
tol = 20
for idx, spot in enumerate(specIdx):
    ints = processedSpectra[spot]
    for jdx, mz_value in enumerate(peakmzs):
        min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
        ionImageSpec[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
    ionImage3D[xCor[spot], yCor[spot], :] = ionImageSpec[idx]
    # spectra3D[xCor[spot], yCor[spot], :] = spectra[idx]
saveimages(ionImage3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, :], peakmzs, r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/ionImages/')
# saveimages(spectra3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, :], peakmzs, r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/ionimages_bad/')
