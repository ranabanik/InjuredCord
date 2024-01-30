# 01.04.2024
from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib as mtl
mtl.use("TKAgg")
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler
from Codes.Utilities import find_nearest, _smooth_spectrum, umap_it, hdbscan_it, chart
from Codes.Utilities import saveimages, poissonScaling
from scipy.signal import argrelextrema
from scipy.stats import kruskal
import pandas as pd
import shap
import numpy as np
from pyimzml.ImzMLParser import _bisect_spectrum
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
rng = np.random.RandomState(31337)

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'

h5Path = os.path.join(fileDir, 'SavGolwl55po2BaseCorrdeg6max1000tol5XlocYlocRegID.h5')

with h5py.File(h5Path, 'r') as pfile:
    print(pfile.keys())
    processedSpectra = np.array(pfile['processedspectra'])
    mzs = np.array(pfile['rawmzs'])
    xCor = np.array(pfile['xCor'])
    yCor = np.array(pfile['yCor'])
    corRegID = np.array(pfile['corRegID'])
meanSpec = np.mean(processedSpectra, axis=0)
peakPositions = argrelextrema(meanSpec, np.greater)[0]
# processedSpectra = processedSpectra[:, peakPositions]
# print("peak selected spectra:", processedSpectra.shape)
peakmzs = mzs#[peakPositions]

gmcsvDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/GM_injured_cord1/'
wmcsvDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/WM_injured_cord1/'

gm1csv = glob(os.path.join(gmcsvDir, '*1*spots.csv'))
gm2csv = glob(os.path.join(gmcsvDir, '*2*spots.csv'))
gm3csv = glob(os.path.join(gmcsvDir, '*3*spots.csv'))
gm4csv = glob(os.path.join(gmcsvDir, '*4*spots.csv'))
gm5csv = glob(os.path.join(gmcsvDir, '*5*spots.csv'))
gm6csv = glob(os.path.join(gmcsvDir, '*6*spots.csv'))

gm1df = pd.read_csv(gm1csv[0])
gm2df = pd.read_csv(gm2csv[0])
gm3df = pd.read_csv(gm3csv[0])
gm4df = pd.read_csv(gm4csv[0])
gm5df = pd.read_csv(gm5csv[0])
gm6df = pd.read_csv(gm6csv[0])

wm1csv = glob(os.path.join(wmcsvDir, '*1*spots.csv'))
wm2csv = glob(os.path.join(wmcsvDir, '*2*spots.csv'))
wm3csv = glob(os.path.join(wmcsvDir, '*3*spots.csv'))
wm4csv = glob(os.path.join(wmcsvDir, '*4*spots.csv'))
wm5csv = glob(os.path.join(wmcsvDir, '*5*spots.csv'))
wm6csv = glob(os.path.join(wmcsvDir, '*6*spots.csv'))

wm1df = pd.read_csv(wm1csv[0])
wm2df = pd.read_csv(wm2csv[0])
wm3df = pd.read_csv(wm3csv[0])
wm4df = pd.read_csv(wm4csv[0])
wm5df = pd.read_csv(wm5csv[0])
wm6df = pd.read_csv(wm6csv[0])

dflist = [
    gm1df, gm2df, gm3df, gm4df, gm5df, gm6df,
    wm1df, wm2df, wm3df, wm4df, wm5df, wm6df
         ]
savecsv = os.path.join(fileDir, 'tic_poisson_pca90_umap_hdbscan_gmwm_labels.csv')
labelDF = pd.read_csv(savecsv)
labels = labelDF['labels'].values #- 1

print(np.unique(labels), len(labels))

specIdx = []
for df in dflist:
    specIdx.extend(df['Spot index'].values)

print(len(specIdx))
# spectra = processedSpectra[specIdx]
# ionImageSpectra = np.zeros([len(specIdx), len(peakPositions)])
# tol = 25
# for idx, spot in enumerate(specIdx):
#     ints = processedSpectra[spot]
#     for jdx, mz_value in enumerate(peakmzs):
#         min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
#         ionImageSpectra[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
#     # ionImage3D[xCor[spot], yCor[spot], :] = ionImageSpectra[idx]
# print("ionImageSpectra.shape: ", ionImageSpectra.shape)
img = np.zeros([max(xCor)+1, max(yCor)+1])
print("img.shape", img.shape)
if 'colors' not in plt.colormaps():
    # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
    colors = [(0.05, 0.05, 0.05, 1.0),
              (0.15, 0.15, 0.75, 1.0),
              (0.75, 0.15, 0.15, 1.0),
              (0.15, 0.75, 0.15, 1.0),
              # (0.15, 0.15, 0.75, 1.0),
              # (0.95, 0.75, 0.05, 1.0),
              # (0.75, 0.15, 0.15, 1.0), # red
              # (0.9569, 0.7686, 0.1882),
              # (0.65, 0.71, 0.91),
              # (0.95, 0.3, 0.22),  # (0.0, 0.5, 0.5),
              # (0.01, 0.01, 0.95, 1.0),
              # (0.01, 0.95, 0.01, 1.0),  # (0.5, 0.5, 0.0),
              # (0.65, 0.71, 0.81),
              # (0.95, 0.01, 0.01, 1.0)
              ]
color_bin = len(np.unique(labels)) + 1   #len(gmdflist)
mtl.colormaps.register(LinearSegmentedColormap.from_list(name='colors', colors=colors, N=color_bin))
for i, s in enumerate(specIdx):
    x_ = xCor[s]
    y_ = yCor[s]
    img[x_, y_] = labels[i]     #+ 1
plt.imshow(img[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1], cmap='colors')
plt.colorbar()
plt.show()
# gm - 2, injured - 3, wm - 1
gmSpotIdx = []
wmSpotIdx = []
injuredSpotIdx = []
for (idx, l) in zip(specIdx, labels):
    if l == 2:
        gmSpotIdx.append(idx)
    if l == 3:
        injuredSpotIdx.append(idx)
    if l == 1:
        wmSpotIdx.append(idx)

print(len(gmSpotIdx), "\n", gmSpotIdx)
print(len(injuredSpotIdx), "\n", injuredSpotIdx)
print(len(wmSpotIdx), "\n", wmSpotIdx)

gmSpectra = processedSpectra[gmSpotIdx]
injuredSpectra = processedSpectra[injuredSpotIdx]
wmSpectra = processedSpectra[wmSpotIdx]

print(gmSpectra.shape, injuredSpectra.shape, wmSpectra.shape)
print(type(gmSpectra), type(injuredSpectra), type(wmSpectra))

gmMedian = np.median(gmSpectra, axis=0)
wmMedian = np.median(wmSpectra, axis=0)
injuredMedian = np.median(injuredSpectra, axis=0)

gmMean = np.mean(gmSpectra, axis=0)
wmMean = np.mean(wmSpectra, axis=0)
injuredMean = np.mean(injuredSpectra, axis=0)

# plt.vlines(peakmzs, 0, gmMean, label='gm', colors='g',
#                   linestyles='solid', alpha=0.5)
# plt.vlines(peakmzs, 0, wmMean, label='wm', colors='k',
#                   linestyles='solid', alpha=0.5)
# plt.vlines(peakmzs, 0, injuredMean, label='injured', colors='r',
#                   linestyles='solid', alpha=0.5)
# plt.legend(fontsize=20)
# plt.show()

plt.plot(peakmzs, gmMean, 'g', label='gm',
                   alpha=0.5)
plt.plot(peakmzs, wmMean, 'k', label='wm',
                  alpha=0.5)
plt.plot(peakmzs, injuredMean, 'r', label='injured',
                   alpha=0.5)
plt.legend(fontsize=20)
plt.show()

plt.fill_between(peakmzs, 0, gmMedian, label='gm', color='g', alpha=0.5, hatch="//")
plt.fill_between(peakmzs, 0, wmMedian, label='wm', color='k', alpha=0.5, hatch="-")
plt.fill_between(peakmzs, 0, injuredMedian, label='injured', color='r', alpha=0.5, hatch="\\")

plt.legend(fontsize=12)
plt.xlabel('Peak MZs', fontsize=14)
plt.ylabel('Median', fontsize=14)
plt.title('Area Under the Curve Plot', fontsize=16)
plt.show()

# plt.vlines(peakmzs, 0, injuredMedian-nearMedian, label='injured - healthy', colors='r',
#                   linestyles='solid', alpha=0.5)
# plt.title("up/down regulation of injured from healthy")
# plt.legend(fontsize=20)
# plt.show()

