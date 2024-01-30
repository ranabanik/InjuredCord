# 12.21.2023
# 12.18.2023
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
# print(csv1)
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
# print(csv1)
wm1df = pd.read_csv(wm1csv[0])
wm2df = pd.read_csv(wm2csv[0])
wm3df = pd.read_csv(wm3csv[0])
wm4df = pd.read_csv(wm4csv[0])
wm5df = pd.read_csv(wm5csv[0])
wm6df = pd.read_csv(wm6csv[0])

# wmdflist = [wm1df, wm2df, wm3df, wm4df, wm5df, wm6df]
dfList = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df,
          wm1df, wm2df, wm3df, wm4df, wm5df, wm6df]

specIdx = []
tissuelabels = []
# for i, gm_df in enumerate(gmdflist, start=1):
#     specIdx.extend(gm_df['Spot index'].values)
#     num_samples = len(gm_df['Spot index'])
#     labels = [i] * num_samples
#     tissuelabels.extend(labels)

for i, wm_df in enumerate(dfList, start=1):
    specIdx.extend(wm_df['Spot index'].values)
    num_samples = len(wm_df['Spot index'])
    labels = [i] * num_samples
    tissuelabels.extend(labels)
print("number of labels: ", len(np.unique(tissuelabels)), ">>",  np.unique(tissuelabels))
spectra = processedSpectra[specIdx]
# spectra = spectra[:, peakPositions]
print("#96 -> spectra.shape", spectra.shape)
regID = corRegID[specIdx]
ionImageSpectra = np.zeros([len(specIdx), len(peakmzs)])
ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakmzs)])
# # spectra3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
tol = 20
for idx, spot in enumerate(specIdx):
    ints = processedSpectra[spot]
    for jdx, mz_value in enumerate(peakmzs):
        min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
        ionImageSpectra[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
    ionImage3D[xCor[spot], yCor[spot], :] = ionImageSpectra[idx]
    # spectra3D[xCor[spot], yCor[spot], :] = spectra[idx]

print("#110 -> ionImageSpec.shape", ionImageSpectra.shape)
# saveimages(ionImage3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, :], peakmzs, r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/ionimagesWM/')
# saveimages(spectra3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, :], peakmzs, r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/ionimages_bad/')

# mzv_ = find_nearest(peakmzs, 11553.0)
# plt.subplot(121)
# plt.imshow(ionImage3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, np.where(peakmzs==mzv_)[0][0]])
# plt.subplot(122)
# plt.imshow(spectra3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, np.where(peakmzs==mzv_)[0][0]])
# plt.show()

# spectra = ionImageSpec
# mzs = peakmzs
regSpecNorm = np.zeros_like(ionImageSpectra) #+ 1e-6
for s in range(regSpecNorm.shape[0]):
    spectrum = ionImageSpectra[s]
    # spectrum = _smooth_spectrum(spectrum, method='savgol', window_length=wl_, polyorder=po_)
    regSpecNorm[s] = spectrum/np.median(spectrum)

# # Level scaling
col_means = np.mean(regSpecNorm, axis=0)
# # Divide each column by its mean
# regSpecScaled = regSpecNorm / col_means
# # Pareto scaling
col_stddevs = np.std(regSpecNorm, axis=0)
#
# # Pareto scaling
regSpecScaled = (regSpecNorm - col_means) / np.sqrt(col_stddevs)

# # Poisson scaling
# regSpecScaled = poissonScaling(regSpecNorm)
# +----------------+
# |      pca       |
# +----------------+
RandomState = 20210131
pca = PCA(random_state=RandomState) #, n_components=10)
# regSpecNormSS = SS().fit_transform(regSpecScaled) # 0-mean, 1-variance
pcScores = pca.fit_transform(regSpecScaled)
print("pcs shape: ", pcScores.shape)
pca_range = np.arange(1, pca.n_components_, 1)
print(">> PCA: number of components #{}".format(pca.n_components_))
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
threshold = 0.90
cut_evr = find_nearest(evr_cumsum, threshold)
nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
print(">> Nearest variance to threshold {:.4f} explained by #PCA components {}".format(cut_evr, nPCs))
if nPCs >= 50:  # 2 conditions to choose nPCs.
    nPCs = 50

mz_100 = ionImageSpectra[..., 100].reshape(-1, 1)
mz_101 = ionImageSpectra[..., 101].reshape(-1, 1)
mz_200 = ionImageSpectra[..., 200].reshape(-1, 1)
scaler = MinMaxScaler()
mz_100_ = scaler.fit_transform(mz_100)
mz_101_ = scaler.fit_transform(mz_101)
mz_200_ = scaler.fit_transform(mz_200)

correlation_result = np.corrcoef(mz_100.flat, mz_100.flat)
r = correlation_result[0, 1]
print("r >>", r)

pcScores = pcScores[:, 0:nPCs]
correlation_result_PCs = np.zeros([ionImageSpectra.shape[1], nPCs])
print("this : ", correlation_result_PCs.shape)
for n in range(nPCs):
    mzPc = pcScores[:, n].reshape(-1, 1)
    # print("this too:", mzPc.shape)
    mzPc = scaler.fit_transform(mzPc)
    for nmz in range(ionImageSpectra.shape[1]):
        mzSpec = ionImageSpectra[..., nmz].reshape(-1, 1)
        mzSpec = scaler.fit_transform(mzSpec)
        # print("this too:", mzSpec.shape)
        correlation_result = np.corrcoef(mzPc.flat, mzSpec.flat)
        # print("result", correlation_result[0, 1])
        correlation_result_PCs[nmz, n] = correlation_result[0, 1]
# plt.imshow(correlation_result_PCs)
# plt.colorbar()
# plt.show()
plt.vlines(peakmzs, 0, correlation_result_PCs.T[0], label='PC 1', colors='k',
                  linestyles='solid', alpha=0.5)
plt.vlines(peakmzs, 0, correlation_result_PCs.T[1], label='PC 2', colors='g',
                  linestyles='solid', alpha=0.5)
plt.vlines(peakmzs, 0, correlation_result_PCs.T[2], label='PC 3', colors='b',
                  linestyles='solid', alpha=0.5)
plt.legend()
plt.show()