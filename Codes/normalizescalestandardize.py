from glob import glob
import os
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtl
mtl.use("TKAgg")
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler as SS
from Codes.Utilities import find_nearest, _smooth_spectrum
from scipy.signal import argrelextrema

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
mzs = mzs[peakPositions]

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/GM_injured_cord/'

csv1 = glob(os.path.join(fileDir, '*1*spots.csv'))
csv2 = glob(os.path.join(fileDir, '*2*spots.csv'))
csv3 = glob(os.path.join(fileDir, '*3*spots.csv'))
csv4 = glob(os.path.join(fileDir, '*4*spots.csv'))
csv5 = glob(os.path.join(fileDir, '*5*spots.csv'))
csv6 = glob(os.path.join(fileDir, '*6*spots.csv'))
print(csv1)
gm1df = pd.read_csv(csv1[0])
gm2df = pd.read_csv(csv2[0])
gm3df = pd.read_csv(csv3[0])
gm4df = pd.read_csv(csv4[0])
gm5df = pd.read_csv(csv5[0])
gm6df = pd.read_csv(csv6[0])

gmdflist = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df]
# dfinjured = [gm4df, gm5df, gm6df]

specIdx = []
for gmdf in gmdflist:
    specIdx.extend(gmdf['Spot index'].values)

spectra = processedSpectra[specIdx]
print(spectra.shape)
regID = corRegID[specIdx]
regSpecNorm = np.zeros_like(spectra)
# regID = healthyRegID
# maxInt = np.max(gmspectra)
# nmz = gmspectra.shape[1]
# refSpec = regSpec[60]
# fnorm = sum(refSpec)
wl_ = 3
po_ = 1
for s in range(regSpecNorm.shape[0]):
    spectrum = spectra[s]
    # spectrum = _smooth_spectrum(spectrum, method='savgol', window_length=wl_, polyorder=po_)
    regSpecNorm[s] = spectrum/np.median(spectrum)

# # Level scaling
col_means = np.mean(regSpecNorm, axis=0)
# print(col_means.shape)
# # Divide each column by its mean
regSpecLevel = regSpecNorm / col_means
# # Pareto scaling
col_stddevs = np.std(regSpecNorm, axis=0)
#
# # Pareto scaling
regSpecPareto = (regSpecNorm - col_means) / np.sqrt(col_stddevs)
# +----------------+
# |      pca       |
# +----------------+
RandomState = 20210131
pca = PCA(random_state=RandomState) #, n_components=10)
regSpecNormSS = SS().fit_transform(regSpecNorm)     # 0-mean, 1-variance

plt.plot(mzs, regSpecNorm[40], 'r', alpha=0.5, label='med norm')
plt.plot(mzs, regSpecLevel[40], 'g', alpha=0.5, label='level scaled')
plt.plot(mzs, regSpecPareto[40], 'b', alpha=0.5, label='pareto scaled')
plt.plot(mzs, regSpecNormSS[40], 'k', alpha=0.5, label='standardized')
plt.plot(mzs, spectra[40], 'm', alpha=0.5, label='raw')
plt.legend()
plt.show()