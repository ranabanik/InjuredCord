from sklearn.decomposition import KernelPCA
import numpy as np
import os
from glob import glob
import h5py
from Codes.Utilities import find_nearest, _smooth_spectrum
from sklearn.preprocessing import StandardScaler as SS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mtl
mtl.use("TKAgg")
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import ndimage
from scipy.signal import argrelextrema
from sklearn.cluster import AgglomerativeClustering

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'
wl_ = 55
po_ = 5
savepath = os.path.join(fileDir, 'SavGolwl55po2BaseCorrdeg6max1000tol5XlocYlocRegID.h5')
print(savepath)
with h5py.File(savepath, 'r') as pfile:
    print(pfile.keys())
    processedSpectra = np.array(pfile['processedspectra'])
    mzs = np.array(pfile['rawmzs'])
    xCor = np.array(pfile['xCor'])
    yCor = np.array(pfile['yCor'])
    corRegID = np.array(pfile['corRegID'])

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

gmdfs = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df]

allspot = []
for gmdf in gmdfs:
    allspot.extend(gmdf['Spot index'].values)

gmspectra = processedSpectra[allspot]
print(gmspectra.shape)
gmCorRegID = corRegID[allspot]
regSpecNorm = np.zeros_like(gmspectra)
maxInt = np.max(gmspectra)
nmz = gmspectra.shape[1]
# refSpec = regSpec[60]
# fnorm = sum(refSpec)
for s in range(gmspectra.shape[0]):
    spectrum = gmspectra[s]
    spectrum = _smooth_spectrum(spectrum, method='savgol', window_length=wl_, polyorder=po_)
    regSpecNorm[s] = spectrum/np.median(spectrum)

# # Level scaling
col_means = np.mean(regSpecNorm, axis=0)
# print(col_means.shape)
# # Divide each column by its mean
regSpecLeveled = regSpecNorm / col_means

# # Pareto scaling
# col_means = np.mean(regSpecNorm, axis=0)
# col_stddevs = np.std(regSpecNorm, axis=0)
#
# # Pareto scaling
# regSpecPareto = (regSpecNorm - col_means) / np.sqrt(col_stddevs)
# +----------------+
# |      pca       |
# +----------------+
RandomState = 20210131
kpca1 = KernelPCA(kernel='rbf', gamma=0.1)
regSpecNormSS = SS().fit_transform(regSpecNorm) # 0-mean, 1-variance
pcScores = kpca1.fit_transform(regSpecNormSS)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# markers = {1: 'o', 2: '^', 3: 's', 4: '+', 5: 'D', 6: 'p'}
markers = {1: 'o', 2: '^', 3: 's', 4: 'D', 5: 'v', 6: 'p'}
if 'colors_' not in plt.colormaps():
    # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
    colors = [(0.15, 0.95, 0.95, 1.0),
              (0.10, 0.10, 0.10, 1.0),
              (0.15, 0.75, 0.15, 1.0),
              (0.15, 0.15, 0.75, 1.0),
              (0.95, 0.75, 0.05, 1.0),
              (0.75, 0.15, 0.15, 1.0), # red
              # (0.9569, 0.7686, 0.1882),
              # (0.65, 0.71, 0.91),
              # (0.95, 0.3, 0.22),  # (0.0, 0.5, 0.5),
              # (0.01, 0.01, 0.95, 1.0),
              # (0.01, 0.95, 0.01, 1.0),  # (0.5, 0.5, 0.0),
              # (0.65, 0.71, 0.81),
              # (0.95, 0.01, 0.01, 1.0)
              ]
color_bin = 6
mtl.colormaps.register(LinearSegmentedColormap.from_list(name='colors_', colors=colors, N=color_bin))

scatter = ax.scatter(pcScores[:, 0], pcScores[:, 1], pcScores[:, 2],
                     c=gmCorRegID,
                     cmap='colors_', #marker='o')
                     marker='o')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Score Plot with region labels (scils)')
# Add a legend
# for region, marker in markers.items():
#     ax.scatter([], [], [], marker='+', label=f'Region {region}')

# Display the legend
ax.legend()
cbar = plt.colorbar(scatter)
plt.show()
