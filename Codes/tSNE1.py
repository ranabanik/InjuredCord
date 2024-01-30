# 01.29.2023
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
from sklearn.manifold import TSNE
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

# dflist = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df]
# dflist = [wm1df, wm2df, wm3df, wm4df, wm5df, wm6df]
dfList = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df,
          wm1df, wm2df, wm3df, wm4df, wm5df, wm6df]

specIdx = []
tissuelabels = []
# for i, gm_df in enumerate(gmdflist, start=1):
#     specIdx.extend(gm_df['Spot index'].values)
#     num_samples = len(gm_df['Spot index'])
#     labels = [i] * num_samples
#     tissuelabels.extend(labels)

for i, df in enumerate(dfList, start=1):
    specIdx.extend(df['Spot index'].values)
    num_samples = len(df['Spot index'])
    labels = [i] * num_samples
    tissuelabels.extend(labels)
print("number of labels: ", len(np.unique(tissuelabels)), ">>",  np.unique(tissuelabels))
spectra = processedSpectra[specIdx]
# spectra = spectra[:, peakPositions]
print("#95 -> spectra.shape", spectra.shape)
# regID = corRegID[specIdx]
# ionImageSpectra = np.zeros([len(specIdx), len(peakPositions)])
# ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
# # spectra3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
# tol = 20
# for idx, spot in enumerate(specIdx):
#     ints = processedSpectra[spot]
#     for jdx, mz_value in enumerate(peakmzs):
#         min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
#         ionImageSpectra[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
#     ionImage3D[xCor[spot], yCor[spot], :] = ionImageSpectra[idx]
    # spectra3D[xCor[spot], yCor[spot], :] = spectra[idx]

# print("#110 -> ionImageSpec.shape", ionImageSpectra.shape)
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
regSpecNorm = np.zeros_like(spectra) #+ 1e-6
for s in range(regSpecNorm.shape[0]):
    spectrum = spectra[s]
    # spectrum = _smooth_spectrum(spectrum, method='savgol', window_length=wl_, polyorder=po_)
    regSpecNorm[s] = spectrum/np.median(spectrum)

# # Level scaling
# col_means = np.mean(regSpecNorm, axis=0)
# # Divide each column by its mean
# regSpecScaled = regSpecNorm / col_means
# # Pareto scaling
# col_stddevs = np.std(regSpecNorm, axis=0)
#
# # Pareto scaling
# regSpecScaled = (regSpecNorm - col_means) / np.sqrt(col_stddevs)

# # Poisson scaling
regSpecScaled = poissonScaling(regSpecNorm, offsetprc=0.01)
# +----------------+
# |      pca       |
# +----------------+
RandomState = 20210131
pca = PCA(random_state=RandomState) #, n_components=10)
# regSpecNormSS = SS().fit_transform(regSpecNorm) # 0-mean, 1-variance
pcScores = pca.fit_transform(regSpecScaled)
print("pcs shape: ", pcScores.shape)
pca_range = np.arange(1, pca.n_components_, 1)
print(">> PCA: number of components #{}".format(pca.n_components_))
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
threshold = 0.95
cut_evr = find_nearest(evr_cumsum, threshold)
nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
print(">> Nearest variance to threshold {:.4f} explained by #PCA components {}".format(cut_evr, nPCs))
if nPCs >= 50:  # 2 conditions to choose nPCs.
    nPCs = 50
# df_pca = pd.DataFrame(data=pcs[:, 0:nPCs], columns=['PC_%d' % (i + 1) for i in range(nPCs)])
plot_pca = True
MaxPCs = nPCs + 3
fig, ax = plt.subplots()
ax.bar(pca_range[0:MaxPCs], evr[0:MaxPCs] * 100, color="steelblue")
ax.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
ax.set_xlabel('Principal component number', fontsize=15)
ax.set_ylabel('Percentage of \n variance explained', fontsize=15)
ax.set_ylim([-0.5, 100])
ax.set_xlim([-0.5, MaxPCs])
ax.grid("on")

ax2 = ax.twinx()
ax2.plot(pca_range[0:MaxPCs], evr_cumsum[0:MaxPCs] * 100, color="tomato", marker="D", ms=7)
ax2.scatter(nPCs, cut_evr * 100, marker='*', s=500, facecolor='blue')
ax2.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
ax2.set_ylabel('Cumulative percentage', fontsize=15)
ax2.set_ylim([-0.5, 100])

# axis and tick theme
ax.tick_params(axis="y", colors="steelblue")
ax2.tick_params(axis="y", colors="tomato")
ax.tick_params(size=10, color='black', labelsize=15)
ax2.tick_params(size=10, color='black', labelsize=15)
ax.tick_params(width=3)
ax2.tick_params(width=3)

ax = plt.gca()  # Get the current Axes instance

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(3)

plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=15)
plt.show()

componentMatrix = pca.components_.T  # eigenvectors or weights or components or coefficients
print(componentMatrix.shape)
componentMatrixDF = pd.DataFrame(componentMatrix[:, 0:nPCs], columns=['PC%02d' % (i + 1) for i in range(nPCs)])
# print(componentMatrixDF)
loadings = pca.components_.T #* np.sqrt(pca.explained_variance_)
print(loadings.shape)
loadingsMatrixDF = pd.DataFrame(loadings[:, 0:nPCs], columns=['PC%02d' % (i + 1) for i in range(nPCs)])
# loadingsMatrixDF.insert(loadingsMatrixDF.shape[1], column='mzs', value=mzs)
# loadingsMatrixDF.loc[len(loadingsMatrixDF)] = pca.explained_variance_ratio_[0:nPCs+1]
savecsv = os.path.join(fileDir, 'loadings5regions.csv')
# loadingsMatrixDF.to_csv(savecsv, index=False, sep=',')
# plt.plot(mzs, loadingsMatrixDF['PC1'].values)
# plt.show()
# if plot_pca:
#     MaxPCs = nPCs + 3
#     fig, ax = plt.subplots(figsize=(20, 8), dpi=200)
#     ax.bar(pca_range[0:MaxPCs], evr[0:MaxPCs] * 100, color="steelblue")
#     ax.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
#     ax.set_xlabel('Principal component number', fontsize=30)
#     ax.set_ylabel('Percentage of \n variance explained', fontsize=30)
#     ax.set_ylim([-0.5, 100])
#     ax.set_xlim([-0.5, MaxPCs])
#     ax.grid("on")
#
#     ax2 = ax.twinx()
#     ax2.plot(pca_range[0:MaxPCs], evr_cumsum[0:MaxPCs] * 100, color="tomato", marker="D", ms=7)
#     ax2.scatter(nPCs, cut_evr * 100, marker='*', s=500, facecolor='blue')
#     ax2.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
#     ax2.set_ylabel('Cumulative percentage', fontsize=30)
#     ax2.set_ylim([-0.5, 100])
#
#     # axis and tick theme
#     ax.tick_params(axis="y", colors="steelblue")
#     ax2.tick_params(axis="y", colors="tomato")
#     ax.tick_params(size=10, color='black', labelsize=25)
#     ax2.tick_params(size=10, color='black', labelsize=25)
#     ax.tick_params(width=3)
#     ax2.tick_params(width=3)
#
#     ax = plt.gca()  # Get the current Axes instance
#
#     for axis in ['top', 'bottom', 'left', 'right']:
#         ax.spines[axis].set_linewidth(3)
#
#     plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
#     plt.show()

# markers = {1: 'o', 2: '^', 3: 's', 4: '+', 5: 'D', 6: 'p'}
# markers = {1: 'o', 2: '1', 3: 's', 4: 'D', 5: 'p', 6: '*',
#            7: 'x', 8: '$a$', 9: '$b$', 10: '$c$', 11: '$d$', 12: '$e$'}
if 'colors_' not in plt.colormaps():
    # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
    colors = [
              # (0.0, 0.0, 0.0, 0.0),
              (0.15, 0.95, 0.95, 1.0),
              (0.10, 0.10, 0.10, 1.0),
              (0.15, 0.75, 0.15, 1.0),
              (0.15, 0.15, 0.75, 1.0),
              (0.95, 0.75, 0.05, 1.0),
              (0.75, 0.15, 0.15, 1.0),
              ]

color_bin = len(np.unique(tissuelabels))
mtl.colormaps.register(LinearSegmentedColormap.from_list(name='colors_', colors=colors, N=color_bin))

# markers = ['o', '1', 's', 'D', 'p', '*', 'x', '$a$', '$b$', '$c$', '$d$', '$e$']
# markers = ['$1$', '$2$', '$3$', '$4$', '$5$', '$6$',
#            '$1$', '$2$', '$3$', '$4$', '$5$', '$6$']
markers = ['.', '.', '.', '.', 'o', 'o',#]
           'x', 'x', 'x', 'x', 'X', 'X']
# colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y',
#           'tab:orange', 'tab:pink', 'tab:olive', 'tab:brown', 'gold']
segments = [
            'gm1', 'gm2', 'gm3', 'gm4', 'gm5', 'gm6',
            'wm1', 'wm2', 'wm3', 'wm4', 'wm5', 'wm6'
    ]
colors = [
          'mediumspringgreen', 'yellowgreen', 'darkgreen', 'orange', 'darkgoldenrod', 'darkred',
          'mediumspringgreen', 'yellowgreen', 'darkgreen', 'orange', 'darkgoldenrod', 'darkred']
tissuelabels = np.array(tissuelabels)
labels = np.unique(tissuelabels)
# print(len(labels), len(colors), len(markers))
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for idf, (label, marker) in enumerate(zip(labels, markers)):
    print(label, marker)
    indices = np.where(tissuelabels == label)[0]
    ax.scatter(pcScores[indices, 0], pcScores[indices, 1], pcScores[indices, 2],
               c=colors[idf], #tissuelabels[indices],
               label=segments[idf],
               # cmap=,  # marker='o')
               marker=marker,  # 'o'
               alpha=0.5)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Score Plot GM-WM, Pareto scaled, median normalized')
# Add a legend
# for region, marker in markers.items():
#     ax.scatter([], [], [], marker='+', label=f'Region {region}')
ax.legend()
plt.show()

t_sne = TSNE(
    n_components=3,
    perplexity=30,
    init="random",
    n_iter=1250,
    random_state=0,
)
S_t_sne = t_sne.fit_transform(pcScores[:, 0:nPCs])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for idf, (label, marker) in enumerate(zip(labels, markers)):
    print(label, marker)
    indices = np.where(tissuelabels == label)[0]
    ax.scatter(S_t_sne[indices, 0], S_t_sne[indices, 1], S_t_sne[indices, 2],
               c=colors[idf], #tissuelabels[indices],
               label=segments[idf],
               # cmap=,  # marker='o')
               marker=marker,  # 'o'
               alpha=0.5)
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('C3')
ax.set_title('t-SNE')
# Add a legend
# for region, marker in markers.items():
#     ax.scatter([], [], [], marker='+', label=f'Region {region}')
ax.legend()
plt.show()