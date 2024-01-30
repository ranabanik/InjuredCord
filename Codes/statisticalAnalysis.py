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
from scipy.stats import ttest_ind

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

hgmdflist = [gm1df, gm2df, gm3df, gm4df]#, gm5df, gm6df]
igmdflist = [gm5df, gm6df]

hgmspecIdx = []
for gmdf in hgmdflist:
    hgmspecIdx.extend(gmdf['Spot index'].values)
igmspecIdx = []
for gmdf in igmdflist:
    igmspecIdx.extend(gmdf['Spot index'].values)

hgmSpectra = processedSpectra[hgmspecIdx]
igmSpectra = processedSpectra[igmspecIdx]
print(hgmSpectra.shape, igmSpectra.shape)

# hgmSpecMean = np.mean(hgmSpectra, axis=0)
# igmSpecMean = np.mean(igmSpectra, axis=0)
# print(hgmSpecMean.shape, igmSpecMean.shape)

print(np.where(mzs == find_nearest(mzs, 14130))[0])

rng = np.random.default_rng()
# stats.ttest_ind(rvs1, rvs5, permutations=10000,
#                 random_state=rng)

p_values = []
for feature_index in range(hgmSpectra.shape[1]):
    _, p_value = ttest_ind(hgmSpectra[:, feature_index],
                           igmSpectra[:, feature_index],
                           permutations=100,
                           equal_var=False,
                           random_state=rng)
    p_values.append(p_value)

# Adjust p-values for multiple comparisons
# For example, using Bonferroni correction:
alpha = 0.05
adjusted_p_values = np.array(p_values) * hgmSpectra.shape[1]

# Identify significant features
significant_features = np.where(adjusted_p_values < alpha)[0]

print("Number of significant features:", len(significant_features))
print("Indices of significant features:", significant_features)

gmdflist = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df]

gmspecIdx = []
for gmdf in gmdflist:
    gmspecIdx.extend(gmdf['Spot index'].values)

significantSpectra = processedSpectra[:, significant_features]
mzs = mzs[significant_features]
spectra = significantSpectra[gmspecIdx]
print(spectra.shape)
regID = corRegID[gmspecIdx]
regSpecNorm = np.zeros_like(spectra)

wl_ = 3
po_ = 1
for s in range(regSpecNorm.shape[0]):
    spectrum = spectra[s]
    # spectrum = _smooth_spectrum(spectrum, method='savgol', window_length=wl_, polyorder=po_)
    regSpecNorm[s] = spectrum#/np.median(spectrum)
print('this')
# # Level scaling
col_means = np.mean(regSpecNorm, axis=0)
# print(col_means.shape)
# # Divide each column by its mean
# regSpecScaled = regSpecNorm / col_means
# # Pareto scaling
col_stddevs = np.std(regSpecNorm, axis=0)
#
# # Pareto scaling
regSpecScaled = (regSpecNorm - col_means) / np.sqrt(col_stddevs)
# +----------------+
# |      pca       |
# +----------------+
RandomState = 20210131
pca = PCA(random_state=RandomState) #, n_components=10)
# regSpecNormSS = SS().fit_transform(spectra)     # 0-mean, 1-variance
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
ax.set_xlabel('Principal component number', fontsize=30)
ax.set_ylabel('Percentage of \n variance explained', fontsize=30)
ax.set_ylim([-0.5, 100])
ax.set_xlim([-0.5, MaxPCs])
ax.grid("on")

ax2 = ax.twinx()
ax2.plot(pca_range[0:MaxPCs], evr_cumsum[0:MaxPCs] * 100, color="tomato", marker="D", ms=7)
ax2.scatter(nPCs, cut_evr * 100, marker='*', s=500, facecolor='blue')
ax2.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
ax2.set_ylabel('Cumulative percentage', fontsize=30)
ax2.set_ylim([-0.5, 100])

# axis and tick theme
ax.tick_params(axis="y", colors="steelblue")
ax2.tick_params(axis="y", colors="tomato")
ax.tick_params(size=10, color='black', labelsize=25)
ax2.tick_params(size=10, color='black', labelsize=25)
ax.tick_params(width=3)
ax2.tick_params(width=3)

ax = plt.gca()  # Get the current Axes instance

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(3)

plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
plt.show()
componentMatrix = pca.components_.T  # eigenvectors or weights or components or coefficients
print(componentMatrix.shape)
componentMatrixDF = pd.DataFrame(componentMatrix[:, 0:nPCs], columns=['PC%02d' % (i + 1) for i in range(nPCs)])
# print(componentMatrixDF)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
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
color_bin = len(gmdflist)
mtl.colormaps.register(LinearSegmentedColormap.from_list(name='colors_', colors=colors, N=color_bin))

scatter = ax.scatter(pcScores[:, 0], pcScores[:, 1], pcScores[:, 2],
                     c=regID,
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

plt.plot(mzs, loadingsMatrixDF['PC01'].values, label='loading 1')
plt.plot(mzs, loadingsMatrixDF['PC02'].values, label='loading 2')
plt.plot(mzs, loadingsMatrixDF['PC03'].values, label='loading 3')
plt.plot(mzs, (regSpecNorm[40]-min(regSpecNorm[40]))/(max(regSpecNorm[40])-min(regSpecNorm[40])), label='sample spec', alpha=0.4)
plt.legend()
plt.show()
topPeaks = 50
SCmatrix = loadingsMatrixDF.values * pca.explained_variance_ratio_[0:nPCs]
print(SCmatrix[1, 4] == (loadingsMatrixDF.values[1, 4]*pca.explained_variance_ratio_[4]))
TCmatrix = np.sum(abs(SCmatrix[:, 0:topPeaks]), axis=1)
peakPositions = argrelextrema(TCmatrix, np.greater)[0]
sorted_peaks_indices = sorted(peakPositions, key=lambda x: TCmatrix[x], reverse=True)
highest_10_peaks = sorted_peaks_indices[:topPeaks]
sortedpositions = sorted(highest_10_peaks)
mzp, TCp = mzs[sortedpositions], TCmatrix[sortedpositions]
plt.plot(mzs, TCmatrix, label='total contribution')
plt.plot(mzs, regSpecNorm[40]/5, 'g', label='sample spectrum')
plt.plot(mzp, TCp, 'r^', alpha=1.0, markersize=5)
plt.legend()
plt.show()

TCsortedLoadings = loadingsMatrixDF.values[sortedpositions]
labels = mzs[sortedpositions]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(TCsortedLoadings[:, 0],
                     TCsortedLoadings[:, 1],
                     TCsortedLoadings[:, 2],
                     cmap='viridis', marker='o')

# Add labels and title
ax.set_xlabel('LPC1')
ax.set_ylabel('LPC2')
ax.set_zlabel('LPC3')
ax.set_title('3D loading plot')
for i, txt in enumerate(labels):
    ax.text(TCsortedLoadings[i, 0], TCsortedLoadings[i, 1], TCsortedLoadings[i, 2], '{:.2f}'.format(txt), color='red')
# Add colorbar
# cbar = plt.colorbar(scatter)
# cbar.set_label('cluster Labels')
# Add center axes
# Add center axes (positive and negative sides)
ax.quiver(0, 0, 0, 1, 0, 0, color='red', arrow_length_ratio=0.1, label='')
ax.quiver(0, 0, 0, -1, 0, 0, color='red', arrow_length_ratio=0.1)  # Negative X-axis
ax.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1, label='')
ax.quiver(0, 0, 0, 0, -1, 0, color='green', arrow_length_ratio=0.1)  # Negative Y-axis
ax.quiver(0, 0, 0, 0, 0, 1, color='blue', arrow_length_ratio=0.1, label='')
ax.quiver(0, 0, 0, 0, 0, -1, color='blue', arrow_length_ratio=0.1)  # Negative Z-axis

# Show the plot
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(len(TCp)), TCp, color='blue', alpha=0.7)
# plt.bar(mzs_values, loadings_values, color='blue', alpha=0.7)
# Add labels and title
# ax.set_xlabel('Index')
ax.set_ylabel('Total contribution')
ax.set_title('Barplot of loadings')

# Optionally, display the values on top of each bar
for i, v in enumerate(TCp):
    if v < 0:
        ax.text(i, v-0.002, f'{v:.3f}', ha='center', va='bottom', size=5)
    else:
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', size=5)

# Show the plot
# ax.set_xticks(mzs_values)
# mzs_values_rounded = [round(value, 2) for value in mzs_values]
# ax.set_xticklabels(mzs_values_rounded, rotation=90, ha='right')
ax.set_xticks(np.arange(topPeaks), ['{:.2f}'.format(value) for value in mzp], rotation=90, ha='right', fontsize=8)
plt.tight_layout()
plt.show()
