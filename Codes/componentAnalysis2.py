import numpy as np
import os
import h5py
from Codes.Utilities import find_nearest
from sklearn.preprocessing import StandardScaler as SS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mtl
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.cluster import AgglomerativeClustering

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

regSpec = processedSpectra[np.where(corRegID == 1)[0]]
print(regSpec.shape)
regSpecNorm = np.zeros_like(regSpec)
refSpec = regSpec[60]
fnorm = sum(refSpec)
for s in range(regSpec.shape[0]):
    spectrum = regSpec[s]
    regSpecNorm[s] = spectrum/fnorm
    #max(spectrum)
    # normalize_spectrum(spectrum, normalize="tic")

# plt.plot(mzs, regSpecNorm[56])
# plt.show()
RandomState = 20210131
# regSpecNormSS = SS().fit_transform(regSpecNorm) # 0-mean, 1-variance

pca = PCA(random_state=RandomState, n_components=10)
pcs = pca.fit_transform(regSpecNorm)
print("pcs shape: ", pcs.shape)
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

componentMatrix = pca.components_.T  # eigenvectors or weights or components or coefficients
print(componentMatrix.shape)
componentMatrixDF = pd.DataFrame(componentMatrix[:,0:5], columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
# loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
# print(loading_matrix)
# SSL_indexing = np.argsort(loading_matrix['PC1'])[::-1]
# print(SSL_indexing)
# SSL_indexing = np.argsort(loading_matrix['PC10'])[::-1]
# print(mzs[SSL_indexing[:20]])
if plot_pca:
    MaxPCs = nPCs + 5
    fig, ax = plt.subplots(figsize=(20, 8), dpi=200)
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

# peakPositions = argrelextrema(abs(loading_matrix['PC4'].values), np.greater)[0]
#
# # Sort the peaks based on their magnitudes
# sorted_peaks_indices = sorted(peakloc, key=lambda x: abs(loading_matrix['PC4'].values)[x], reverse=True)
#
# # Take the positions of the highest 10 peaks
# highest_10_peaks = sorted_peaks_indices[:10]
sqComponentMatrix = np.square(componentMatrixDF)
ssqComponentMatrix = np.sum(sqComponentMatrix, axis=0)
print(ssqComponentMatrix)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadingsMatrixDF = pd.DataFrame(loadings[:, 0:5], columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
plt.plot(mzs, loadingsMatrixDF['PC1'].values)
plt.show()

peakPositions = argrelextrema(abs(loadingsMatrixDF['PC2'].values), np.greater)[0]
#
# # Sort the peaks based on their magnitudes
sorted_peaks_indices = sorted(peakPositions, key=lambda x: abs(loadingsMatrixDF['PC2'].values)[x], reverse=True)
#
# # Take the positions of the highest 10 peaks
highest_10_peaks = sorted_peaks_indices[:20]
# print(highest_10_peaks)
sortedpositions = sorted(highest_10_peaks)
print(mzs[sortedpositions])
mzs_values = mzs[sortedpositions]
loadings_values = 1000*loadingsMatrixDF['PC2'].values[sortedpositions]
# Create a bar plot with values
fig, ax = plt.subplots(1, 1)
ax.bar(range(len(loadings_values)), loadings_values, color='blue', alpha=0.7)
# plt.bar(mzs_values, loadings_values, color='blue', alpha=0.7)
# Add labels and title
ax.set_xlabel('Index')
ax.set_ylabel('PC1 Loadings')
ax.set_title('Bar Plot of PC1 Loadings')

# Optionally, display the values on top of each bar
for i, v in enumerate(loadings_values):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

# Show the plot
# ax.set_xticks(mzs_values)
mzs_values_rounded = [round(value, 2) for value in mzs_values]
ax.set_xticklabels(mzs_values_rounded, rotation=90, ha='right', fontsize=8)
plt.show()

pc = 'PC4'
peakPositions = argrelextrema(abs(loadingsMatrixDF[pc].values), np.greater)[0]
#
# # Sort the peaks based on their magnitudes
sorted_peaks_indices = sorted(peakPositions, key=lambda x: abs(loadingsMatrixDF[pc].values)[x], reverse=True)
#
# # Take the positions of the highest 10 peaks
highest_10_peaks = sorted_peaks_indices[:20]
# print(highest_10_peaks)
sortedpositions = sorted(highest_10_peaks)
print("mzs", mzs[sortedpositions])
mzs_values = mzs[sortedpositions]
loadings_values = 1000*loadingsMatrixDF[pc].values[sortedpositions]
# Create a bar plot with values
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(len(loadings_values)), loadings_values, color='blue', alpha=0.7)
# plt.bar(mzs_values, loadings_values, color='blue', alpha=0.7)
# Add labels and title
# ax.set_xlabel('Index')
ax.set_ylabel('PC4 loadings x 1000')
ax.set_title('Barplot of loadings')

# Optionally, display the values on top of each bar
for i, v in enumerate(loadings_values):
    if v < 0:
        ax.text(i, v-0.002, f'{v:.3f}', ha='center', va='bottom', size=5)
    else:
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', size=5)

# Show the plot
# ax.set_xticks(mzs_values)
# mzs_values_rounded = [round(value, 2) for value in mzs_values]
# ax.set_xticklabels(mzs_values_rounded, rotation=90, ha='right')
ax.set_xticks(np.arange(20), ['{:.2f}'.format(value) for value in mzs_values], rotation=90, ha='right', fontsize=8)
plt.tight_layout()
plt.show()

# +----------------+
# |  score plot    |
# +----------------+
agg = AgglomerativeClustering(n_clusters=4)
assignment = agg.fit_predict(pcs)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=assignment, cmap='viridis', marker='o', label='Data Points')

# Add labels and title
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Score Plot for PCA with Agg. clustering Labels')

# Add colorbar
cbar = plt.colorbar(scatter)
# cbar.set_label('cluster Labels')

# Show the plot
plt.legend()
plt.show()

# differential peaks
# print(100*pca.explained_variance_ratio_)
# print(max(abs(loadingsMatrixDF['PC5'])*1000))

# print(loadingsMatrixDF.values.shape)
# print(pca.explained_variance_ratio_[0:5].shape)

# print(np.prod(loadingsMatrixDF.values, pca.explained_variance_ratio_[0:5]))
SCmatrix = loadingsMatrixDF.values * pca.explained_variance_ratio_[0:5]

# print(SCmatrix.shape)

print(SCmatrix[1, 4]==(loadingsMatrixDF.values[1, 4]*pca.explained_variance_ratio_[4]))
TCmatrix = np.sum(abs(SCmatrix[:, 0:3]), axis=1)
peakPositions = argrelextrema(TCmatrix, np.greater)[0]
sorted_peaks_indices = sorted(peakPositions, key=lambda x: TCmatrix[x], reverse=True)
highest_10_peaks = sorted_peaks_indices[:50]
sortedpositions = sorted(highest_10_peaks)
mzp, TCp = mzs[sortedpositions], TCmatrix[sortedpositions]
print(mzp)
# sorted_indices = np.argsort(TCmatrix)
# highest_10_indices = sorted_indices[-10:]
# print(highest_10_indices)
plt.plot(mzs, TCmatrix)
plt.plot(mzs, regSpecNorm[60]/5, 'g')
plt.plot(mzp, TCp, 'r^', alpha=1.0, markersize=5)
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


readSegPath = os.path.join('/media/banikr/banikr/SpinalCordInjury/MALDI/230713-Chen-intactproteins/sum_argrel_base_corr_savgol_low_beta_VAE2023-08-11-11-05-44',
                           '5_label_seg.h5')
with h5py.File(readSegPath, 'r') as pfile:  # saves the data
    segImg = np.array(pfile['seg'])
plt.imshow(segImg)
plt.show()
