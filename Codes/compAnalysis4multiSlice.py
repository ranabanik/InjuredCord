import numpy as np
import os
import h5py
from Codes.Utilities import find_nearest
from sklearn.preprocessing import StandardScaler as SS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mtl
mtl.use("TKAgg")
import pandas as pd
from scipy import ndimage
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

meanSpec = np.mean(processedSpectra, axis=0)
peakPositions = argrelextrema(meanSpec, np.greater)[0]
processedSpectra = processedSpectra[:, peakPositions]
print("peak selected spectra:", processedSpectra.shape)
mzs = mzs[peakPositions]
readSegPath = os.path.join('/media/banikr/DATA/MALDI/230713-Chen-intactproteins/sum_argrel_base_corr_savgol_low_beta_VAE2023-08-11-11-05-44/',
                           '5_label_seg.h5')
with h5py.File(readSegPath, 'r') as pfile:  # saves the data
    segImg = np.array(pfile['seg'])
# plt.imshow(segImg)
# plt.show()
mask = np.zeros_like(segImg)
mask[xCor[np.where(corRegID == 1)[0]], yCor[np.where(corRegID == 1)[0]]] = 1
mask[xCor[np.where(corRegID == 2)[0]], yCor[np.where(corRegID == 2)[0]]] = 1
mask[xCor[np.where(corRegID == 3)[0]], yCor[np.where(corRegID == 3)[0]]] = 1
mask[xCor[np.where(corRegID == 4)[0]], yCor[np.where(corRegID == 4)[0]]] = 1
mask[xCor[np.where(corRegID == 5)[0]], yCor[np.where(corRegID == 5)[0]]] = 1
# mask[xCor[np.where(corRegID == 6)[0]], yCor[np.where(corRegID == 6)[0]]] = 1

# Set all pixels in segImg to 0 except those specified by the mask
segImg = segImg * mask
plt.imshow(segImg == 3)
plt.colorbar()
plt.show()

# x, y = np.where(segImg==3)
# print(x, '\n', y)
grayPixels = []
labelReg = []
for i, (x_, y_) in enumerate(zip(xCor, yCor)):
    if segImg[x_, y_] == 3:
        grayPixels.append(i)
        # labelReg.append([segImg[x_,y_]])
print(grayPixels)
print(corRegID[grayPixels])
# print(corRegID)
print(processedSpectra[grayPixels].shape)
regSpec = processedSpectra[grayPixels]
regSpecNorm = np.zeros_like(regSpec)
maxInt = np.max(regSpec)
nmz = regSpec.shape[1]
# refSpec = regSpec[60]
# fnorm = sum(refSpec)
for s in range(regSpec.shape[0]):
    spectrum = regSpec[s]
    regSpecNorm[s] = spectrum/np.median(spectrum)

# # Level scaling
# col_means = np.mean(regSpecNorm, axis=0)
# print(col_means.shape)
# # Divide each column by its mean
# regSpecLeveled = regSpec / col_means

# Pareto scaling
col_means = np.mean(regSpecNorm, axis=0)
col_stddevs = np.std(regSpecNorm, axis=0)

# Pareto scaling
regSpecPareto = (regSpecNorm - col_means) / np.sqrt(col_stddevs)
# +----------------+
# |      pca       |
# +----------------+
RandomState = 20210131
pca = PCA(random_state=RandomState) #, n_components=10)
# regSpecNormSS = SS().fit_transform(regSpecNorm) # 0-mean, 1-variance
pcScores = pca.fit_transform(regSpecPareto)
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

componentMatrix = pca.components_.T  # eigenvectors or weights or components or coefficients
print(componentMatrix.shape)
componentMatrixDF = pd.DataFrame(componentMatrix[:, 0:nPCs], columns=['PC%02d' % (i + 1) for i in range(nPCs)])
# print(componentMatrixDF)
loadings = -1*pca.components_.T * np.sqrt(pca.explained_variance_)
loadingsMatrixDF = pd.DataFrame(loadings[:, 0:nPCs], columns=['PC%02d' % (i + 1) for i in range(nPCs)])
loadingsMatrixDF.insert(loadingsMatrixDF.shape[1], column='mzs', value=mzs)
loadingsMatrixDF.loc[len(loadingsMatrixDF)] = pca.explained_variance_ratio_[0:nPCs+1]
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

scatter = ax.scatter(pcScores[:, 0], pcScores[:, 1], pcScores[:, 2],
                     c=corRegID[grayPixels],
                     cmap='viridis', marker='o')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Score Plot for PCA with Agg. clustering Labels')
cbar = plt.colorbar(scatter)
# cbar.set_label('cluster Labels')
plt.legend()
plt.show()

plt.plot(mzs, loadingsMatrixDF['PC01'].values[0:-1], label='loading 1')
plt.plot(mzs, loadingsMatrixDF['PC02'].values[0:-1], label='loading 2')
plt.plot(mzs, loadingsMatrixDF['PC03'].values[0:-1], label='loading 3')
plt.plot(mzs, regSpec[40], label='sample spec', alpha=0.4)
plt.legend()
plt.show()

SCmatrix = loadingsMatrixDF.iloc[:-1, :-1]* pca.explained_variance_ratio_#[0:6]
print(SCmatrix[1, 4] == (loadingsMatrixDF.values[1, 4]*pca.explained_variance_ratio_[4]))
TCmatrix = np.sum(abs(SCmatrix[:, 0:6]), axis=1)
peakPositions = argrelextrema(TCmatrix, np.greater)[0]
sorted_peaks_indices = sorted(peakPositions, key=lambda x: TCmatrix[x], reverse=True)
highest_10_peaks = sorted_peaks_indices[:50]
sortedpositions = sorted(highest_10_peaks)
mzp, TCp = mzs[sortedpositions], TCmatrix[sortedpositions]
plt.plot(mzs, TCmatrix)
plt.plot(mzs, regSpecNorm[40]/5, 'g')
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

if __name__ != '__main__':
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    group5 = []
    for gp in grayPixels:
        if corRegID[gp] == 1:
            group1.append(gp)
        if corRegID[gp] == 2:
            group2.append(gp)
        if corRegID[gp] == 3:
            group3.append(gp)
        if corRegID[gp] == 4:
            group4.append(gp)
        if corRegID[gp] == 5:
            group5.append(gp)

    print(group1, '\n', group2, '\n', group3, '\n', group4, '\n', group5)
    group1GM = processedSpectra[group1]
    group2GM = processedSpectra[group2]
    group3GM = processedSpectra[group3]
    group4GM = processedSpectra[group4]
    group5GM = processedSpectra[group5]
    print(group2GM.shape)

    from scipy.stats import f_oneway
    data_groups = [group2GM, group1GM, group1GM, group1GM, group2GM]#, group3GM, group4GM, group5GM]
    p_values = []
    for feature_index in range(data_groups[0].shape[1]):
        feature_data = [group[:, feature_index] for group in data_groups]
        _, p_value = f_oneway(*feature_data)
        p_values.append(p_value)
        # break
    # Example: Count the number of features with significant differences (assuming a significance level of 0.05)
    num_significant = np.sum(np.array(p_values) < 0.05)
    print(f"Number of features with significant differences: {num_significant}")