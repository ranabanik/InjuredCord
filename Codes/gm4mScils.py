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
from Codes.Utilities import find_nearest
from scipy.signal import argrelextrema

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/GM_injured_cord/'

fileCSVs = glob(os.path.join(fileDir, '*.csv'))
print(fileCSVs)

df1 = pd.read_csv(fileCSVs[0])
mzs = np.array(df1['m/z'].values)
df1 = df1.drop(columns=['m/z'])
df2 = pd.read_csv(fileCSVs[1]).drop(columns=['m/z'])
df3 = pd.read_csv(fileCSVs[2]).drop(columns=['m/z'])
df4 = pd.read_csv(fileCSVs[3]).drop(columns=['m/z'])
df5 = pd.read_csv(fileCSVs[4]).drop(columns=['m/z'])
df6 = pd.read_csv(fileCSVs[5]).drop(columns=['m/z'])

# print(gm1spectraDF['m/z'].values)
# print("this", len(gm1spectraDF.columns))
# print(gm1spectraDF[' Spot 820'].values)
#
# fileDir1 = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'
#
# savepath = os.path.join(fileDir1, 'SavGolwl55po2BaseCorrdeg6max1000tol5XlocYlocRegID.h5')
# print(savepath)
# with h5py.File(savepath, 'r') as pfile:
#     print(pfile.keys())
#     processedSpectra = np.array(pfile['processedspectra'])
#     mzs = np.array(pfile['rawmzs'])
#     xCor = np.array(pfile['xCor'])
#     yCor = np.array(pfile['yCor'])
#     corRegID = np.array(pfile['corRegID'])
#
# # print("this", len(np.where(corRegID==1)[0]))
# spectrum = processedSpectra[819]
# spectrum = spectrum/np.median(spectrum)

# plt.plot(mzs[1:-1], spectrum[1:-1], 'r')
# plt.plot(gm1spectraDF['m/z'].values, gm1spectraDF[' Spot 820'].values, 'b')
# plt.show()
# print(df6)
dfs = [df1, df2, df3, df4, df5, df6]
# # Check if the first column values are equal
# first_column_values_equal = all(dfs[i]['m/z'].equals(dfs[i-1]['m/z']) for i in range(1, len(dfs)))
#
# # Display the result
# if first_column_values_equal:
#     print("The first column values are equal in all DataFrames.")
# else:
#     print("The first column values are not equal in all DataFrames.")

# Concatenate DataFrames along the columns (axis=1)
result_df = pd.concat(dfs, axis=1)
print(result_df)

# print(pd.Series(range(1, 7), index=result_df.columns), ignore_index=True)
# result_df = result_df.append(pd.Series(range(1, 7), index=result_df.columns), ignore_index=True)
# print(result_df)

# Create a list indicating the source DataFrame for each column
regID = [i for i in range(1, len(dfs) + 1) for _ in range(len(dfs[i-1].columns))]

# print("\nSource List:", len(source_list))
# print(source_list)

spectra = result_df.values.T
print(spectra.shape)
# Find rows where all values are zeroes
# non_zero_rows = np.any(spectra != 0, axis=0)
# print(">> ", non_zero_rows)
columns_to_remove = np.where(np.sum(spectra, axis=0) == 0)[0]
spectra = np.delete(spectra, columns_to_remove, axis=1)
print(spectra.shape)
mzs = np.delete(mzs, columns_to_remove, axis=0)
# Pareto scaling
col_means = np.mean(spectra, axis=0)
col_stddevs = np.std(spectra, axis=0)

# Pareto scaling
spectraPareto = (spectra - col_means) / np.sqrt(col_stddevs)

# +----------------+
# |      pca       |
# +----------------+
RandomState = 20210131
pca = PCA(random_state=RandomState) #, n_components=10)
# regSpecNormSS = SS().fit_transform(regSpecNorm) # 0-mean, 1-variance
pcScores = pca.fit_transform(spectraPareto)
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
if 'msml_list' not in plt.colormaps():
    # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
    colors = [(.10, .10, .10, 1.0),  # bg
              (0.75, 0.15, 0.15, 1.0),  # wm1
              (0.15, 0.75, 0.15, 1.0),  # wm2
              (0.15, 0.15, 0.75, 1.0),  # gm1
              (0.95, 0.75, 0.05, 1.0),  # blood/injury
              (0.15, 0.95, 0.95, 1.0),  # conn tissue
              # (0.9569, 0.7686, 0.1882),
              # (0.65, 0.71, 0.91),
              # (0.95, 0.3, 0.22),  # (0.0, 0.5, 0.5),
              # (0.01, 0.01, 0.95, 1.0),
              # (0.01, 0.95, 0.01, 1.0),  # (0.5, 0.5, 0.0),
              # (0.65, 0.71, 0.81),
              # (0.95, 0.01, 0.01, 1.0)
              ]
color_bin = 6
mtl.colormaps.register(LinearSegmentedColormap.from_list(name='colors', colors=colors, N=color_bin))

scatter = ax.scatter(pcScores[:, 0], pcScores[:, 1], pcScores[:, 2],
                     c=regID,
                     cmap='colors', #marker='o')
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
plt.plot(mzs, spectra[40]/5, label='sample spec', alpha=0.4)
plt.legend()
plt.show()

SCmatrix = loadingsMatrixDF.values * pca.explained_variance_ratio_[0:nPCs]
print(SCmatrix[1, 4] == (loadingsMatrixDF.values[1, 4]*pca.explained_variance_ratio_[4]))
TCmatrix = np.sum(abs(SCmatrix[:, 0:10]), axis=1)
peakPositions = argrelextrema(TCmatrix, np.greater)[0]
sorted_peaks_indices = sorted(peakPositions, key=lambda x: TCmatrix[x], reverse=True)
highest_10_peaks = sorted_peaks_indices[:50]
sortedpositions = sorted(highest_10_peaks)
mzp, TCp = mzs[sortedpositions], TCmatrix[sortedpositions]
plt.plot(mzs, TCmatrix)
plt.plot(mzs, spectraPareto[40]/5, 'g')
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