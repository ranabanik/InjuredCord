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

path1 = os.path.join(fileDir, 'loadings1regions.csv')
path2 = os.path.join(fileDir, 'loadings2regions.csv')
path3 = os.path.join(fileDir, 'loadings3regions.csv')
path4 = os.path.join(fileDir, 'loadings4regions.csv')
path5 = os.path.join(fileDir, 'loadings5regions.csv')

loadingsDF1 = pd.read_csv(path1)
loadingsDF2 = pd.read_csv(path2)
loadingsDF3 = pd.read_csv(path3)
loadingsDF4 = pd.read_csv(path4)
loadingsDF5 = pd.read_csv(path5)

mzs = loadingsDF5.iloc[:-1, -1].values

# print(loadingsDF1)
explainedVarianceRatio1 = loadingsDF1.iloc[-1].values[0:-1]
explainedVarianceRatio2 = loadingsDF2.iloc[-1].values[0:-1]
explainedVarianceRatio3 = loadingsDF3.iloc[-1].values[0:-1]
explainedVarianceRatio4 = loadingsDF4.iloc[-1].values[0:-1]
explainedVarianceRatio5 = loadingsDF5.iloc[-1].values[0:-1]

loadingsMatrix1 = loadingsDF1.iloc[:-1, :-1]
loadingsMatrix2 = loadingsDF2.iloc[:-1, :-1]
loadingsMatrix3 = loadingsDF3.iloc[:-1, :-1]
loadingsMatrix4 = loadingsDF4.iloc[:-1, :-1]
loadingsMatrix5 = loadingsDF5.iloc[:-1, :-1]

print(loadingsMatrix1.shape, explainedVarianceRatio1.shape)
SCmatrix1 = loadingsMatrix1 * explainedVarianceRatio1
SCmatrix2 = loadingsMatrix2 * explainedVarianceRatio2
SCmatrix3 = loadingsMatrix3 * explainedVarianceRatio3
SCmatrix4 = loadingsMatrix4 * explainedVarianceRatio4
SCmatrix5 = loadingsMatrix5 * explainedVarianceRatio5

TCvector1 = np.sum(abs(SCmatrix1), axis=1)
TCvector2 = np.sum(abs(SCmatrix2), axis=1)
TCvector3 = np.sum(abs(SCmatrix3), axis=1)
TCvector4 = np.sum(abs(SCmatrix4), axis=1)
TCvector5 = np.sum(abs(SCmatrix5), axis=1)

print(TCvector5.shape, TCvector4.shape, TCvector3.shape, TCvector2.shape, TCvector1.shape)

plt.plot(mzs, TCvector1, alpha=0.2, label='TC1')
plt.plot(mzs, TCvector2, alpha=0.4, label='TC2')
plt.plot(mzs, TCvector3, alpha=0.6, label='TC3')
plt.plot(mzs, TCvector4, alpha=0.8, label='TC4')
plt.plot(mzs, TCvector5, '.k', linewidth=0.3, alpha=1.0, label='TC5')
plt.legend()
plt.ylabel("total contribution score to 95% variance in data", fontsize=12)
plt.xlabel("m/z", fontsize=12)
plt.show()

from scipy.stats import f_oneway

data_groups = np.array([TCvector1, TCvector2, TCvector3, TCvector4, TCvector5])  # , group3GM, group4GM, group5GM]
print(data_groups.shape)
result = f_oneway(*data_groups)

# Print the ANOVA result
print("ANOVA F-statistic:", result.statistic)
print("ANOVA p-value:", result.pvalue)
# p_values = []
# for feature_index in range(data_groups[0].shape[1]):
#     feature_data = [group[:, feature_index] for group in data_groups]
#     _, p_value = f_oneway(*feature_data)
#     p_values.append(p_value)
#     # break
# # Example: Count the number of features with significant differences (assuming a significance level of 0.05)
# num_significant = np.sum(np.array(p_values) < 0.05)
# print(f"Number of features with significant differences: {num_significant}")