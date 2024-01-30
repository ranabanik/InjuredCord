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

wmDFlist = [wm1df, wm2df, wm3df, wm4df, wm5df, wm6df]
gmDFlist = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df]

ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1])#, len(peakPositions)])

for idf, (wm, gm) in enumerate(zip(wmDFlist, gmDFlist)):
    print(idf, len(wm['Spot index']),  len(gm['Spot index']))
    for s in range(len(wm['Spot index'])):
        # print(s, df.iloc[s, 0])
        x_ = xCor[wm.iloc[s, 0]]
        y_ = yCor[wm.iloc[s, 0]]
        ionImage3D[x_, y_] = 1
    for s in range(len(gm['Spot index'])):
        # print(s, df.iloc[s, 0])
        x_ = xCor[gm.iloc[s, 0]]
        y_ = yCor[gm.iloc[s, 0]]
        ionImage3D[x_, y_] = 2
plt.imshow(ionImage3D)
plt.show()

fig, ax = plt.subplots()

# Display the ionImage3D using imshow
im = ax.imshow(ionImage3D)

# Annotate 'Spot index' in each pixel
for s in range(processedSpectra.shape[0]):
    x_ = xCor[s]
    y_ = yCor[s]
    # spot_index = df.iloc[s]['Spot index']
    ax.text(y_, x_, str(s), ha='center', va='center', color='red', fontsize=8)

# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Colorbar Label')

plt.show()

# reg1DF = pd.read_csv(r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/region_1.csv')
# reg1DF['Spot index'] = reg1DF['Spot index'].astype(int)
# fig, ax = plt.subplots()

# Display the ionImage3D using imshow
# im = ax.imshow(ionImage3D)
#
# # Annotate 'Spot index' in each pixel
# # for s in range(len(reg1DF['Spot index'])):
# #     x_ = xCor[reg1DF.iloc[s, 0]]
# #     y_ = yCor[reg1DF.iloc[s, 0]]
# #     spot_index = reg1DF.iloc[s]['Spot index']
# #     ax.text(y_, x_, str(spot_index), ha='center', va='center', color='red', fontsize=8)
# for s in range(processedSpectra.shape[0]):
#     x_ = xCor[s]
#     y_ = yCor[s]
#     # spot_index = reg1DF.iloc[s]['Spot index']
#     ax.text(y_, x_, str(s), ha='center', va='center', color='red', fontsize=8)
# # Add a colorbar
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label('Colorbar Label')

plt.show()