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

gmfileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/GM_injured_cord/'

csv1 = glob(os.path.join(gmfileDir, '*1*spots.csv'))
csv2 = glob(os.path.join(gmfileDir, '*2*spots.csv'))
csv3 = glob(os.path.join(gmfileDir, '*3*spots.csv'))
csv4 = glob(os.path.join(gmfileDir, '*4*spots.csv'))
csv5 = glob(os.path.join(gmfileDir, '*5*spots.csv'))
csv6 = glob(os.path.join(gmfileDir, '*6*spots.csv'))
print(csv1)
gm1df = pd.read_csv(csv1[0])
gm2df = pd.read_csv(csv2[0])
gm3df = pd.read_csv(csv3[0])
gm4df = pd.read_csv(csv4[0])
gm5df = pd.read_csv(csv5[0])
gm6df = pd.read_csv(csv6[0])
# print(gm3df)
wmfileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/WM_injured_cord'
wmcsvList = glob(os.path.join(wmfileDir, '*.csv'))

for l, dfl in enumerate(wmcsvList):
    print(l, dfl)

wm6wholedf = pd.read_csv(wmcsvList[8])
print(wm6wholedf)
# merged_df = pd.merge(wm3wholedf, gm3df, on='Spot index', how='left', indicator=True)
# df_result = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)
df_result = wm6wholedf[~wm6wholedf['Spot index'].isin(gm6df['Spot index'])]
print(len(df_result['Spot index']))


# print(np.where(df_result['Spot index'] == 417)[0][0])

row_to_move = df_result.iloc[np.where(df_result['Spot index'] == 417)[0][0]].copy()
df_result = df_result.drop(np.where(df_result['Spot index'] == 417)[0][0])
gm6df = gm6df.append(row_to_move, ignore_index=True)
gm6df['Spot index'] = gm6df['Spot index'].astype(int)
print(len(df_result['Spot index']))
print("then")
print(len(gm6df['Spot index']))
ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1])#, len(peakPositions)])
for idf, df in enumerate([gm6df, df_result]):#, wm3wholedf]:
    print(len(df['Spot index']))
    for s in range(len(df['Spot index'])):
        # print(s, df.iloc[s, 0])
        x_ = xCor[df.iloc[s, 0]]
        y_ = yCor[df.iloc[s, 0]]
        ionImage3D[x_, y_] = idf + 1

plt.imshow(ionImage3D)
plt.show()

savecsv = os.path.join(wmfileDir, 'WM-6_spots.csv')
df_result.to_csv(savecsv, index=False, sep=',')

fig, ax = plt.subplots()

# Display the ionImage3D using imshow
im = ax.imshow(ionImage3D)

# Annotate 'Spot index' in each pixel
for s in range(len(df['Spot index'])):
    x_ = xCor[df.iloc[s, 0]]
    y_ = yCor[df.iloc[s, 0]]
    spot_index = df.iloc[s]['Spot index']
    ax.text(y_, x_, str(spot_index), ha='center', va='center', color='red', fontsize=8)

# Add a colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Colorbar Label')

plt.show()