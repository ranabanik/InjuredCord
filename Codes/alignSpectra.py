# 01.31.2024
import os
from glob import glob
import h5py
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
from msalign import msalign
import matplotlib.pyplot as plt
import matplotlib as mtl
mtl.use("TKAgg")

def overlay_plot(ax, x, array, peak):
    """Generate overlay plot, showing each signal and the alignment peak(s)"""
    for i, y in enumerate(array):
        y = array[i]
        print(i, y.shape)
        y = (y / y.max()) + (i * 0.2)
        ax.plot(x, y, lw=3)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_xlabel("Index", fontsize=18)
    ax.set_xlim((x[0], x[-1]))
    ax.vlines(peak, *ax.get_ylim())

# fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'
fileDir = r'C:\Data\MALDI\230713-Chen-intactproteins'

h5Path = os.path.join(fileDir, 'SavGolwl55po2BaseCorrdeg6max1000tol5XlocYlocRegID.h5')

with h5py.File(h5Path, 'r') as pfile:
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

h5Path = os.path.join(fileDir, 'alignedSpectra.h5')
with h5py.File(h5Path, 'r') as pfile:
    print(pfile.keys())
    alignedSpectra = np.array(pfile['spectra'])

# gmcsvDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/GM_injured_cord1/'
gmcsvDir = os.path.join(fileDir, 'GM_injured_cord1')
# wmcsvDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/WM_injured_cord1/'
wmcsvDir = os.path.join(fileDir, 'WM_injured_cord1')

gm1csv = glob(os.path.join(gmcsvDir, '*1*spots.csv'))
gm2csv = glob(os.path.join(gmcsvDir, '*2*spots.csv'))
gm3csv = glob(os.path.join(gmcsvDir, '*3*spots.csv'))
gm4csv = glob(os.path.join(gmcsvDir, '*4*spots.csv'))
gm5csv = glob(os.path.join(gmcsvDir, '*5*spots.csv'))
gm6csv = glob(os.path.join(gmcsvDir, '*6*spots.csv'))

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

wm1df = pd.read_csv(wm1csv[0])
wm2df = pd.read_csv(wm2csv[0])
wm3df = pd.read_csv(wm3csv[0])
wm4df = pd.read_csv(wm4csv[0])
wm5df = pd.read_csv(wm5csv[0])
wm6df = pd.read_csv(wm6csv[0])

dflist = [
    gm1df, gm2df, gm3df, gm4df, gm5df, gm6df,
    wm1df, wm2df, wm3df, wm4df, wm5df, wm6df
         ]
savecsv = os.path.join(fileDir, 'tic_poisson_pca90_umap_hdbscan_gmwm_labels.csv')
labelDF = pd.read_csv(savecsv)
labels = labelDF['labels'].values #- 1

print(np.unique(labels), len(labels))

specIdx = []
for df in dflist:
    specIdx.extend(df['Spot index'].values)
print(len(specIdx))

spectra = processedSpectra[specIdx]
peaks = mzs[peakPositions]
array = spectra#[10:20]
# print("array.shape: ", array.shape[0], type(array), "\\", array)
# x = spectra[0]
alignedSpectra = msalign(peakmzs, array, peaks,
                         # return_shifts=True,
                         align_by_index=True,
                         only_shift=False, method="pchip")
print("wtf", alignedSpectra.shape)

savepath = os.path.join(fileDir, 'alignedSpectra.h5')
print(savepath)
# with h5py.File(savepath, 'w') as pfile:
#     pfile['spectra'] = np.array(alignedSpectra, dtype=spectra[0][0].dtype)

# aligner = Aligner(
#     peakmzs,
#     array,
#     peaks,
#     # weights=None,
#     return_shifts=True,
#     align_by_index=True,
#     only_shift=True,
#     method="pchip",
# )
# aligner.run()
# aligned_array = aligner.align()

# display before and after shifting
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
overlay_plot(ax[0], peakmzs, array, peaks)
overlay_plot(ax[1], peakmzs, alignedSpectra, peaks)
plt.show()