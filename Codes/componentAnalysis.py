import sys
sys.path.append(r'/home/banikr/PycharmProjects/spinalcordinjury/')
import os
import numpy as np
from glob import glob
from Codes.Utilities import chart, ImzmlAll, _smooth_spectrum, umap_it, hdbscan_it
import peakutils
from scipy import ndimage
import matplotlib.pyplot as plt
import h5py
import matplotlib as mtl
# mtl.use("TKAgg")
from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler as SS

wl_ = 5
po_ = 2
deg_ = 6
max_it_ = 1000
tol_ = 1e-5

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'
dataPath = os.path.join(fileDir, '230713-Chen-intactproteins.imzML')
print(dataPath)
with h5py.File(dataPath, 'r') as pfile:
    print(pfile.keys())
    spectra = np.array(pfile.get('spectra'))
    mzs = np.array(pfile.get('mzs'))
    xloc = np.array(pfile.get('xloc'))
    yloc = np.array(pfile.get('yloc'))
print(xloc)
print(yloc)
max(xloc), max(yloc)
img = np.zeros((max(xloc) + 1, max(yloc) + 1))
# img.shape
for sIdx, (x, y) in enumerate(zip(xloc, yloc)):
    img[x, y] = 1
    # print(sIdx)
labeledArr, num_ids = ndimage.label(img, structure=np.ones((3, 3)))
# plt.imshow(labeledArr)
# plt.colorbar()
# plt.show()
regID = 1
regionSpectra = []
for sIdx, (x, y) in enumerate(zip(xloc, yloc)):
    if labeledArr[x, y] == regID:
        regionSpectra.append(spectra[sIdx])

print(">> ", len(regionSpectra))
regionSpectra = np.array(regionSpectra).squeeze()
print(regionSpectra.shape)
regionSpectra_ss = SS().fit_transform(regionSpectra)
data_umap = umap_it(regionSpectra_ss)
# u_comp = data_umap.shape[1]
# df_umap = pd.DataFrame(data=data_umap[:, 0:u_comp], columns=['umap_%d' % (i + 1) for i in range(u_comp)])
# df_umap.to_csv(savecsv, index=False, sep=',')

labels = hdbscan_it(data_umap, min_cluster_size=20, min_samples=10)
# plt.show()

chart(data_umap, labels)
print("UMAP-HDBSCAN found {} labels in the data".format(len(np.unique(labels))))

