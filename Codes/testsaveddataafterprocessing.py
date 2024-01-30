import sys
sys.path.append(r'/home/banikr/PycharmProjects/spinalcordinjury/')
import os
import numpy as np
from glob import glob
from Codes.Utilities import chart, ImzmlAll, _smooth_spectrum, umap_it, hdbscan_it, normalize_spectrum
import peakutils
from scipy import ndimage
import matplotlib.pyplot as plt
import h5py
import matplotlib as mtl
# mtl.use("TKAgg")
from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler as SS
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'

savepath = os.path.join(fileDir, 'SavGolBaseCorrXlocYlocRegID.h5')
print(savepath)
with h5py.File(savepath, 'r') as pfile:
    print(pfile.keys())
    processedSpectra = np.array(pfile['processedspectra'])
    mzs = np.array(pfile['rawmzs'])
    xCor = np.array(pfile['xCor'])
    yCor = np.array(pfile['yCor'])#yCor[0].dtype)
    corRegID = np.array(pfile['corRegID'])# = np.array(regList, dtype=int)#regList[0].dtype)

# print(processedSpectra.shape)
# print(np.unique(corRegID))
# print(np.where(corRegID == 6)[0])
#
# img = np.zeros((max(xCor) + 1, max(yCor) + 1))
# for cor, x, y in zip(corRegID, xCor, yCor):
#     img[x, y] = cor
# plt.imshow(img)
# plt.colorbar()
# plt.show()

regSpec = processedSpectra[np.where(corRegID == 1)[0]]
regSpecNorm = np.zeros_like(regSpec)
for s in range(regSpec.shape[0]):
    spectrum = regSpec[s]
    regSpecNorm[s] = normalize_spectrum(spectrum, normalize="tic")

# plt.plot(mzs, regSpecNorm[34])
# plt.show()



