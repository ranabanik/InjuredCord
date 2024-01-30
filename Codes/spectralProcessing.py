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
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm

wl_ = 55
po_ = 2
deg_ = 6
max_it_ = 1000
tol_ = 1e-5

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'
dataPath = os.path.join(fileDir, '230713-Chen-intactproteins.imzML')
print(dataPath)

ImzObj = ImzMLParser(dataPath)

wl_ = 55
po_ = 2
deg_ = 6
max_it_ = 1000
tol_ = 1e-5

# print(ImzObj.coordinates)
maxX = ImzObj.imzmldict['max count of pixels x']
maxY = ImzObj.imzmldict['max count of pixels y']
img = np.zeros((maxX + 1, maxY + 1))

for sIdx, (x, y, z) in enumerate(ImzObj.coordinates):
    img[x, y] = 1
labeledArr, num_ids = ndimage.label(img, structure=np.ones((3, 3)))
# plt.imshow(labeledArr)
# plt.colorbar()
# plt.show()
processedSpectra = []
xCor = []
yCor = []
regList = []
for sIdx, (x, y, z) in enumerate(tqdm(ImzObj.coordinates)):
    mzs, signal_ = ImzObj.getspectrum(sIdx)
    smoothedSpec = _smooth_spectrum(signal_, method='savgol', window_length=wl_, polyorder=po_)
    bl = peakutils.baseline(smoothedSpec, deg=deg_, max_it=max_it_, tol=tol_)
    smoothedBaseCorrSpec = smoothedSpec - bl
    processedSpectra.append(smoothedBaseCorrSpec)
    xCor.append(x)
    yCor.append(y)
    regList.append(labeledArr[x, y])
    # if sIdx == 10:
    #     break
ntolpower=5     # power of tolerance in -ve
savepath = os.path.join(fileDir, 'SavGolwl{}po{}BaseCorrdeg{}max{}tol{}XlocYlocRegID.h5'.format(wl_, po_, deg_,
                                                                                                max_it_, ntolpower))
print(savepath)
with h5py.File(savepath, 'w') as pfile:
    pfile['processedspectra'] = np.array(processedSpectra, dtype=signal_[0].dtype)
    pfile['rawmzs'] = np.array(mzs, dtype=mzs[0].dtype)
    pfile['xCor'] = np.array(xCor, dtype=int)
    pfile['yCor'] = np.array(yCor, dtype=int)   #yCor[0].dtype)
    pfile['corRegID'] = np.array(regList, dtype=int)    #regList[0].dtype)


