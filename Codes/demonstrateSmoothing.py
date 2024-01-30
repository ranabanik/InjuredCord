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
mtl.use("TKAgg")
from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler as SS
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'
dataPath = os.path.join(fileDir, '230713-Chen-intactproteins.imzML')
print(dataPath)

ImzObj = ImzMLParser(dataPath)

wl_ = 57
po_ = 5
deg_ = 6
max_it_ = 1000
tol_ = 1e-5
fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
refSpec = ImzObj.getspectrum(60)[1]
p = ax.plot(refSpec)
outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=wl_, polyorder=po_)
p, = ax.plot(outspectrum)
plt.subplots_adjust(bottom=0.25)
ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
win_len = Slider(ax_slide, 'window length', valmin=5, valmax=199, valinit=99, valstep=2)
# por_len = Slider(ax_slide, 'window length', valmin=po_, valmax=99, valinit=99, valstep=2)
def update(val):
    current_v = int(win_len.val)
    # current_v = int(por_len.val)
    outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=current_v, polyorder=po_)
    # outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=wl_, polyorder=current_v)
    p.set_ydata(outspectrum)
    fig.canvas.draw()
win_len.on_changed(update)
# por_len.on_changed(update)
plt.show()