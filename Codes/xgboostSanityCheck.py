# 01.16.2024
from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib as mtl
mtl.use("TKAgg")
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler as SS
from Codes.Utilities import find_nearest, _smooth_spectrum, umap_it, hdbscan_it, chart
from Codes.Utilities import saveimages, poissonScaling
from scipy.signal import argrelextrema
import pandas as pd
import shap
import numpy as np
from pyimzml.ImzMLParser import _bisect_spectrum
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

rng = np.random.RandomState(31337)

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'

if __name__ != '__main__':
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
    # processedSpectra = processedSpectra[:, peakPositions]
    # print("peak selected spectra:", processedSpectra.shape)
    peakmzs = mzs#[peakPositions]

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

    dflist = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df]
    # dflist = [wm1df, wm2df, wm3df, wm4df, wm5df, wm6df]
    # dfList = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df,
    #           wm1df, wm2df, wm3df, wm4df, wm5df, wm6df]

    specIdx = []
    tissuelabels = []

    for i, df in enumerate(dflist, start=1):
        specIdx.extend(df['Spot index'].values)
        num_samples = len(df['Spot index'])
        labels = [i] * num_samples
        tissuelabels.extend(labels)
    print("number of labels: ", len(np.unique(tissuelabels)), ">>",  np.unique(tissuelabels))
    spectra = processedSpectra[specIdx]
    # spectra = spectra[:, peakPositions]
    print("#95 -> spectra.shape", spectra.shape)
    # ionImageSpectra = np.zeros([len(specIdx), len(peakPositions)])
    # ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
    # # spectra3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
    # tol = 20
    # for idx, spot in enumerate(specIdx):
    #     ints = processedSpectra[spot]
    #     for jdx, mz_value in enumerate(peakmzs):
    #         min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
    #         ionImageSpectra[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
    #     ionImage3D[xCor[spot], yCor[spot], :] = ionImageSpectra[idx]
        # spectra3D[xCor[spot], yCor[spot], :] = spectra[idx]

    # print("#110 -> ionImageSpec.shape", ionImageSpectra.shape)
    # saveimages(ionImage3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, :], peakmzs, r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/ionimagesWM/')
    # saveimages(spectra3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, :], peakmzs, r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/ionimages_bad/')

    # mzv_ = find_nearest(peakmzs, 11553.0)
    # plt.subplot(121)
    # plt.imshow(ionImage3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, np.where(peakmzs==mzv_)[0][0]])
    # plt.subplot(122)
    # plt.imshow(spectra3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, np.where(peakmzs==mzv_)[0][0]])
    # plt.show()

    # spectra = ionImageSpec
    # mzs = peakmzs
    regSpecNorm = np.zeros_like(spectra) #+ 1e-6
    for s in range(regSpecNorm.shape[0]):
        spectrum = spectra[s]
        # spectrum = _smooth_spectrum(spectrum, method='savgol', window_length=wl_, polyorder=po_)
        regSpecNorm[s] = spectrum/np.median(spectrum)

    # # Level scaling
    col_means = np.mean(regSpecNorm, axis=0)
    col_stddevs = np.std(regSpecNorm, axis=0)

    # # Pareto scaling
    regSpecScaled = (regSpecNorm - col_means) / np.sqrt(col_stddevs)

    # # Poisson scaling
    # regSpecScaled = poissonScaling(regSpecNorm)
    # +----------------+
    # |      pca       |
    # +----------------+
    RandomState = 20210131
    pca = PCA(random_state=RandomState) #, n_components=10)
    # regSpecNormSS = SS().fit_transform(regSpecScaled) # 0-mean, 1-variance
    pcScores = pca.fit_transform(regSpecScaled)
    print("pcs shape: ", pcScores.shape)
    pca_range = np.arange(1, pca.n_components_, 1)
    print(">> PCA: number of components #{}".format(pca.n_components_))
    evr = pca.explained_variance_ratio_
    evr_cumsum = np.cumsum(evr)
    threshold = 0.90
    cut_evr = find_nearest(evr_cumsum, threshold)
    nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
    print(">> Nearest variance to threshold {:.4f} explained by #PCA components {}".format(cut_evr, nPCs))
    if nPCs >= 50:  # 2 conditions to choose nPCs.
        nPCs = 50
    # df_pca = pd.DataFrame(data=pcs[:, 0:nPCs], columns=['PC_%d' % (i + 1) for i in range(nPCs)])
    plot_pca = True
    MaxPCs = nPCs + 3
    fig, ax = plt.subplots()
    ax.bar(pca_range[0:MaxPCs], evr[0:MaxPCs] * 100, color="steelblue")
    ax.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
    ax.set_xlabel('Principal component number', fontsize=15)
    ax.set_ylabel('Percentage of \n variance explained', fontsize=15)
    ax.set_ylim([-0.5, 100])
    ax.set_xlim([-0.5, MaxPCs])
    ax.grid("on")

    ax2 = ax.twinx()
    ax2.plot(pca_range[0:MaxPCs], evr_cumsum[0:MaxPCs] * 100, color="tomato", marker="D", ms=7)
    ax2.scatter(nPCs, cut_evr * 100, marker='*', s=500, facecolor='blue')
    ax2.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
    ax2.set_ylabel('Cumulative percentage', fontsize=15)
    ax2.set_ylim([-0.5, 100])

    # axis and tick theme
    ax.tick_params(axis="y", colors="steelblue")
    ax2.tick_params(axis="y", colors="tomato")
    ax.tick_params(size=10, color='black', labelsize=15)
    ax2.tick_params(size=10, color='black', labelsize=15)
    ax.tick_params(width=3)
    ax2.tick_params(width=3)

    ax = plt.gca()  # Get the current Axes instance

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)

    plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=15)
    plt.show()

    componentMatrix = pca.components_.T  # eigenvectors or weights or components or coefficients
    print(componentMatrix.shape)
    componentMatrixDF = pd.DataFrame(componentMatrix[:, 0:nPCs], columns=['PC%02d' % (i + 1) for i in range(nPCs)])
    # print(componentMatrixDF)
    loadings = pca.components_.T #* np.sqrt(pca.explained_variance_)
    print(loadings.shape)
    loadingsMatrixDF = pd.DataFrame(loadings[:, 0:nPCs], columns=['PC%02d' % (i + 1) for i in range(nPCs)])
    # loadingsMatrixDF.insert(loadingsMatrixDF.shape[1], column='mzs', value=mzs)
    # loadingsMatrixDF.loc[len(loadingsMatrixDF)] = pca.explained_variance_ratio_[0:nPCs+1]
    savecsv = os.path.join(fileDir, 'loadings5regions.csv')

    if 'colors_' not in plt.colormaps():
        # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
        colors = [
                  # (0.0, 0.0, 0.0, 0.0),
                  (0.15, 0.95, 0.95, 1.0),
                  (0.10, 0.10, 0.10, 1.0),
                  (0.15, 0.75, 0.15, 1.0),
                  (0.15, 0.15, 0.75, 1.0),
                  (0.95, 0.75, 0.05, 1.0),
                  (0.75, 0.15, 0.15, 1.0),
                  ]

    color_bin = len(np.unique(tissuelabels))
    mtl.colormaps.register(LinearSegmentedColormap.from_list(name='colors_', colors=colors, N=color_bin))

    markers = ['.', '.', '.', '.', 'o', 'o',]

    segments = [
                'gm1', 'gm2', 'gm3', 'gm4', 'gm5', 'gm6',
                # 'wm1', 'wm2', 'wm3', 'wm4', 'wm5', 'wm6'
        ]
    colors = [
              # 'mediumspringgreen', 'yellowgreen', 'darkgreen', 'orange', 'darkgoldenrod', 'darkred',
              'mediumspringgreen', 'yellowgreen', 'darkgreen', 'orange', 'darkgoldenrod', 'darkred']
    tissuelabels = np.array(tissuelabels)
    labels = np.unique(tissuelabels)
    # print(len(labels), len(colors), len(markers))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for idf, (label, marker) in enumerate(zip(labels, markers)):
        print(label, marker)
        indices = np.where(tissuelabels == label)[0]
        ax.scatter(pcScores[indices, 0], pcScores[indices, 1], pcScores[indices, 2],
                   c=colors[idf], #tissuelabels[indices],
                   label=segments[idf],
                   # cmap=,  # marker='o')
                   marker=marker,  # 'o'
                   alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Score Plot GM-WM, Pareto scaled, median normalized')
    ax.legend()
    plt.show()

    pcaScoreImages = np.zeros([max(xCor)+1, max(yCor)+1, nPCs])
    for idx, spot in enumerate(specIdx):
        pcaScoreImages[xCor[spot], yCor[spot], :] = pcScores[idx, 0:nPCs]

    fig, axes = plt.subplots(nPCs, 2, figsize=(15, 25))
    # Loop through the subplots and plot images and graphs
    for i in range(nPCs):
        # Plot images in the first column
        rotated_image = np.rot90(pcaScoreImages[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, i])
        axes[i, 0].imshow(rotated_image,
                          cmap='coolwarm')
        axes[i, 0].set_title('{:.3f}'.format(pca.explained_variance_ratio_[i]))
        axes[i, 0].axis('off')  # Turn off axis ticks and labels

        # Plot graphs in the second column
        # axes[i, 1].plot(data_graphs[i], label=f'Graph {i + 1}')
        axes[i, 1].vlines(peakmzs[peakPositions], 0, loadingsMatrixDF.values[:, i][peakPositions], label='loading {}'.format(i+1), colors='k', linestyles='solid', alpha=0.5)
        # axes[i, 1].set_title(f'Graph {i + 1}')

        # Add colorbar to the images
        img_plot = axes[i, 0].imshow(rotated_image, cmap='coolwarm')
        fig.colorbar(img_plot, ax=axes[i, 0], orientation='vertical', fraction=0.05, pad=0.1)

    # Adjust layout to prevent clipping of titles
    fig.tight_layout()
    plt.show()

    favarimax = FactorAnalysis(n_components=nPCs, rotation="varimax", random_state=RandomState)
    varScores = favarimax.fit_transform(pcScores[:, 0:nPCs])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(varScores[:, 0], varScores[:, 1], varScores[:, 2],
                         c=tissuelabels,
                         cmap='colors_', #marker='o')
                         marker='o')
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.set_zlabel('V3')
    ax.set_title('Varimax on PCScore, Poisson scaled, median normalized')
    # Add a legend
    # for region, marker in markers.items():
    #     ax.scatter([], [], [], marker='+', label=f'Region {region}')

    # Display the legend
    ax.legend()
    cbar = plt.colorbar(scatter)
    plt.show()

    adamPCA = np.zeros([len(specIdx), len(peakPositions)])
    for sI in range(len(specIdx)):
        adamPCA[sI, :] = pcScores[sI, 0]*loadingsMatrixDF.values[:, 0][peakPositions] + \
        np.random.normal(0, 0.01, adamPCA.shape[1])
    print("adamPCA: ", adamPCA.shape)

    fig, axes = plt.subplots(2, 1, figsize=(15, 25))
    axes[0].vlines(peakmzs[peakPositions], 0, loadingsMatrixDF.values[:, 0][peakPositions], label='loading', colors='k', linestyles='solid', alpha=0.5)
    axes[1].vlines(peakmzs[peakPositions], 0, adamPCA[30], label='PC1 x loading + noise', colors='k', linestyles='solid', alpha=0.5)
    fig.tight_layout()
    plt.show()

    savecsv = os.path.join(fileDir, 'umap_hdbscan_gm_labels.csv')
    labelDF = pd.read_csv(savecsv)
    labels = labelDF['labels'].values    # labels must start from 0 instead of 1
    print("labels: ", np.unique(labels))

    for i in range(len(labels)):
        if labels[i] == 4:
            labels[i] = 3
        elif labels[i] == 3:
            labels[i] = 1

    # for i in range(len(labels)):
    #     if labels[i] == 1:
    #         labels[i] = 3
    #     elif labels[i] == 3:
    #         labels[i] = 1

    print("interchanged labels: ", np.unique(labels))
    labelImage = np.zeros([max(xCor)+1, max(yCor)+1])

    for idx, spot in enumerate(specIdx):
        labelImage[xCor[spot], yCor[spot]] = labels[idx]

    plt.imshow(labelImage[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1])
    plt.colorbar()
    plt.show()

    data = adamPCA
    peakmzs = mzs[peakPositions]
    labels = labels - 1
    print("labels for XGBoost: ", np.unique(labels))
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    trial_precision = []
    trial_recall = []
    trial_f1 = []
    trial_accuracy = []
    trial_balanced_accuracy = []
    xgb_imp_df = pd.DataFrame()
    nb_runs = 50
    for nb in range(nb_runs):
        sum_accuracy = 0
        sum_balanced_accuracy = 0
        sum_precision = 0
        sum_recall = 0
        sum_f1 = 0
        sum_xgb_feat_imp = np.zeros(len(peakmzs))
        for train_index, test_index in kf.split(data):
            # balancing 'target' class weights
            X_train = data[train_index]
            # np.random.shuffle(train_index)
            y_train = labels[train_index]
            X_test = data[test_index]
            y_test = labels[test_index]
            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=y_train)
            xgb_model = XGBClassifier(learning_rate=0.8,
                                      # n_estimators=1000,
                                      max_depth=15,
                                      min_child_weight=1,
                                      gamma=0,
                                      # missing=1,
                                      subsample=0.8,   # default 1
                                      colsample_bytree=0.8,
                                      reg_lambda=1,     # default L2 value
                                      objective='multi:softmax',
                                      # objective='binary:logistic',  # lost function to be minimized
                                      # objective='multi:softprob', #'multi:softmax',  #'reg:linear',
                                      # nthread=4,
                                      # scale_pos_weight=1,  # because of class imbalance
                                      early_stopping_rounds=10,
                                      eval_metric=['merror', 'mlogloss', 'auc'],
                                      seed=27 + nb,
                                      n_jobs=1).fit(
                X_train, y_train,
                verbose=0,
                sample_weight=sample_weights,  # class weights to combat unbalanced 'target'
                eval_set=[(X_train, y_train),
                          (X_test, y_test)]
                )
            predictions = xgb_model.predict(X_test)
            actuals = y_test
            print('\n------------------ Confusion Matrix -----------------\n')
            print(confusion_matrix(actuals, predictions))
            sum_accuracy += accuracy_score(actuals, predictions)
            sum_balanced_accuracy += balanced_accuracy_score(actuals, predictions)
            sum_precision += precision_score(actuals, predictions, average='weighted')
            sum_recall += recall_score(actuals, predictions, average='weighted')
            sum_f1 += f1_score(actuals, predictions, average='weighted')
            sum_xgb_feat_imp += xgb_model.feature_importances_
            print('\nAccuracy: {:.2f}'.format(accuracy_score(actuals, predictions)))
            print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(actuals, predictions)))

            print('Micro Precision: {:.2f}'.format(precision_score(actuals, predictions, average='micro')))
            print('Micro Recall: {:.2f}'.format(recall_score(actuals, predictions, average='micro')))
            print('Micro F1-score: {:.2f}\n'.format(f1_score(actuals, predictions, average='micro')))

            print('Macro Precision: {:.2f}'.format(precision_score(actuals, predictions, average='macro')))
            print('Macro Recall: {:.2f}'.format(recall_score(actuals, predictions, average='macro')))
            print('Macro F1-score: {:.2f}\n'.format(f1_score(actuals, predictions, average='macro')))

            print('Weighted Precision: {:.2f}'.format(precision_score(actuals, predictions, average='weighted')))
            print('Weighted Recall: {:.2f}'.format(recall_score(actuals, predictions, average='weighted')))
            print('Weighted F1-score: {:.2f}'.format(f1_score(actuals, predictions, average='weighted')))

            print('\n--------------- Classification Report ---------------\n')
            print(classification_report(actuals, predictions))
            print('---------------------- XGBoost ----------------------')
        trial_accuracy.append(sum_accuracy/kf.n_splits)
        trial_balanced_accuracy.append(sum_balanced_accuracy/kf.n_splits)
        trial_precision.append(sum_precision/kf.n_splits)
        trial_recall.append(sum_recall/kf.n_splits)
        trial_f1.append(sum_f1/kf.n_splits)
        sum_xgb_feat_imp /= kf.n_splits
        new_column = pd.Series(sum_xgb_feat_imp, name='{}'.format(nb + 1))
        xgb_imp_df = pd.concat([xgb_imp_df, new_column], axis=1)
    peakmzs_rounded = [round(value, 3) for value in peakmzs]
    xgb_imp_df.index = pd.Index(peakmzs_rounded, name='features')
    print(xgb_imp_df)
    metricDF = pd.DataFrame({
        'accuracy': trial_accuracy,
        'balanced_accuracy': trial_balanced_accuracy,
        'recall': trial_recall,
        'precision': trial_precision,
        'f1': trial_f1
    })
    savecsv = os.path.join(fileDir, 'GMfeat_importance_PC1xLoad1_std0_01.csv')
    # print(savecsv)
    xgb_imp_df.to_csv(savecsv, index=True, sep=',')
    savecsv = os.path.join(fileDir, 'GMfeat_importance_metric_PC1xLoad1_std0_01.csv')
    metricDF.to_csv(savecsv, index=True, sep=',')

savecsv = os.path.join(fileDir, 'GMfeat_importance50.csv')#'GMfeat_importance_PC1xLoad1.csv')
featImpDF = pd.read_csv(savecsv)
columns_to_average = featImpDF.columns[1:]
featImpAvg = featImpDF[columns_to_average].mean(axis=1)
# featImpAvg = np.mean(featImpDF.loc[:, 1:], axis=1)

print(featImpAvg.shape)

plt.plot(featImpDF['features'], featImpAvg)
plt.show()