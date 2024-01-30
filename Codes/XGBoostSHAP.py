from glob import glob
import os
import h5py
from Codes.Utilities import find_nearest
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mtl
# mtl.use("TKAgg")
from pyimzml.ImzMLParser import _bisect_spectrum
from scipy.signal import argrelextrema
import pandas as pd
import shap
import numpy as np
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_iris
rng = np.random.RandomState(31337)

if __name__ == '__main__':
    fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'

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
    peakmzs = mzs[peakPositions]
    gmcsvDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/GM_injured_cord1/'
    wmcsvDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/WM_injured_cord1/'

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
    specIdx = []
    for df in dflist:
        specIdx.extend(df['Spot index'].values)
    savecsv = os.path.join(fileDir, 'pca_umap_hdbscan_gmwm_labels.csv')
    labelDF = pd.read_csv(savecsv)
    labels = labelDF['labels'].values    # labels must start from 0 instead of 1
    print("labels: ", np.unique(labels))
    # for i in range(len(labels)): # uncomment this for GM
    #     if labels[i] == 4:
    #         labels[i] = 3
    #     elif labels[i] == 3:
    #         labels[i] = 1

    # for i in range(len(labels)):
    #     if labels[i] == 1:
    #         labels[i] = 3
    #     elif labels[i] == 3:
    #         labels[i] = 1
    print("interchanged labels: ", np.unique(labels))
    labelImage = np.zeros([max(xCor)+1, max(yCor)+1])
    tol = 150
    ionImageSpectra = np.zeros([len(specIdx), len(peakPositions)])
    ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
    for idx, spot in enumerate(specIdx):
        ints = processedSpectra[spot]
        for jdx, mz_value in enumerate(peakmzs):
            min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
            ionImageSpectra[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
        ionImage3D[xCor[spot], yCor[spot], :] = ionImageSpectra[idx]
        labelImage[xCor[spot], yCor[spot]] = labels[idx]
    plt.imshow(labelImage[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1])
    plt.colorbar()
    plt.show()
    # for s in range(ionImageSpectra.shape[0]):
    #     spectrum = ionImageSpectra[s]
    #     # spectrum = _smooth_spectrum(spectrum, method='savgol', window_length=wl_, polyorder=po_)
    #     ionImageSpectra[s] = spectrum / np.sum(spectrum)

    # col_means = np.mean(ionImageSpectra, axis=0)
    # # print(col_means.shape)
    # # # Divide each column by its mean
    # # regSpecScaled = regSpecNorm / col_means
    # # # Pareto scaling
    # col_stddevs = np.std(ionImageSpectra, axis=0)
    # # # Pareto scaling
    # ionImageSpectra = (ionImageSpectra - col_means) / np.sqrt(col_stddevs)
    data = ionImageSpectra
    labels = labels - 1
    print("labels for XGBoost: ", np.unique(labels))
    spectraDF = pd.DataFrame(data=data,
                             columns=['{:.3f}'.format(m) for m in peakmzs])  # so that shap plots display m/zs
    # print(spectraDF)
    # spectraDF = spectraDF.sample(frac=1, axis=1).reset_index(drop=True)
    spectraDFshuffled = spectraDF.sample(frac=1).reset_index(drop=True) # ,axis=1 to shuffle features
    # print(spectraDFshuffled.iloc[0:100])
    if __name__ != '__main__':  # are features correlated/dependent?
        method_ = ['pearson', 'spearman', 'kendall']
        corr = spectraDF.corr(method=method_[0])
        # if corr.
        grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        ax = sns.heatmap(corr, ax=ax, annot=False,
                         # vmin=0, vmax=1, center=0.5,
                         cbar_ax=cbar_ax,
                         cbar_kws={"orientation": "horizontal"},
                         cmap="YlGnBu",
                         # linewidths=0.5,
                         square=True)
        plt.show()
if __name__ != '__main__':
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    trial_precision = []
    trial_recall = []
    trial_f1 = []
    trial_accuracy = []
    trial_balanced_accuracy = []
    xgb_imp_df = pd.DataFrame()
    nb_runs = 5
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
            print("X_train.shape: ", X_train.shape)
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
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, plot_type="bar", class_names=['healthy', 'between', 'injured'],
                              feature_names=peakmzs)
if __name__ == '__main__':
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42)
    # xgbcl = XGBClassifier(booster='gbtree',     # for tree model
    #                       device='cpu',
    #                       # nthread = 4,  # will be set default by availability
    #                       eta=0.3,  # learning rate default: 0.3
    #                       gamma=0.0,  # min_split_tree loss default: 0 to inf.
    #                       base_score=0.5,
    #                       colsample_bylevel=1.0,
    #                       max_delta_step=0.0,   # default: 0
    #                       min_child_weight=1.0,
    #                       missing=None,
    #                       n_jobs=-1,
    #                       objective='multi:softmax',
    #                       random_state=rng,
    #                       reg_alpha=0.0,
    #                       reg_lambda=1.0,
    #                       scale_pos_weight=1.0,
    #                       tree_method='auto'
    #                       )
    #
    # param_grid = {
    #     'colsample_bytree': [.75, 1],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.8],
    #     'max_depth': [1, 2, 3, 5],
    #     'subsample': [.75, 1],
    #     'n_estimators': list(range(50, 400, 500))
    # }
    # grid_search = GridSearchCV(estimator=xgbcl, scoring='roc_auc', param_grid=param_grid, n_jobs=-1, cv=kfold)
    # grid_result = grid_search.fit(X_train, y_train)
    #
    # print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}','\n')
    xgb_model = XGBClassifier(
                              learning_rate=0.8,
                              n_estimators=100,
                              max_depth=4,
                              min_child_weight=1,
                              gamma=0.5,
                              # missing=None,
                              subsample=0.8,   # default 1
                              colsample_bytree=0.8,
                              reg_lambda=1,     # default L2 value
                              objective='multi:softmax',
                              # objective='binary:logistic',  # lost function to be minimized
                              # objective='multi:softprob', #'multi:softmax',  #'reg:linear',
                              # nthread=4,
                              # scale_pos_weight=1,  # because of class imbalance
                              # early_stopping_rounds=10,
                              eval_metric=['merror', 'mlogloss', 'auc'],
                              seed=207,   # 207
                              n_jobs=1
                              )
    # class_names = {0: 'WM', 1: 'GM', 2: 'GM(inj)', 3: 'injured'}
    class_names = {0: 'WM', 2: 'GM', 1: 'inj'}
    # class_names = {0: 'healthy', 1: 'between', 2: 'injured'}
    def fit_and_score(estimator, X_train, X_test, y_train, y_test):
        """Fit the estimator on the train set and score it on both sets"""
        estimator.fit(X_train, y_train, eval_set=[
                                                  (X_train, y_train),
                                                  (X_test, y_test)
                                                 ])
        train_score = estimator.score(X_train, y_train)
        test_score = estimator.score(X_test, y_test)
        return estimator, train_score, test_score
    # results = {}
    fold = 0
    for train_index, test_index in kfold.split(data, labels):
        fold += 1
    #     print("\nfold: {}".format(fold))
        X_train = data[train_index]
    #     # np.random.shuffle(train_index)
        y_train = labels[train_index]
        X_test = data[test_index]
        y_test = labels[test_index]
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train)
        xgb_model = clone(xgb_model)
        # est, train_score, test_score = fit_and_score(
    #     #     clone(xgb_model), X_train, X_test, y_train, y_test
    #     # )
    #     # results[est] = (train_score, test_score)
    #     # print(type(clone(xgb_model)))
        xgb_model.fit(X_train, y_train,
                      # verbose=0,
                      sample_weight=sample_weights,
                      eval_set=[
                              (X_train, y_train),
                              (X_test, y_test)
                               ]
                      )
        print(xgb_model.score(X_train, y_train))
        print(xgb_model.score(X_test, y_test))
        # predictions = xgb_model.predict(X_test)
        # actuals = y_test
        # print('\n------------------ Confusion Matrix -----------------\n')
        # print('Accuracy: {:.2f}'.format(accuracy_score(actuals, predictions)))
        # print('Balanced Accuracy: {:.2f}'.format(balanced_accuracy_score(actuals, predictions)))
        results_ = xgb_model.evals_result()
        epochs = len(results_['validation_0']['mlogloss'])
        x_axis = range(0, epochs)
        # fig, ax = plt.subplots(figsize=(9, 5))
        # ax.plot(x_axis, results_['validation_0']['mlogloss'], label='Train')
        # ax.plot(x_axis, results_['validation_1']['mlogloss'], label='Test')
        # ax.legend()
        # plt.ylabel('mlogloss')
        # plt.title('XGBoost loss')
        # plt.show()
    # print("results >> \n", results)
    # class_labels = ['WM', 'GM', 'GM(inj)', 'injured']
    class_labels = ['WM', 'inj', 'GM']
    plot_confusion_matrix(xgb_model, X_test, y_test,
                          display_labels=class_labels, cmap='binary', colorbar=False)
    plt.show()
    explainer = shap.TreeExplainer(xgb_model)  # for tree, background data is not necessary, data=spectraDFshuffled)
    shap_values = explainer.shap_values(spectraDF)
    shap.summary_plot(shap_values, spectraDF.values, max_display=50, plot_type="bar",
                                  class_names=class_names,
                                  feature_names=spectraDF.columns)
    plt.show()
    plt.vlines(peakmzs, 0, np.mean(shap_values[0], axis=0), label='wm', colors='k',
                      linestyles='solid', alpha=0.5)
    plt.vlines(peakmzs, 0, np.mean(shap_values[1], axis=0), label='gm', colors='g',
                      linestyles='solid', alpha=0.5)
    plt.vlines(peakmzs, 0, np.mean(shap_values[2], axis=0), label='gm+', colors='b',
                      linestyles='solid', alpha=0.5)
    # plt.vlines(peakmzs, 0, np.mean(shap_values[3], axis=0), label='injured', colors='r',
    #                   linestyles='solid', alpha=0.5)
    plt.legend()
    plt.show()
    # shap.summary_plot(shap_values, spectraDF.values, feature_names=spectraDF.columns)
    # plt.show()
    # shap.plots.beeswarm(shap_values)
    # plt.show()
    # i = 18
    # shap.force_plot(explainer.expected_value[0],
    #                 shap_values[0][i],
    #                 spectraDF.values[i],
    #                 feature_names=spectraDF.columns)
    # plt.show()
    # row = 18
    # X_test = pd.DataFrame(data=X_test, columns=['{:.3f}'.format(m) for m in peakmzs]) # so that shap plots display m/zs
    # shap.waterfall_plot(
    #         shap.Explanation(values=shap_values[1][row],
    #                          base_values=explainer.expected_value[0],
    #                          data=spectraDF.iloc[row],
    #                          feature_names=spectraDF.columns.tolist()))
    # plt.show()

    spatialSHAP = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
    for idx, spot in enumerate(specIdx):
        # print("idx: {}, spot: {}".format(idx, spot))
        spatialSHAP[xCor[spot], yCor[spot], :] = shap_values[1][idx, :]

    mz_v = 2607.899 #9255#6280#9570.324#9217.902#7769.137
    nearest_mzv = find_nearest(peakmzs, mz_v)
    print("nearest_mzv: ", nearest_mzv)
    plt.imshow(spatialSHAP[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1,
               np.where(peakmzs == nearest_mzv)[0]])
    plt.colorbar()
    plt.show()

