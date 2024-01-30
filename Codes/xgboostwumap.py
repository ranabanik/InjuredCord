from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib as mtl
# mtl.use("TKAgg")
from pyimzml.ImzMLParser import _bisect_spectrum
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler as SS
from Codes.Utilities import saveimages, find_nearest, _smooth_spectrum, umap_it, hdbscan_it, chart
from scipy.signal import argrelextrema
import pandas as pd
import shap
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
rng = np.random.RandomState(31337)

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
print("peak selected spectra:", processedSpectra.shape)
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

# gmdflist = [gm1df, gm2df, gm3df, gm4df, gm5df, gm6df]
dflist = [
    gm1df, gm2df, gm3df, gm4df, gm5df, gm6df,
    # wm1df, wm2df, wm3df, wm4df, wm5df, wm6df
    ]
# dfinjured = [gm4df, gm5df, gm6df]

specIdx = []
for df in dflist:
    specIdx.extend(df['Spot index'].values)

ionImageSpectra = np.zeros([len(specIdx), len(peakPositions)])
ionImage3D = np.zeros([max(xCor)+1, max(yCor)+1, len(peakPositions)])
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
tol = 50
for idx, spot in enumerate(specIdx):
    ints = processedSpectra[spot]
    for jdx, mz_value in enumerate(peakmzs):
        min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
        ionImageSpectra[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
    ionImage3D[xCor[spot], yCor[spot], :] = ionImageSpectra[idx]
    labelImage[xCor[spot], yCor[spot]] = labels[idx]

# saveimages(ionImage3D[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1, :], peakmzs, r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/ionimagesWM/')
plt.imshow(labelImage[min(xCor):max(xCor)+1, min(yCor):max(yCor)+1])
plt.colorbar()
plt.show()

# spectra = processedSpectra[specIdx]
# print("ionImageSpectra.shape: ", ionImageSpectra.shape)
# regID = corRegID[specIdx]
# regSpecNorm = np.zeros_like(spectra)
# # regID = healthyRegID
# # maxInt = np.max(gmspectra)
# # nmz = gmspectra.shape[1]
# # refSpec = regSpec[60]
# # fnorm = sum(refSpec)
# wl_ = 3
# po_ = 1
# for s in range(regSpecNorm.shape[0]):
#     spectrum = spectra[s]
#     # spectrum = _smooth_spectrum(spectrum, method='savgol', window_length=wl_, polyorder=po_)
#     regSpecNorm[s] = spectrum#/np.median(spectrum)
#
# # # Level scaling
col_means = np.mean(ionImageSpectra, axis=0)
# # print(col_means.shape)
# # # Divide each column by its mean
# # regSpecScaled = regSpecNorm / col_means
# # # Pareto scaling
col_stddevs = np.std(ionImageSpectra, axis=0)
# #
# # # Pareto scaling
ionImageSpectra = (ionImageSpectra - col_means)# / np.sqrt(col_stddevs)

# # iris = load_iris()
# # y = iris['target']
# # X = iris['data']
# # print(iris)
if __name__ == '__main__':
    data = ionImageSpectra
    spectraDF = pd.DataFrame(data=data, columns=['{:.3f}'.format(m) for m in peakmzs]) # so that shap plots display m/zs
    labels = labels - 1
    print("labels for XGBoost: ", np.unique(labels))
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    trial_precision = []
    trial_recall = []
    trial_f1 = []
    trial_accuracy = []
    trial_balanced_accuracy = []
    xgb_imp_df = pd.DataFrame()
    nb_runs = 10
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
                                      # missing=1,  # don't change it!
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
            # print('\n------------------ Confusion Matrix -----------------\n')
            # print(confusion_matrix(actuals, predictions))
            sum_accuracy += accuracy_score(actuals, predictions)
            sum_balanced_accuracy += balanced_accuracy_score(actuals, predictions)
            sum_precision += precision_score(actuals, predictions, average='weighted')
            sum_recall += recall_score(actuals, predictions, average='weighted')
            sum_f1 += f1_score(actuals, predictions, average='weighted')
            sum_xgb_feat_imp += xgb_model.feature_importances_
            print('\nAccuracy: {:.2f}'.format(accuracy_score(actuals, predictions)))
            print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(actuals, predictions)))

            # print('Micro Precision: {:.2f}'.format(precision_score(actuals, predictions, average='micro')))
            # print('Micro Recall: {:.2f}'.format(recall_score(actuals, predictions, average='micro')))
            # print('Micro F1-score: {:.2f}\n'.format(f1_score(actuals, predictions, average='micro')))
            #
            # print('Macro Precision: {:.2f}'.format(precision_score(actuals, predictions, average='macro')))
            # print('Macro Recall: {:.2f}'.format(recall_score(actuals, predictions, average='macro')))
            # print('Macro F1-score: {:.2f}\n'.format(f1_score(actuals, predictions, average='macro')))
            #
            # print('Weighted Precision: {:.2f}'.format(precision_score(actuals, predictions, average='weighted')))
            # print('Weighted Recall: {:.2f}'.format(recall_score(actuals, predictions, average='weighted')))
            # print('Weighted F1-score: {:.2f}'.format(f1_score(actuals, predictions, average='weighted')))
            #
            # print('\n--------------- Classification Report ---------------\n')
            # print(classification_report(actuals, predictions))
            # print('---------------------- XGBoost ----------------------')
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, plot_type="bar", class_names=['healthy', 'between', 'injured'],
                              feature_names=spectraDF.columns)

    #     trial_accuracy.append(sum_accuracy/kf.n_splits)
    #     trial_balanced_accuracy.append(sum_balanced_accuracy/kf.n_splits)
    #     trial_precision.append(sum_precision/kf.n_splits)
    #     trial_recall.append(sum_recall/kf.n_splits)
    #     trial_f1.append(sum_f1/kf.n_splits)
    #     sum_xgb_feat_imp /= kf.n_splits
    #     new_column = pd.Series(sum_xgb_feat_imp, name='{}'.format(nb + 1))
    #     xgb_imp_df = pd.concat([xgb_imp_df, new_column], axis=1)
    # peakmzs_rounded = [round(value, 3) for value in peakmzs]
    # xgb_imp_df.index = pd.Index(peakmzs_rounded, name='features')
    # print(xgb_imp_df)
    # metricDF = pd.DataFrame({
    #     'accuracy': trial_accuracy,
    #     'balanced_accuracy': trial_balanced_accuracy,
    #     'recall': trial_recall,
    #     'precision': trial_precision,
    #     'f1': trial_f1
    # })
    # print(metricDF)
    savecsv = os.path.join(fileDir, 'GMfeat_importance{}_{}.csv'.format(tol, nb_runs))
    # print(savecsv)
    # xgb_imp_df.to_csv(savecsv, index=True, sep=',')
    savecsv = os.path.join(fileDir, 'GMfeat_importance{}_{}_metric.csv'.format(tol, nb_runs))
    # metricDF.to_csv(savecsv, index=True, sep=',')
    # # perm_importance = permutation_importance(xgb_model, spectra[test_index], labels[test_index])
    # sorted_idx = np.argsort(xgb_model.feature_importances_)[::-1][0:20]
    # # plt.bar(mzs[sorted_idx], 1000*perm_importance.importances_mean[sorted_idx])
    # # plt.xlabel("Permutation Importance")
    # # plt.show()
    # featImpVal = 1000*xgb_model.feature_importances_[sorted_idx]
    # mzs_values = mzs[sorted_idx]
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.bar(range(len(sorted_idx)), featImpVal, color='blue', alpha=0.7)
    # # plt.bar(mzs_values, loadings_values, color='blue', alpha=0.7)
    # # Add labels and title
    # # ax.set_xlabel('Index')
    # ax.set_ylabel('perm feat imp. x 1000')
    # ax.set_title('XGBoost feature importance')
    #
    # # Optionally, display the values on top of each bar
    # for i, v in enumerate(featImpVal):
    #     if v < 0:
    #         ax.text(i, v-0.002, f'{v:.3f}', ha='center', va='bottom', size=5)
    #     else:
    #         ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', size=5)
    #
    # ax.set_xticks(np.arange(len(sorted_idx)), ['{:.2f}'.format(value) for value in mzs_values], rotation=90, ha='right', fontsize=8)
    # plt.tight_layout()
    # plt.show()
    # permutation_importance = permutation_importance(xgb_model, spectra[test_index], labels[test_index])

    # correlation_matrix = np.corrcoef(spectra, rowvar=False)
    # sn.heatmap(correlation_matrix)
    # plt.show()
    # from xgboost import plot_importance
    #
    # mtl.use("TKAgg")
    # fig, ax = plt.subplots(figsize=(9, 5))
    # plot_importance(xgb_model, ax=ax)
    # plt.show()
    #
    # from matplotlib.pylab import rcParams
    # import xgboost as xgb
    #
    # rcParams['figure.figsize'] = 28, 12
    # xgb.plot_tree(xgb_model)
    # plt.show()
    # results = xgb_model.evals_result()
    # epochs = len(results['validation_0']['merror'])
    # x_axis = range(0, epochs)
    # # print(results)
    # print(epochs)
    # fig, ax = plt.subplots(figsize=(9, 5))
    # ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    # ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    # ax.legend()
    # plt.ylabel('mlogloss')
    # plt.title('GridSearchCV XGBoost mlogloss')
    # plt.show()
    #
    # # xgboost 'merror' plot
    # fig, ax = plt.subplots(figsize=(9, 5))
    # ax.plot(x_axis, results['validation_0']['merror'], label='Train')
    # ax.plot(x_axis, results['validation_1']['merror'], label='Test')
    # ax.legend()
    # plt.ylabel('merror')
    # plt.title('GridSearchCV XGBoost merror')
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(9, 5))
    # ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    # ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    # ax.legend()
    # plt.ylabel('auc')
    # plt.title('GridSearchCV XGBoost auc')
    # plt.show()
    # spectraDF = pd.DataFrame(data=data, columns=['{:.3f}'.format(m) for m in peakmzs]) # so that shap plots display m/zs
    # explainer = shap.TreeExplainer(xgb_model)
    # shap_values = explainer.shap_values(spectraDF)
    # shap.summary_plot(shap_values, spectraDF)
    # X_sampled = data[test_index]
    # shap.summary_plot(shap_values, X_sampled)
    # shap.plots.waterfall(shap_values[0])

