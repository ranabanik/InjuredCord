import os
from glob import glob
import h5py
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from pyimzml.ImzMLParser import _bisect_spectrum
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import matplotlib.gridspec as gridspec
import shap

num_feat = 100
tol = 150
fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'
diffCSV = os.path.join(fileDir, 'GMWMfeatNulldiff{}_{}.csv'.format(num_feat, tol))
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
peakmzs = mzs[peakPositions]
if __name__ == '__main__':
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
    savecsv = os.path.join(fileDir, 'umap_hdbscan_gmwm_labels.csv')
    labelDF = pd.read_csv(savecsv)
    labels = labelDF['labels'].values  # labels must start from 0 instead of 1
    print("labels: ", np.unique(labels))
    print("interchanged labels: ", np.unique(labels))
    labelImage = np.zeros([max(xCor) + 1, max(yCor) + 1])
    tol = 150
    ionImageSpectra = np.zeros([len(specIdx), len(peakPositions)])
    ionImage3D = np.zeros([max(xCor) + 1, max(yCor) + 1, len(peakPositions)])
    for idx, spot in enumerate(specIdx):
        ints = processedSpectra[spot]
        for jdx, mz_value in enumerate(peakmzs):
            min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
            ionImageSpectra[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
        ionImage3D[xCor[spot], yCor[spot], :] = ionImageSpectra[idx]
        labelImage[xCor[spot], yCor[spot]] = labels[idx]
    plt.imshow(labelImage[min(xCor):max(xCor) + 1, min(yCor):max(yCor) + 1])
    plt.colorbar()
    plt.show()
    labels = labels - 1
    print("labels for XGBoost: ", np.unique(labels))
if __name__ != '__main__':  # this cell is for null and feat importance run
    spectraDF = pd.DataFrame(data=ionImageSpectra,
                             columns=['{:.3f}'.format(m) for m in peakmzs])  # so that shap plots display m/zs
    # print(spectraDF)
    # spectraDF = spectraDF.sample(frac=1, axis=1).reset_index(drop=True)
    # spectraDFshuffled = spectraDF.sample(frac=1).reset_index(drop=True)  # ,axis=1 to shuffle features
    # print(spectraDFshuffled.iloc[0:100])
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
    trial_precision = []
    trial_recall = []
    trial_f1 = []
    trial_accuracy = []
    trial_balanced_accuracy = []
    xgb_imp_df = pd.DataFrame()
    nb_runs = 50
    for nb in range(nb_runs):
        print("\n")
        print("+--------------------+")
        print("+---------{}---------+".format(nb+1))
        print("+--------------------+")
        print("\n")
        xgb_model = XGBClassifier(
            learning_rate=0.8,
            n_estimators=100,
            max_depth=4,
            min_child_weight=1,
            gamma=0.5,
            # missing=None,
            subsample=0.8,  # default 1
            colsample_bytree=0.8,
            reg_lambda=1,  # default L2 value
            objective='multi:softmax',
            # objective='binary:logistic',  # lost function to be minimized
            # objective='multi:softprob', #'multi:softmax',  #'reg:linear',
            # nthread=4,
            # scale_pos_weight=1,  # because of class imbalance
            # early_stopping_rounds=10,
            eval_metric=['merror', 'mlogloss', 'auc'],
            seed=207 + nb,
            n_jobs=1
        )
        sum_accuracy = 0
        sum_balanced_accuracy = 0
        sum_precision = 0
        sum_recall = 0
        sum_f1 = 0
        sum_xgb_feat_imp = np.zeros(len(peakmzs))
        for train_index, test_index in kf.split(spectraDF, labels):
            # balancing 'target' class weights
            X_train = spectraDF.values[train_index]
            # np.random.shuffle(train_index)
            y_train = labels[train_index]
            X_test = spectraDF.values[test_index]
            y_test = labels[test_index]
            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=y_train)
            xgb_model = clone(xgb_model)
            xgb_model.fit(X_train, y_train,
                          # verbose=0,
                          sample_weight=sample_weights,
                          eval_set=[
                              (X_train, y_train),
                              (X_test, y_test)
                                   ])
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
    print("xgb_imp_df: \n", xgb_imp_df)
    metricDF = pd.DataFrame({
        'accuracy': trial_accuracy,
        'balanced_accuracy': trial_balanced_accuracy,
        'recall': trial_recall,
        'precision': trial_precision,
        'f1': trial_f1
        })
    print("metricDF: \n",metricDF)
    savecsv = os.path.join(fileDir, 'GMWMfeat_importance{}_{}.csv'.format(tol, nb_runs))
    # xgb_imp_df.to_csv(savecsv, index=True, sep=',')
    savecsv = os.path.join(fileDir, 'GMWMfeat_importance{}_{}_metric.csv'.format(tol, nb_runs))
    # metricDF.to_csv(savecsv, index=True, sep=',')
if __name__ != '__main__':
    nullImpCSV = os.path.join(fileDir, 'GMWMnull_importance150_50.csv')
    featImpCSV = os.path.join(fileDir, 'GMWMfeat_importance150_50.csv')
    nullImpDF = pd.read_csv(nullImpCSV)
    featImpDF = pd.read_csv(featImpCSV)
    # peakmzs = [round(value, 3) for value in peakmzs]
    # featImpDF['features'] = pd.concat([peakmzs_rounded, featImpDF], axis=1)
    # featImpDF.index = pd.Index(peakmzs_rounded, name='features')
    # nullImpDF.set_index(peakmzs_rounded, name='features', inplace=True)
    # featImpDF.set_index(peakmzs_rounded, name='features', inplace=True)
    nullImpDF.insert(0, 'features', peakmzs)
    featImpDF.insert(0, 'features', peakmzs)
    nullImpDF.set_index('features', inplace=True)
    featImpDF.set_index('features', inplace=True)
    # print(featImpDF)
# if __name__ != '__main__':
    num_feat = 100
    tol = 150
    feature_scores = []
    for (_f, _n, _i) in zip(nullImpDF.index.values, nullImpDF.mean(axis=1).values, featImpDF.mean(axis=1).values):
        feature_scores.append(('{:3f}'.format(_f), _n, _i))

    scores_df = pd.DataFrame(feature_scores, columns=['features', 'null_imp', 'feat_imp'])
    scores_df['difference'] = scores_df['feat_imp'] - scores_df['null_imp']

    plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(1, 3)

    ax = plt.subplot(gs[0, 0])
    sns.barplot(x='null_imp', y='features', data=scores_df.sort_values('null_imp', ascending=False).iloc[0:num_feat],
                palette='viridis')
    ax.set_title('Feature scores wrt null importance', fontweight='bold', fontsize=14)

    ax = plt.subplot(gs[0, 1])
    sns.barplot(x='feat_imp', y='features', data=scores_df.sort_values('feat_imp', ascending=False).iloc[0:num_feat],
                palette='viridis')
    ax.set_title('Feature scores wrt real importance', fontweight='bold', fontsize=14)

    ax = plt.subplot(gs[0, 2])
    sns.barplot(x='difference', y='features', data=scores_df.sort_values('difference', ascending=False).iloc[0:num_feat],
                palette='viridis')
    ax.set_title('Feature scores (real - null)', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

    savecsv = os.path.join(fileDir, 'GMWMfeatNulldiff{}_{}.csv'.format(num_feat, tol))
    # print(scores_df.sort_values('difference', ascending=False)[0:num_feat])
    diff100DF = scores_df.sort_values('difference', ascending=False)[0:num_feat]
    print(diff100DF)
    # diff100DF.to_csv(savecsv, index=True, sep=',')
if __name__ == '__main__':      # after loading top 100 difference between null and feat importance
    diff100DF = pd.read_csv(diffCSV)
    diff100Idx = np.sort(diff100DF.iloc[:, 0].values)
    # print(diff100Idx)#, type(diff100Idx))
    ionImageSpectra = ionImageSpectra[:, diff100Idx]
    # print(ionImageSpectra.shape)
    peakmzs = peakmzs[diff100Idx]
    # print(peakmzs)
    spectraDF = pd.DataFrame(data=ionImageSpectra,
                             columns=['{:.3f}'.format(m) for m in peakmzs])  # so that shap plots display m/zs
    # print(spectraDF)
    # spectraDF = spectraDF.sample(frac=1, axis=1).reset_index(drop=True)
    spectraDFshuffled = spectraDF.sample(frac=1).reset_index(drop=True)     # ,axis=1 to shuffle features
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
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
                              seed=207,
                              n_jobs=1
                              )
    fold = 0
    for train_index, test_index in kfold.split(spectraDF, labels):
        fold += 1
    #     print("\nfold: {}".format(fold))
        X_train = spectraDF.values[train_index]
    #     # np.random.shuffle(train_index)
        y_train = labels[train_index]
        X_test = spectraDF.values[test_index]
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
    explainer = shap.TreeExplainer(xgb_model, data=spectraDFshuffled)
    shap_values = explainer.shap_values(spectraDF)
    # class_names = {0: 'healthy', 1: 'between', 2: 'injured'}
    class_names = {0: 'WM', 1: 'GM', 2: 'GM+', 3: 'injured'}
    shap.summary_plot(shap_values, spectraDF.values, max_display=100, plot_type="bar",
                                  class_names=class_names,
                                  feature_names=spectraDF.columns)
    plt.show()
    plt.vlines(peakmzs, 0, np.mean(shap_values[0], axis=0), label='wm', colors='k',
                      linestyles='solid', alpha=0.5)
    plt.vlines(peakmzs, 0, np.mean(shap_values[1], axis=0), label='gm', colors='g',
                      linestyles='solid', alpha=0.5)
    plt.vlines(peakmzs, 0, np.mean(shap_values[2], axis=0), label='gm+', colors='b',
                      linestyles='solid', alpha=0.5)
    plt.vlines(peakmzs, 0, np.mean(shap_values[3], axis=0), label='injured', colors='r',
                      linestyles='solid', alpha=0.5)
    plt.legend()
    plt.show()