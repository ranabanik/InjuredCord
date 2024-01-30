import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

fileDir = r'/media/banikr/DATA/MALDI/230713-Chen-intactproteins/'

nullImpDF = pd.read_csv(os.path.join(fileDir, 'GMnull_importance50.csv'))
featImpDF = pd.read_csv(os.path.join(fileDir, 'GMfeat_importance50.csv'))
print(nullImpDF)
nullImpDF.set_index('features', inplace=True)
featImpDF.set_index('features', inplace=True)
# Creating a new DataFrame with 'features' as the index and mean values
# mean_null_df = pd.DataFrame({'mean_values': nullImpDF.mean(axis=1)})
# mean_feat_df = pd.DataFrame({'mean_values': featImpDF.mean(axis=1)})
# # mean_df = mean_df.sort_values(by='mean_values', ascending=False)
print(featImpDF)
# print(nullImpDF.mean(axis=1).values)

feature_scores = []
for (_f, _n, _i) in zip(nullImpDF.index.values, nullImpDF.mean(axis=1).values, featImpDF.mean(axis=1).values):
    feature_scores.append(('{:3f}'.format(_f), _n, _i))

scores_df = pd.DataFrame(feature_scores, columns=['features', 'null_imp', 'feat_imp'])
scores_df['difference'] = scores_df['feat_imp'] - scores_df['null_imp']

plt.figure(figsize=(16, 12), dpi=300)
gs = gridspec.GridSpec(1, 3)

ax = plt.subplot(gs[0, 0])
sns.barplot(x='null_imp', y='features', data=scores_df.sort_values('null_imp', ascending=False).iloc[0:50], palette='viridis')
ax.set_title('Feature scores wrt null importance', fontweight='bold', fontsize=14)

ax = plt.subplot(gs[0, 1])
sns.barplot(x='feat_imp', y='features', data=scores_df.sort_values('feat_imp', ascending=False).iloc[0:50], palette='viridis')
ax.set_title('Feature scores wrt real importance', fontweight='bold', fontsize=14)

ax = plt.subplot(gs[0, 2])
sns.barplot(x='difference', y='features', data=scores_df.sort_values('difference', ascending=False).iloc[0:50], palette='viridis')
ax.set_title('Feature scores (real - null)', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.show()



