import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from sklearn.inspection import permutation_importance

winogrande = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__),
                                                       '../../data/translations_and_llm_responses/2. Winogrande Data.csv')),
                         encoding='utf-8')
print('winnogrande tab 2', winogrande.shape)

capp = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__),
                                                 '../../data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv')),
                   encoding='utf-8')
print('winnogrande tab 3', capp.shape)

winogrande_capp = pd.merge(winogrande, capp, on='Key',
                           how='left')  # instead of 'left' can also use 'inner' or 'right', etc.
print(winogrande_capp.shape)


# Define 'Good' as rated by tab 3 annotators
def get_target(row):
    print(row)
    if isinstance(row['Translation Quality per Annotator 1'], str):
        if 'Good' in row['Translation Quality per Annotator 1'] and 'Good' in row[
            'Translation Quality per Annotator 2']:
            return 1
    return 0


df = winogrande_capp.__deepcopy__()
print(df.columns)

# ## Define columns for quality targets, as well as float and categorical columns
quality_targets = ['Translation Quality per Annotator 1', 'Translation Quality per Annotator 2']

float_cols = ['ROUGE-1 Score for Initial Translation',
              'ROUGE-1 Score for Corrected Translation']

cat_cols = ['Assessment of Translation Quality by Verifying Translator',
            'Warning Observed by Initial Translator',
            'Warning Observed by Verifying Translator',
            'Target Language_x',
            ]
quality_cols = float_cols + cat_cols
all_cols = quality_cols + quality_targets

# trim columns to those of interest
df = df[all_cols]

# ensure float columns are numeric
for col in float_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# get target column based on 2 annotators
df['target'] = df.apply(get_target, axis=1)

# sanity check 65% of the data is labeled 'good' by both annotators
df[df.target == 1].shape[0] / df.shape[0]

# drop the remaining quality columns
df = df.drop(quality_targets, axis=1)
df = pd.get_dummies(df, columns=cat_cols)


# Balance the dataset
def create_balanced_dataset(df):
    num_good = df[df["target"] == 1].shape[0]
    num_bad = df[df["target"] == 0].shape[0]

    good_subset = df[df["target"] == 1].sample(num_bad, random_state=42)
    balanced_df = pd.concat([good_subset, df[df["target"] == 0]])
    return balanced_df


balanced_df = create_balanced_dataset(df)
print(balanced_df["target"].value_counts())

y = balanced_df['target']
X = balanced_df.copy().drop(['target'], axis=1)

# split data into test, train
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# looking over features
print(X)

best_params = {
    'n_estimators': 350,
    'min_samples_split': 100,
    'min_samples_leaf': 25,
    'max_features': 'sqrt',
    'max_depth': None,
    'bootstrap': True,
    'random_state': 42,
}

# Train on the best parameters
forest = RandomForestClassifier(**best_params)
forest.fit(X_train, y_train)

# # Get results of balanced dataset
forest.score(X_test, y_test)

# # confusion matrix and results
cf_matrix = confusion_matrix(y_test, forest.predict(X_test))

plt.figure()
group_names = ['TN', 'FP', 'FN', 'TP']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Condition')

target_names = ['bad', 'good']
print(classification_report(y_test, forest.predict(X_test), target_names=target_names))

feature_names = [x.replace('_x_', '_').replace('_⚠️:', ': ').replace('Target Language_', 'Language: ').replace('ROUGE-1 Score for ', 'ROUGE-1 vs MT: ').replace('Assessment of Translation Quality by Verifying Translator_No - Needs Correction', 'Translation: Needed Correction').replace('Assessment of Translation Quality by Verifying Translator_Yes - Perfect Translation', 'Translation: Did Not Need Correction').replace('Warning Observed by Initial Translator:  Significant Similarity with AI Translation', 'User Warning: Translation Very Similar to MT Initial Translation').replace('Warning Observed by Verifying Translator:  Significant Similarity with AI Translation', 'User Warning: Translation Very Similar to MT Corrected Translation').replace('Warning Observed by Initial Translator: PERFECT AI Translation Match', 'User Warning: Translation Matches MT Initial Translation').replace('Warning Observed by Verifying Translator: PERFECT AI Translation Match', 'User Warning: Translation Matches MT Corrected Translation').replace('Warning Observed by Initial Translator:  POOR AI Translation Match', 'User Warning: Translation Very Dissimilar to MT Initial Translation').replace('Warning Observed by Verifying Translator:  POOR AI Translation Match', 'User Warning: Translation Very Dissimilar to MT Corrected Translation').replace('Initial Translation', '(Initial Translation)').replace('Corrected Translation', '(Corrected Translation)') for x in X.columns]

result = permutation_importance(
    forest, X_test, y_test, n_repeats=20, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=[x.replace('_x_', '_').replace('_⚠️:', ': ').replace('Target Language_', 'Language: ').replace('ROUGE-1 Score for ', 'ROUGE-1 vs MT: ').replace('Assessment of Translation Quality by Verifying Translator_No - Needs Correction', 'Translation: Needed Correction').replace('Assessment of Translation Quality by Verifying Translator_Yes - Perfect Translation', 'Translation: Did Not Need Correction').replace('Warning Observed by Initial Translator:  Significant Similarity with AI Translation', 'User Warning: Translation Very Similar to MT Initial Translation').replace('Warning Observed by Verifying Translator:  Significant Similarity with AI Translation', 'User Warning: Translation Very Similar to MT Corrected Translation').replace('Warning Observed by Initial Translator: PERFECT AI Translation Match', 'User Warning: Translation Matches MT Initial Translation').replace('Warning Observed by Verifying Translator: PERFECT AI Translation Match', 'User Warning: Translation Matches MT Corrected Translation').replace('Warning Observed by Initial Translator:  POOR AI Translation Match', 'User Warning: Translation Very Dissimilar to MT Initial Translation').replace('Warning Observed by Verifying Translator:  POOR AI Translation Match', 'User Warning: Translation Very Dissimilar to MT Corrected Translation').replace('Initial Translation', '(Initial Translation)').replace('Corrected Translation', '(Corrected Translation)') for x in X.columns[sorted_importances_idx]]
)

ax = importances.plot.box(vert=False, whis=10)
ax.axvline(x=0, color="k", linestyle="--")

# Add x and y axis labels
ax.set_xlabel("Permutation Importance")
ax.set_ylabel("Feature")

plt.savefig(os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/figures/figure_A19.pdf')), format='pdf',
            dpi=1200, bbox_inches='tight')
