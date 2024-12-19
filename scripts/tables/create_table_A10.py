import pandas as pd
from statsmodels.stats.inter_rater import to_table
import os

table_name = "table_A10.csv"

# Load the data
annotator_data = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv')))

# get all unique languages from the 'Target Language' column of the annotator data
languages = annotator_data['Target Language'].unique()

# get the total number of dataset items
total_items = len(annotator_data)

# Generate the flattened confusion matrix for each language in one big table for each question
print("\nFlattened confusion matrices for translation quality:")
flattened_categories = ["Wrong / Wrong", "Wrong / Good", "Wrong / Understandable", "Good / Wrong", "Good / Good", "Good / Understandable", "Understandable / Wrong", "Understandable / Good", "Understandable / Understandable", "Considered Good", "Considered Wrong"]
df = pd.DataFrame()
for lang in languages:
    lang_data = annotator_data[annotator_data['Target Language'] == lang]
    raters = lang_data[['Translation Quality per Annotator 1', 'Translation Quality per Annotator 2']]
    confusion_matrix, _ = to_table(raters.to_numpy())
    flat = confusion_matrix.flatten()  # flatten the confusion matrix

    # Count how many are considered a good translation (both annotators rated as "Good translation" or "Understandable")
    good = raters.apply(lambda x: x.iloc[0] != "Completely wrong" and x.iloc[1] != "Completely wrong", axis=1).sum()

    # count how many are considered a bad translation (either annotator rated as "Completely wong")
    bad = raters.apply(lambda x: x.iloc[0] == 'Completely wrong' or x.iloc[1] == 'Completely wrong', axis=1).sum()

    # add to the column
    flat = flat.tolist() + [good, bad]
    df[lang] = flat  # add the flattened confusion matrix to the dataframe the next column
    df[lang] = df[lang].astype(int)

df.index = flattened_categories
df['Avg. (%)'] = 100 * (df.sum(axis=1) / total_items)  # add percent of total for each row


# reorder rows
reorder_rows = "Good / Good", "Good / Understandable", "Good / Wrong", "Understandable / Good", "Understandable / Understandable", "Understandable / Wrong", "Wrong / Good", "Wrong / Understandable", "Wrong / Wrong", "Considered Good", "Considered Wrong"
df = df.reindex(reorder_rows)

print(df)
df.index.name = 'Eval. 1 / Eval. 2'
df.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name}')))
