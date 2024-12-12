import pandas as pd
from statsmodels.stats.inter_rater import to_table
import os

table_name = "table_A1.csv"

# Load the data
annotator_data = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv')))

# get all unique languages from the 'Target Language' column of the annotator data
languages = annotator_data['Target Language'].unique()

# get the total number of dataset items
total_items = len(annotator_data)

print("\nFlattened confusion matrices for cultural appropriateness:")
flattened_categories = ["D.U. / D.U.", "D.U. / Not sure", "D.U. / Typical", "D.U. / Strange", "Not sure / D.U.", "Not sure / Not sure", "Not sure / Typical", "Not sure / Strange", "Typical / D.U.", "Typical / Not sure", "Typical / Typical", "Typical / Strange", "Strange / D.U.", "Strange / Not sure", "Strange / Typical", "Strange / Strange", "Appropriate + Good", "Inappropriate + Good"]
df = pd.DataFrame()
for lang in languages:
    lang_data = annotator_data[annotator_data['Target Language'] == lang]
    raters = lang_data[['Cultural Appropriateness per Annotator 1', 'Cultural Appropriateness per Annotator 2']]
    confusion_matrix, _ = to_table(raters.to_numpy())
    flat = confusion_matrix.flatten()  # flatten the confusion matrix

    raters = lang_data[['Cultural Appropriateness per Annotator 1', 'Cultural Appropriateness per Annotator 2',
                        'Translation Quality per Annotator 1', 'Translation Quality per Annotator 2']]

    # Count how many are considered typical (both annotators rated as "Typical") given good
    approp = raters.apply(lambda x: (x.iloc[0] == "No, the sentence is typical" and x.iloc[1] == "No, the sentence is typical") and (x.iloc[2] != "Completely wrong" and x.iloc[3] != "Completely wrong"), axis=1).sum()
    # count how many are considered strange (either annotator did not rate as "Typical") given good
    inapprop = raters.apply(lambda x: (x.iloc[0] != 'No, the sentence is typical' or x.iloc[1] != 'No, the sentence is typical') and (x.iloc[2] != "Completely wrong" and x.iloc[3] != "Completely wrong"), axis=1).sum()

    # add to the column
    flat = flat.tolist() + [approp, inapprop]
    df[lang] = flat  # add the flattened confusion matrix to the dataframe the next column
    df[lang] = df[lang].astype(int)

df.index = flattened_categories
df['Avg. (%)'] = 100 * (df.sum(axis=1) / total_items)  # add percent of total for each row

# reorder rows
reorder_rows = ["Typical / Typical", "Typical / Not sure", "Typical / Strange", "Typical / D.U.", "Not sure / Typical", "Not sure / Not sure", "Not sure / Strange", "Not sure / D.U.", "Strange / Typical", "Strange / Not sure", "Strange / Strange", "Strange / D.U.", "D.U. / Typical", "D.U. / Not sure", "D.U. / Strange", "D.U. / D.U.", "Appropriate + Good", "Inappropriate + Good"]
df = df.reindex(reorder_rows)

# save to csv
print(df)
df.index.name = 'Eval. 1 / Eval. 2'
df.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name}')))
