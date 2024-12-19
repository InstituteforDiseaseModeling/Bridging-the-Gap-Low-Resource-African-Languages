import pandas as pd
from statsmodels.stats.inter_rater import to_table, aggregate_raters, fleiss_kappa, cohens_kappa
import os

table_name = "table_A16.csv"

# Load the data
annotator_data = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv')))

languages = ['af', 'zu', 'xh', 'am', 'bm', 'ig', 'nso', 'sn', 'st', 'tn', 'ts']

# final table columns: language, Q1 Cohen's Kappa, Q1 Fleiss' Kappa, Q2 Cohen's Kappa, Q2 Fleiss' Kappa
result_df = pd.DataFrame()
result_df['Language'] = languages

# for each language, calculate the kappa scores between the annotators (columns 'Translation Quality per Annotator 1' and 'Translation Quality per Annotator 2')
for lang in languages:
    lang_data = annotator_data[annotator_data['Target Language'] == lang]
    raters = lang_data[['Translation Quality per Annotator 1', 'Translation Quality per Annotator 2']].to_numpy()

    confusion_matrix, _ = to_table(raters)
    aggregate, _ = aggregate_raters(raters)
    cohen = cohens_kappa(confusion_matrix, return_results=False)
    fleiss = fleiss_kappa(aggregate)

    result_df.loc[result_df['Language'] == lang, 'Q1 Cohen\'s Kappa'] = round(cohen, 3)
    result_df.loc[result_df['Language'] == lang, 'Q1 Fleiss\' Kappa'] = round(fleiss, 3)

# kappa scores for Translation appropriateness ('Cultural Appropriateness per Annotator 1', 'Cultural Appropriateness per Annotator 2')
for lang in languages:
    lang_data = annotator_data[annotator_data['Target Language'] == lang]
    raters = lang_data[['Cultural Appropriateness per Annotator 1', 'Cultural Appropriateness per Annotator 2']].to_numpy()

    confusion_matrix, _ = to_table(raters)
    aggregate, _ = aggregate_raters(raters)
    cohen = cohens_kappa(confusion_matrix, return_results=False)
    fleiss = fleiss_kappa(aggregate)

    result_df.loc[result_df['Language'] == lang, 'Q2 Cohen\'s Kappa'] = round(cohen, 3)
    result_df.loc[result_df['Language'] == lang, 'Q2 Fleiss\' Kappa'] = round(fleiss, 3)

print("Inter-rater Reliability Scores:")
print(result_df)

# Save the results to a CSV file
result_df.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name}')), index=False)
