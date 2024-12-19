import pandas as pd
import os

# Settings
MODEL = 'o1-mini-2024-09-12'  # o1 model name used in filenames
MODEL_PRETTY_NAME = 'o1-mini'  # How o1 will appear in table
MODEL_PRETTY_NAME_TO_PLACE_ABOVE = 'GPT-4o'  # Model to place o1 results above using name in Table 2

# Load data
benchmarks = ['winogrande', 'mmlu-college_medicine', 'mmlu-clinical_knowledge', 'mmlu-virology', 'belebele']
table_2_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/tables/table_2.csv'))
table_2 = pd.read_csv(table_2_path, encoding='utf-8')
additional_dfs = {}
for benchmark in benchmarks:
    o1_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/o1-related/{MODEL}_on_{benchmark}.csv'))
    o1_df = pd.read_csv(o1_path, encoding='utf-8')
    o1_df.rename(columns={'Model': 'Model (Evaluation Benchmark)'}, inplace=True)
    o1_df['Model (Evaluation Benchmark)'] = o1_df['Model (Evaluation Benchmark)'].apply(lambda x: f"{MODEL_PRETTY_NAME} ({benchmark})")
    additional_dfs[benchmark] = o1_df

# Insert additional rows into the main dataframe
for label, row_df in additional_dfs.items():
    # Find the first row index matching "GPT-4o ..." with the appropriate label
    index_to_insert = table_2.index[table_2["Model (Evaluation Benchmark)"].str.contains(f"{MODEL_PRETTY_NAME_TO_PLACE_ABOVE} \\({label}\\)")][0]

    # Insert the row above the matched index
    table_2 = pd.concat([table_2.iloc[:index_to_insert], row_df, table_2.iloc[index_to_insert:]]).reset_index(drop=True)

# Save to updated table
output_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/o1-related/table_2_with_o1-mini.csv'))
table_2.to_csv(output_path, index=False, encoding='utf-8')
