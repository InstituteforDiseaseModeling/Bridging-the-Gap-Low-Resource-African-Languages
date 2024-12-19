import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import llm_responses, languages, runs, baselines_ft

table_name_mean = 'tables_A20-A27_means.csv'
table_name_std = 'tables_A20-A27_stds.csv'

# Initialize output CSV structure
ordered_eval_benchmarks = ['winogrande', 'mmlu-clinical_knowledge', 'mmlu-virology', 'belebele']
ordered_train_benchmarks = ['winogrande', 'mmlu-college_medicine']
full_index = ['Base Llama 3 70B IT (N/A)']
for train_benchmark in ordered_train_benchmarks:
    for language in languages:
        full_index.append(f'{train_benchmark} ({language})')
full_columns = []
for eval_benchmark in ordered_eval_benchmarks:
    for language in languages:
        full_columns.append(f'{eval_benchmark} ({language})')
output_matrix = pd.DataFrame(
    data=0.0,
    columns=full_columns,
    index=full_index,
)

# Add baseline (no-tuning) row
for eval_benchmark in ordered_eval_benchmarks:
    for language in languages:
        output_matrix.at['Base Llama 3 70B IT (N/A)', f'{eval_benchmark} ({language})'] = round(baselines_ft[(eval_benchmark, language)]*100, 1)

# Rename index column
output_matrix.index.name = 'Fine-Tuning Dataset (language)'

# Create output matrices for storing means and stds
output_matrix_mean = output_matrix.copy()
output_matrix_std = output_matrix.copy()

# Populate table
for train_benchmark in ordered_train_benchmarks:
    matching_train_benchmark = llm_responses[(llm_responses['Fine-Tuning.Data'] == train_benchmark) &
                                             (llm_responses['Model.Was Fine-Tuned']) &
                                             (pd.isnull(llm_responses['Fine-Tuning.Data Partition.Data Quality']))]
    for train_language in languages:
        matching_train_language = matching_train_benchmark[matching_train_benchmark['Fine-Tuning.Language'] == train_language]
        for eval_benchmark in ordered_eval_benchmarks:
            matching_eval_benchmark = matching_train_language[matching_train_language['Evaluation.Data'] == eval_benchmark]
            for eval_language in languages:
                matching_eval_language = matching_eval_benchmark[matching_eval_benchmark['Evaluation.Target Language'] == eval_language]
                accuracies = []
                for run in runs:
                    relevant_rows = matching_eval_language[matching_eval_language['Evaluation.Trial Number'] == run]
                    accuracy = round(sum(relevant_rows['Evaluation.Model Response Was Correct'].tolist()) /
                                     relevant_rows.shape[0] * 100, 1)
                    accuracies.append(accuracy)
                mean = round(np.mean(accuracies), 1)
                std = round(np.std(accuracies, ddof=1), 1)
                output_matrix_mean.at[f'{train_benchmark} ({train_language})', f'{eval_benchmark} ({eval_language})'] = mean
                output_matrix_std.at[f'{train_benchmark} ({train_language})', f'{eval_benchmark} ({eval_language})'] = std

# Save to CSV
output_matrix_mean.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name_mean}')))
output_matrix_std.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name_std}')))
