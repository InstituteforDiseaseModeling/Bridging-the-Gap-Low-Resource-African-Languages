import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import llm_responses, languages, runs, baselines_ft, quantities, qualities

table_name_prefix = "tables_A21-A24"
os.makedirs(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name_prefix}')), exist_ok=True)

# Identify where there are mono-lingual lifts of at least 5% when using the full fine-tuning dataset
threshold = 5
cross_lingual_table = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/tables/tables_A20-A27_means.csv')), index_col=0)
eval_benchmarks = ['winogrande', 'mmlu-clinical_knowledge', 'mmlu-virology', 'belebele']
train_benchmarks = ['winogrande', 'mmlu-college_medicine']
permitted_languages = defaultdict(list)  # mapping of (train_benchmark, eval_benchmark) to list of languages with lifts above threshold
for eval_benchmark in eval_benchmarks:
    for eval_language in languages:
        for train_benchmark in train_benchmarks:
            for train_language in languages:
                # If mono-lingual and lift of at least threshold %...
                if train_language == eval_language and cross_lingual_table.at[f'{train_benchmark} ({train_language})', f'{eval_benchmark} ({eval_language})'] - round(baselines_ft[(eval_benchmark, eval_language)]*100, 1) >= threshold:
                    permitted_languages[(train_benchmark, eval_benchmark)].append(train_language)

# Create CSV files for each train/eval benchmark combination
for (train_benchmark, eval_benchmark), perm_langs in permitted_languages.items():
    matching_benchmarks = llm_responses[(llm_responses['Fine-Tuning.Data'] == train_benchmark) &
                                        (llm_responses['Evaluation.Data'] == eval_benchmark) &
                                        (~pd.isnull(llm_responses['Fine-Tuning.Data Partition.Data Quality']))]
    for language in perm_langs:
        matching_language = matching_benchmarks[matching_benchmarks['Fine-Tuning.Language'] == language]

        # Initialize output CSV structure
        output_matrix = pd.DataFrame(
            data=0.0,
            columns=['0%'] + quantities,
            index=qualities,
        )
        # Rename index column
        output_matrix.index.name = 'Data Quality'

        output_matrix_mean = output_matrix.copy()
        output_matrix_std = output_matrix.copy()

        # Assign baseline columns
        output_matrix_mean['0%'] = [round(baselines_ft[(eval_benchmark, language)]*100, 1)] * len(qualities)
        output_matrix_std['0%'] = [0.0] * 2

        # Populate the rest of the matrices
        for quality in qualities:
            matching_quality = matching_language[matching_language['Fine-Tuning.Data Partition.Data Quality'] == quality]
            for quantity in quantities:
                matching_quantity = matching_quality[matching_quality['Fine-Tuning.Data Partition.Data Quality.Percent Used'] == quantity]
                accuracies = []
                for run in runs:
                    relevant_rows = matching_quantity[matching_quantity['Evaluation.Trial Number'] == run]
                    accuracy = round(sum(relevant_rows['Evaluation.Model Response Was Correct'].tolist()) /
                                     relevant_rows.shape[0] * 100, 1)
                    accuracies.append(accuracy)
                mean = round(np.mean(accuracies), 1)
                std = round(np.std(accuracies, ddof=1), 1)
                output_matrix_mean.at[quality, quantity] = mean
                output_matrix_std.at[quality, quantity] = std

        # Save to CSV
        output_matrix_mean.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name_prefix}/train-on_{train_benchmark}_eval-on_{eval_benchmark}_in_{language}_means.csv')))
        output_matrix_std.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name_prefix}/train-on_{train_benchmark}_eval-on_{eval_benchmark}_in_{language}_stds.csv')))
