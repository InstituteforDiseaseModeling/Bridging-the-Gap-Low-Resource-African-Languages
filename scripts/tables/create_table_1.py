import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import llm_responses, african_languages

table_name = 'table_1.csv'

# Initialize output CSV structure
output_columns = ['belebele', 'winogrande', 'mmlu-college_medicine', 'mmlu-clinical_knowledge', 'mmlu-virology']
output_index = ['GPT-4o (eng)', 'GPT-4o (afr; avg)', 'GPT-4 (afr; avg)', 'Aya 101 (afr; avg)', 'Llama 3 70B (afr; avg)',
                'Aya 23 (afr; avg)', 'GPT-3.5 (afr; avg)', 'Llama 3 8B (afr; avg)', 'Bloomz 7B (afr; avg)',
                'Phi-3 3B (afr; avg)', 'Random (afr; avg)', 'GPT-4o (eng - (afr; avg))']
output_matrix = pd.DataFrame(
    data=0.0,
    columns=output_columns,
    index=output_index,
)

# Filter LLM responses to get necessary values
# GPT-4o performance on English benchmarks
eng_gpt4o_perf = llm_responses[(llm_responses['Model.Unique Identifier'] == 'gpt-4o-2024-05-13') &
                               (~llm_responses['Model.Was Fine-Tuned']) &
                               (llm_responses['Evaluation.Target Language'] == 'en') &
                               (llm_responses['Evaluation.Translation Approach'].str.contains('Human'))]

# Populate first row of table with English performance by GPT-4o
for benchmark in output_columns:
    run_accuracies = []
    relevant_responses = eng_gpt4o_perf[eng_gpt4o_perf['Evaluation.Data'] == benchmark]
    accuracy = round(sum(relevant_responses['Evaluation.Model Response Was Correct'].tolist()) / relevant_responses.shape[0] * 100, 1)
    output_matrix.at['GPT-4o (eng)', benchmark] = accuracy

# Map model identifiers to in-table names
models = {
    'gpt-4o-2024-05-13': 'GPT-4o (afr; avg)',
    'gpt-4-turbo-2024-04-09': 'GPT-4 (afr; avg)',
    'CohereForAI/aya-101': 'Aya 101 (afr; avg)',
    'unsloth/llama-3-70b-Instruct-bnb-4bit': 'Llama 3 70B (afr; avg)',
    'CohereForAI/aya-23-35B': 'Aya 23 (afr; avg)',
    'gpt-3.5-turbo-1106': 'GPT-3.5 (afr; avg)',
    'unsloth/llama-3-8b-Instruct-bnb-4bit': 'Llama 3 8B (afr; avg)',
    'bigscience/bloomz-7b1': 'Bloomz 7B (afr; avg)',
    'unsloth/Phi-3-mini-4k-instruct-bnb-4bit': 'Phi-3 3B (afr; avg)',
}

# Populate table
for model, index_name in models.items():
    matching_model = llm_responses[(llm_responses['Model.Unique Identifier'] == model) &
                                   (~llm_responses['Model.Was Fine-Tuned']) &
                                   (llm_responses['Evaluation.Translation Approach'].str.contains('Human'))]
    for benchmark in output_columns:
        matching_benchmark = matching_model[matching_model['Evaluation.Data'] == benchmark]
        accuracies = []
        for language in african_languages:
            matching_language = matching_benchmark[matching_benchmark['Evaluation.Target Language'] == language]
            accuracy = round(sum(matching_language['Evaluation.Model Response Was Correct'].tolist()) / matching_language.shape[0] * 100, 1)
            accuracies.append(accuracy)

        mean = round(np.mean(accuracies), 1)
        output_matrix.at[index_name, benchmark] = mean

# Add random baseline
for benchmark in output_columns:
    output_matrix.at['Random (afr; avg)', benchmark] = 25.0 if benchmark != 'winogrande' else 50.0

# Add performance gap
output_matrix.loc['GPT-4o (eng - (afr; avg))'] = round(output_matrix.loc['GPT-4o (eng)'] - output_matrix.loc['GPT-4o (afr; avg)'], 1)

# Rename index column
output_matrix.index.name = 'Model'

# Save to CSV
output_matrix.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name}')))
