import pandas as pd
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import llm_responses, languages

table_name = 'table_A3.csv'

# Initialize output CSV structure
ordered_benchmarks = ['winogrande', 'mmlu-college_medicine', 'mmlu-clinical_knowledge', 'mmlu-virology', 'belebele']
models = {
    'gpt-4o-2024-05-13': 'GPT-4o',
    'gpt-4-turbo-2024-04-09': 'GPT-4',
    'gpt-3.5-turbo-1106': 'GPT-3.5',
    'unsloth/llama-3-70b-Instruct-bnb-4bit': 'Llama 3 70B IT',
    'unsloth/llama-3-8b-Instruct-bnb-4bit': 'Llama 3 8B IT',
    'unsloth/Phi-3-mini-4k-instruct-bnb-4bit': 'Phi 3 Mini 4K IT',
    'CohereForAI/aya-23-35B': 'Aya 23 35B',
    'CohereForAI/aya-101': 'Aya 101',
    'bigscience/bloomz-7b1': 'BLOOMZ 7b1',
}
full_index = []
for benchmark in ordered_benchmarks:
    for model in models.values():
        full_index.append(f'{model} ({benchmark})')
output_matrix = pd.DataFrame(
    data=0.0,
    columns=languages,
    index=full_index,
)

# Populate table
for benchmark in ordered_benchmarks:
    matching_benchmark = llm_responses[(llm_responses['Evaluation.Data'] == benchmark) &
                                       (~llm_responses['Model.Was Fine-Tuned']) &
                                       (llm_responses['Evaluation.Translation Approach'].str.contains('Human'))]
    for model_id, model in models.items():
        matching_model = matching_benchmark[matching_benchmark['Model.Unique Identifier'] == model_id]
        for language in languages:
            matching_language = matching_model[matching_model['Evaluation.Target Language'] == language]
            accuracy = round(sum(matching_language['Evaluation.Model Response Was Correct'].tolist()) / matching_language.shape[0] * 100, 1)
            # if benchmark == 'belebele':
            #     assert matching_language.shape[0] == 900
            # elif benchmark == 'mmlu-clinical_knowledge':
            #     assert matching_language.shape[0] == 265
            # elif benchmark == 'mmlu-college_medicine':
            #     assert matching_language.shape[0] == 173
            # elif benchmark == 'mmlu-virology':
            #     assert matching_language.shape[0] == 166
            # else:
            #     assert matching_language.shape[0] == 1767
            output_matrix.at[f'{model} ({benchmark})', language] = accuracy

# Rename index column
output_matrix.index.name = 'Model (Evaluation Benchmark)'

# Save to CSV
output_matrix.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name}')))
