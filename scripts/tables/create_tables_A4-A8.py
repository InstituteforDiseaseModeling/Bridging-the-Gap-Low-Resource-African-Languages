import pandas as pd
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import llm_responses, african_languages, languages

table_name = 'tables_A4-A8.csv'

# Load OOTB results
ootb = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/table_A3.csv')), encoding='utf-8').set_index('Model (Evaluation Benchmark)')

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
        full_index.append(f'{model} ({benchmark}) (en -> x)')
    for model in models.values():
        full_index.append(f'{model} ({benchmark}) (en -> x) (difference)')
    for model in models.values():
        full_index.append(f'{model} ({benchmark}) (x -> en)')
    for model in models.values():
        full_index.append(f'{model} ({benchmark}) (x -> en) (difference)')
output_matrix = pd.DataFrame(
    data=0.0,
    columns=languages,
    index=full_index,
)

# Populate table
for benchmark in ordered_benchmarks:
    matching_benchmark = llm_responses[(llm_responses['Evaluation.Data'] == benchmark) &
                                       (~llm_responses['Model.Was Fine-Tuned']) &
                                       (llm_responses['Evaluation.Question ID'].str.contains('-gt-'))]
    for model_id, model in models.items():
        matching_model = matching_benchmark[matching_benchmark['Model.Unique Identifier'] == model_id]
        for language in african_languages:
            matching_language = matching_model[matching_model['Evaluation.Target Language'] == language]
            accuracy = round(sum(matching_language['Evaluation.Model Response Was Correct'].tolist()) / matching_language.shape[0] * 100, 1)
            if benchmark == 'belebele':
                assert matching_language.shape[0] == 900
            elif benchmark == 'mmlu-clinical_knowledge':
                assert matching_language.shape[0] == 265
            elif benchmark == 'mmlu-college_medicine':
                assert matching_language.shape[0] == 173
            elif benchmark == 'mmlu-virology':
                assert matching_language.shape[0] == 166
            else:
                assert matching_language.shape[0] == 1767
            output_matrix.at[f'{model} ({benchmark}) (en -> x)', language] = accuracy
            output_matrix.at[f'{model} ({benchmark}) (en -> x) (difference)', language] = round(ootb.at[f'{model} ({benchmark})', language] - output_matrix.at[f'{model} ({benchmark}) (en -> x)', language], 1)
        output_matrix.at[f'{model} ({benchmark}) (en -> x)', 'en'] = ootb.at[f'{model} ({benchmark})', 'en']
        output_matrix.at[f'{model} ({benchmark}) (en -> x) (difference)', 'en'] = 0.0

    matching_benchmark = llm_responses[(llm_responses['Evaluation.Data'] == benchmark) &
                                       (~llm_responses['Model.Was Fine-Tuned']) &
                                       (llm_responses['Evaluation.Question ID'].str.contains('-bt-'))]
    for model_id, model in models.items():
        matching_model = matching_benchmark[matching_benchmark['Model.Unique Identifier'] == model_id]
        for language in african_languages:
            matching_language = matching_model[matching_model['Evaluation.Source Language'] == language]
            accuracy = round(sum(matching_language['Evaluation.Model Response Was Correct'].tolist()) / matching_language.shape[0] * 100, 1)
            if benchmark == 'belebele':
                assert matching_language.shape[0] == 900
            elif benchmark == 'mmlu-clinical_knowledge':
                assert matching_language.shape[0] == 265
            elif benchmark == 'mmlu-college_medicine':
                assert matching_language.shape[0] == 173
            elif benchmark == 'mmlu-virology':
                assert matching_language.shape[0] == 166
            else:
                assert matching_language.shape[0] == 1767
            output_matrix.at[f'{model} ({benchmark}) (x -> en)', language] = accuracy
            output_matrix.at[f'{model} ({benchmark}) (x -> en) (difference)', language] = round(ootb.at[f'{model} ({benchmark})', language] - output_matrix.at[f'{model} ({benchmark}) (x -> en)', language], 1)
        output_matrix.at[f'{model} ({benchmark}) (x -> en)', 'en'] = ootb.at[f'{model} ({benchmark})', 'en']
        output_matrix.at[f'{model} ({benchmark}) (x -> en) (difference)', 'en'] = 0.0

# Rename index column
output_matrix.index.name = 'Model (Evaluation Benchmark) (Machine Translation Direction) (If Present: Difference Between Performance on Human Translations)'

# Save to CSV
output_matrix.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name}')))
