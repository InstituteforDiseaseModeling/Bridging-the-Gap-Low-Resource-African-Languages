import pandas as pd
import os

# Trial runs for repeated experiments
runs = list(range(1, 4))

# Quality bucket labels
qualities = ['low', 'high']

# Quantity sampling size labels
quantities = ['25%', '50%', '75%', '100%']

# Languages used in our experiments
languages = {
    'en': 'English',
    'af': 'Afrikaans',
    'zu': 'Zulu',
    'xh': 'Xhosa',
    'am': 'Amharic',
    'bm': 'Bambara',
    'ig': 'Igbo',
    'nso': 'Sepedi',
    'sn': 'Shona',
    'st': 'Sesotho',
    'tn': 'Setswana',
    'ts': 'Tsonga',
}

# Mapping of language code to suffix used in data filenames
suffix_map = {
    'en': '',
    'af': '_af',
    'zu': '_zu',
    'xh': '_xh',
    'am': '_am',
    'bm': '_bm',
    'ig': '_ig',
    'nso': '_nso',
    'sn': '_sn',
    'st': '_st',
    'tn': '_tn',
    'ts': '_ts',
}

# African languages only
african_languages = {lang_code: language for lang_code, language in languages.items() if lang_code != 'en'}

# Evaluation benchmarks used in our experiments
benchmarks = ['belebele', 'mmlu-college_medicine', 'mmlu-clinical_knowledge', 'mmlu-virology', 'winogrande']

# Evaluation benchmarks used when fine-tuning
fine_tuning_eval_benchmarks = [benchmark for benchmark in benchmarks if benchmark != 'mmlu-college_medicine']

# Fine-tuning benchmarks used for training
fine_tuning_train_benchmarks = ['mmlu-college_medicine', 'winogrande']

llm_responses_path = os.path.join(os.path.dirname(__file__), '../data/translations_and_llm_responses/5. LLM Responses.csv')

# Load LLM responses
llm_responses = pd.read_csv(llm_responses_path, encoding='utf-8')

# Out-of-the-box baseline for fine-tuning experiments (always done with Llama 3 70B IT)
ootb_ft = llm_responses[(~llm_responses['Model.Was Fine-Tuned']) &
                        (llm_responses['Model.Unique Identifier'] == 'unsloth/llama-3-70b-Instruct-bnb-4bit') &
                        (llm_responses['Evaluation.Translation Approach'].str.contains('Human')) &
                        (llm_responses['Evaluation.Data'] != 'mmlu-college_medicine')]

# Baseline performances by benchmark and language combination on Llama 3 70B IT (the model used for fine-tuning experiments)
baselines_ft = {}
for benchmark in fine_tuning_eval_benchmarks:
    for language in languages:
        relevant_rows = ootb_ft[(ootb_ft['Evaluation.Target Language'] == language) &
                                (ootb_ft['Evaluation.Data'] == benchmark)]
        baselines_ft[(benchmark, language)] = sum(relevant_rows['Evaluation.Model Response Was Correct'].tolist()) / relevant_rows.shape[0]

# Get evaluation data containing all benchmarks used in experiments in one place
evaluation_data_path = os.path.join(os.path.dirname(__file__), '../data/translations_and_llm_responses/6. Evaluation Data.csv')
evaluation_data = pd.read_csv(evaluation_data_path, encoding='utf-8')

# Get Winogrande translation data containing Round 1 and Round 2 versions
winogrande_data_path = os.path.join(os.path.dirname(__file__), '../data/translations_and_llm_responses/2. Winogrande Data.csv')
winogrande_data = pd.read_csv(winogrande_data_path, encoding='utf-8')

# MMLU split names
mmlu_splits = ['dev', 'test', 'val']
