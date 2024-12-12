import json
import pandas as pd
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import languages, suffix_map, mmlu_splits

"""
This script creates JSONL fine-tuning dataset files that can be used to fine-tune LLMs using supervised fine-tuning.
The format is identical to the one used by OpenAI's Fine-Tuning API: 
https://platform.openai.com/docs/guides/fine-tuning/overview
"""

# Set system message for SFT
SYSTEM_MESSAGE = "You are a chatbot."

# Set up input and output paths
input_dir_mmlu = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/mmlu_cm_ck_vir'))
input_dir_winogrande = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/winogrande_s'))
output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/fine-tuning_datasets/full'))
os.makedirs(output_dir, exist_ok=True)

# Iterate over each language
for lang_code, suffix in suffix_map.items():
    language = languages[lang_code]

    # Get entire college medicine section
    mmlu_full = []
    for split in mmlu_splits:
        mmlu_full.append(pd.read_csv(os.path.join(input_dir_mmlu, f'college_medicine_{split}{suffix}.csv'), header=None))
    mmlu_data = pd.concat(mmlu_full)

    # Get Winogrande train split
    winogrande_data = pd.read_csv(os.path.join(input_dir_winogrande, f'winogrande{suffix}.csv'))
    winogrande_data = winogrande_data[winogrande_data['Split'] == 'train_s'].sort_values(by=['qID']).reset_index(drop=True)  # Ensure consistent order across langs

    # Construct MMLU supervised fine-tuning dataset
    transformed_mmlu = []
    for index, row in mmlu_data.iterrows():
        this_prompt = f"""Given the following question and answer choices, output only the letter corresponding to the correct answer. Do not add any explanation.

Question: {row.iloc[0]}
A. {row.iloc[1]}
B. {row.iloc[2]}
C. {row.iloc[3]}
D. {row.iloc[4]}
Answer:
"""
        transformed_entry = {
            "messages": [
                {"role": "system",
                 "content": SYSTEM_MESSAGE},
                {"role": "user", "content": this_prompt},
                {"role": "assistant", "content": row.iloc[5]}
            ]
        }
        transformed_mmlu.append(transformed_entry)

    # Write while ensuring proper encoding
    with open(os.path.join(output_dir, f'mmlu-college_medicine_{lang_code}.jsonl'), 'w', encoding='utf-8') as f:
        for item in transformed_mmlu:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Construct Winogrande supervised fine-tuning dataset
    transformed_winogrande = []
    for index, row in winogrande_data.iterrows():
        this_prompt = f"""Given the following sentence that is missing a word or a few words (denoted with an underscore) and two options to fill in the missing word or words, output only the number corresponding to the correct option. Do not add any explanation.

Sentence: {winogrande_data.iloc[index][f"{language} Sentence"]}
Option1: {winogrande_data.iloc[index][f"{language} Option 1"]}
Option2: {winogrande_data.iloc[index][f"{language} Option 2"]}
Correct Option:
"""
        transformed_entry = {
            "messages": [
                {"role": "system",
                 "content": SYSTEM_MESSAGE},
                {"role": "user", "content": this_prompt},
                {"role": "assistant", "content": str(int(winogrande_data.iloc[index]['Answer']))}
            ]
        }
        transformed_winogrande.append(transformed_entry)

    # Write while ensuring proper encoding
    with open(os.path.join(output_dir, f'winogrande-train_s_{lang_code}.jsonl'), 'w', encoding='utf-8') as f:
        for item in transformed_winogrande:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
