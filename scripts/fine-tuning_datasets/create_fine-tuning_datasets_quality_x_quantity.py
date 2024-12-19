import json
import pandas as pd
import os
import sys
import shutil
from collections import defaultdict
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import languages, suffix_map, mmlu_splits, qualities, fine_tuning_eval_benchmarks, baselines_ft, quantities

# Set system message for SFT
SYSTEM_MESSAGE = "You are a chatbot."

with open(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator/quality_buckets.json')), 'r', encoding='utf-8') as fp:
    gpt4o_scores = json.load(fp)

# Set up input and output paths
input_dir_mmlu = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/mmlu_cm_ck_vir'))
input_dir_winogrande = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/winogrande_s'))
output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/fine-tuning_datasets/quality_x_quantity'))
# Clear output dir and recreate it
shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Identify where there are mono-lingual lifts of at least 5% when using the full fine-tuning dataset
threshold = 5
cross_lingual_table = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/tables/tables_A20-A27_means.csv')), index_col=0)
train_benchmarks = ['mmlu-college_medicine', 'winogrande-train_s']
permitted_combinations = {train_benchmark: defaultdict(set) for train_benchmark in train_benchmarks}
for eval_benchmark in fine_tuning_eval_benchmarks:
    for eval_language in languages:
        for train_benchmark in train_benchmarks:
            train_benchmark = train_benchmark.replace("-train_s", "")  # Remove train_s for this step to match baselines_ft
            for train_language in languages:
                # If mono-lingual and lift of at least threshold %...
                if train_language == eval_language and cross_lingual_table.at[f'{train_benchmark} ({train_language})', f'{eval_benchmark} ({eval_language})'] - round(baselines_ft[(eval_benchmark, eval_language)]*100, 1) >= threshold:
                    permitted_combinations[train_benchmark + ('-train_s' if train_benchmark == 'winogrande' else '')][eval_benchmark].add(train_language)  # fix benchmark name mismatch

# Create FT (fine-tuning) datasets split by quality
for language, full_language in languages.items():
    for tb in train_benchmarks:
        for eb in fine_tuning_eval_benchmarks:
            if language not in permitted_combinations[tb][eb]:  # Only make FT datasets where the benchmark and language combinations had a 5% lift or higher
                continue
            for quality in qualities:
                # Get valid IDs for this FT dataset
                relevant_ids = set(gpt4o_scores[language][tb][eb][quality])

                if tb == 'mmlu-college_medicine':
                    # Get full college medicine section
                    mmlu_full = []
                    for split in mmlu_splits:
                        this_mmlu = pd.read_csv(
                            os.path.join(input_dir_mmlu, f'college_medicine_{split}{suffix_map[language]}.csv'),
                            encoding='utf-8', header=None)
                        this_mmlu['split'] = [split] * this_mmlu.shape[0]
                        this_mmlu['index_value'] = this_mmlu.index.tolist()
                        mmlu_full.append(this_mmlu)

                    mmlu_data = pd.concat(mmlu_full)

                    # Create ID column from split and index_value
                    mmlu_data['id'] = mmlu_data['split'] + '-' + mmlu_data['index_value'].astype(str)

                    # Only include rows where ID is in quality bucket
                    mmlu_data = mmlu_data[mmlu_data['id'].astype(str).isin(relevant_ids)].reset_index(drop=True)
                    print(language, tb, eb, quality, mmlu_data.shape[0])

                else:
                    # Get Winogrande train_s set
                    winogrande_data = pd.read_csv(os.path.join(input_dir_winogrande, f'winogrande{suffix_map[language]}.csv'), encoding='utf-8')
                    winogrande_data = winogrande_data[winogrande_data['Split'] == 'train_s'].sort_values(by=['qID']).reset_index(drop=True)  # Ensure consistent order across langs

                    # Only include rows where index is in quality bucket
                    winogrande_data = winogrande_data[winogrande_data.index.astype(str).isin(relevant_ids)].reset_index(drop=True)
                    print(language, tb, eb, quality, winogrande_data.shape[0])

                if tb == 'mmlu-college_medicine':
                    # Build MMLU fine-tuning datasets with prompts
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

                    # Save with proper encoding
                    with open(os.path.join(output_dir, f'language-is_{language}_train-is_{tb}_test-is_{eb}_quality-is_{quality}.jsonl'), 'w', encoding='utf-8') as f:
                        for item in transformed_mmlu:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')

                else:
                    # Build Winogrande fine-tuning datasets with prompts
                    transformed_winogrande = []
                    for index, row in winogrande_data.iterrows():
                        this_prompt = f"""Given the following sentence that is missing a word or a few words (denoted with an underscore) and two options to fill in the missing word or words, output only the number corresponding to the correct option. Do not add any explanation.

Sentence: {winogrande_data.iloc[index][f"{full_language} Sentence"]}
Option1: {winogrande_data.iloc[index][f"{full_language} Option 1"]}
Option2: {winogrande_data.iloc[index][f"{full_language} Option 2"]}
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

                    # Save with proper encoding
                    with open(os.path.join(output_dir, f'language-is_{language}_train-is_{tb}_test-is_{eb}_quality-is_{quality}.jsonl'), 'w', encoding='utf-8') as f:
                        for item in transformed_winogrande:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Load all the saved files and replace them with randomly sampled versions
for filename in sorted(os.listdir(output_dir)):
    full_path = os.path.join(output_dir, filename)
    for quantity in quantities:
        quantity = int(quantity[:-1])  # remove % and convert to int
        # Randomly sample at quantity % size
        this_df = pd.read_json(full_path, lines=True).sample(frac=quantity/100, random_state=42)
        # Save to JSONL with proper encoding
        this_df.to_json(os.path.join(output_dir, filename.replace('_low', f'_low_quantity-percent-is_{quantity}').replace('_high', f'_high_quantity-percent-is_{quantity}')), lines=True, orient='records', force_ascii=False, index=False)

    # Remove original file
    os.remove(full_path)
