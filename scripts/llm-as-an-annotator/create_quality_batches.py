import json
import pandas as pd
import os
import sys
from collections import defaultdict
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import languages, suffix_map, baselines_ft, mmlu_splits

# Set up generation hyperparameters
SYSTEM_MESSAGE = "You are a chatbot."
MODEL = 'gpt-4o'
MAX_TOKENS = 3
TEMPERATURE = 0.7
TOP_P = 0.9
TRIALS = 3

# Set up input/output dirs
input_dir_mmlu = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/mmlu_cm_ck_vir'))
input_dir_winogrande = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/winogrande_s'))
output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator'))
os.makedirs(output_dir, exist_ok=True)

# Identify where there are mono-lingual lifts of at least 5% when using the full fine-tuning dataset
threshold = 5
cross_lingual_table = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/tables/tables_A20-A27_means.csv')), index_col=0)
eval_benchmarks = ['winogrande', 'mmlu-clinical_knowledge', 'mmlu-virology', 'belebele']
train_benchmarks = ['winogrande', 'mmlu-college_medicine']
permitted_languages = defaultdict(set)  # mapping of (train_benchmark, eval_benchmark) to set of languages with lifts above threshold
for eval_benchmark in eval_benchmarks:
    for eval_language in languages:
        for train_benchmark in train_benchmarks:
            for train_language in languages:
                # If mono-lingual and lift of at least threshold %...
                if train_language == eval_language and cross_lingual_table.at[f'{train_benchmark} ({train_language})', f'{eval_benchmark} ({eval_language})'] - round(baselines_ft[(eval_benchmark, eval_language)]*100, 1) >= threshold:
                    permitted_languages[(train_benchmark, eval_benchmark)].add(train_language)

# Assign languages where there were sufficient lifts
mmlu_on_mmlu_ck = permitted_languages[('mmlu-college_medicine', 'mmlu-clinical_knowledge')]

winogrande_on_winogrande = permitted_languages[('winogrande', 'winogrande')]

mmlu_on_winogrande = permitted_languages[('mmlu-college_medicine', 'winogrande')]

winogrande_on_mmlu_ck = permitted_languages[('winogrande', 'mmlu-clinical_knowledge')]

mmlu_on_mmlu_v = permitted_languages[('mmlu-college_medicine', 'mmlu-virology')]

winogrande_on_mmlu_v = permitted_languages[('winogrande', 'mmlu-virology')]

mmlu_on_belebele = permitted_languages[('mmlu-college_medicine', 'belebele')]

winogrande_on_belebele = permitted_languages[('winogrande', 'belebele')]

# Train benchmarks
benchmarks = ['mmlu', 'winogrande']

# Initialize quality assessment batch
batch = []

# Iterate over languages
for language, full_language in languages.items():
    # Iterate over training benchmarks
    for benchmark in benchmarks:
        if benchmark == 'mmlu':
            # Get full college medicine section
            mmlu_full = []
            for split in mmlu_splits:
                this_mmlu = pd.read_csv(os.path.join(input_dir_mmlu, f'college_medicine_{split}{suffix_map[language]}.csv'), encoding='utf-8', header=None)
                this_mmlu['split'] = [split] * this_mmlu.shape[0]
                this_mmlu['index_value'] = this_mmlu.index.tolist()
                mmlu_full.append(this_mmlu)

            mmlu_data = pd.concat(mmlu_full)

        # Get winogrande train split
        else:
            winogrande_data = pd.read_csv(os.path.join(input_dir_winogrande, f'winogrande{suffix_map[language]}.csv'), encoding='utf-8')
            winogrande_data = winogrande_data[winogrande_data['Split'] == 'train_s'].sort_values(by=['qID']).reset_index(drop=True)  # Ensure consistent order across langs

        if benchmark == 'mmlu':
            for index, row in mmlu_data.iterrows():
                # Get the actual fine-tuning dataset row
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
    
                # Prompt TRIALS many times
                for i in range(TRIALS):
                    # If the language had a lift of at least threshold on the given train/eval benchmark combination, add prompts to do quality assessments using the fine-tuning dataset row
                    if language in mmlu_on_mmlu_ck:
                        batch.append(
                            {"custom_id": f"{MODEL}-on-{language}-{benchmark}-college_medicine-{row['split']}-{row['index_value']}_vs_mmlu-clinical_knowledge_{i}",
                             "method": "POST",
                             "url": "/v1/chat/completions",
                             "body": {"model": MODEL,
                                      "messages": [{"role": "user",
                                                    "content": f"I am trying to curate a supervised fine-tuning dataset that will be used to improve the performance of my LLM on the clinical knowledge section of MMLU in {full_language} (I have translated MMLU from English into many other languages). Please rate the following fine-tuning dataset sample on a scale of 1-10. Your rating should reflect the anticipated usefulness of the sample if included in the fine-tuning dataset for improving my LLM's performance on the clinical knowledge section of MMLU (with 1 implying least useful and 10 implying most useful). Do not include any additional explanation; just provide the number rating:\n\n{transformed_entry['messages']}"}],
                                      "max_tokens": MAX_TOKENS,
                                      "temperature": TEMPERATURE,
                                      "top_p": TOP_P}}
                        )

                    if language in mmlu_on_mmlu_v:
                        batch.append(
                            {"custom_id": f"{MODEL}-on-{language}-{benchmark}-college_medicine-{row['split']}-{row['index_value']}_vs_mmlu-virology_{i}",
                             "method": "POST",
                             "url": "/v1/chat/completions",
                             "body": {"model": MODEL,
                                      "messages": [{"role": "user",
                                                    "content": f"I am trying to curate a supervised fine-tuning dataset that will be used to improve the performance of my LLM on the virology section of MMLU in {full_language} (I have translated MMLU from English into many other languages). Please rate the following fine-tuning dataset sample on a scale of 1-10. Your rating should reflect the anticipated usefulness of the sample if included in the fine-tuning dataset for improving my LLM's performance on the virology section of MMLU (with 1 implying least useful and 10 implying most useful). Do not include any additional explanation; just provide the number rating:\n\n{transformed_entry['messages']}"}],
                                      "max_tokens": MAX_TOKENS,
                                      "temperature": TEMPERATURE,
                                      "top_p": TOP_P}}
                        )

                    if language in mmlu_on_winogrande:
                        batch.append(
                            {"custom_id": f"{MODEL}-on-{language}-{benchmark}-college_medicine-{row['split']}-{row['index_value']}_vs_winogrande_{i}",
                             "method": "POST",
                             "url": "/v1/chat/completions",
                             "body": {"model": MODEL,
                                      "messages": [{"role": "user",
                                                    "content": f"I am trying to curate a supervised fine-tuning dataset that will be used to improve the performance of my LLM on Winogrande in {full_language} (I have translated Winogrande from English into many other languages). Please rate the following fine-tuning dataset sample on a scale of 1-10. Your rating should reflect the anticipated usefulness of the sample if included in the fine-tuning dataset for improving my LLM's performance on Winogrande (with 1 implying least useful and 10 implying most useful). Do not include any additional explanation; just provide the number rating:\n\n{transformed_entry['messages']}"}],
                                      "max_tokens": MAX_TOKENS,
                                      "temperature": TEMPERATURE,
                                      "top_p": TOP_P}}
                        )

                    if language in mmlu_on_belebele:
                        batch.append(
                            {"custom_id": f"{MODEL}-on-{language}-{benchmark}-college_medicine-{row['split']}-{row['index_value']}_vs_belebele_{i}",
                             "method": "POST",
                             "url": "/v1/chat/completions",
                             "body": {"model": MODEL,
                                      "messages": [{"role": "user",
                                                    "content": f"I am trying to curate a supervised fine-tuning dataset that will be used to improve the performance of my LLM on Belebele, a diversity-focused evaluation benchmark designed to assess the performance of large language models (LLMs) across various African languages, in {full_language}. Please rate the following fine-tuning dataset sample on a scale of 1-10. Your rating should reflect the anticipated usefulness of the sample if included in the fine-tuning dataset for improving my LLM's performance on Belebele (with 1 implying least useful and 10 implying most useful). Do not include any additional explanation; just provide the number rating:\n\n{transformed_entry['messages']}"}],
                                      "max_tokens": MAX_TOKENS,
                                      "temperature": TEMPERATURE,
                                      "top_p": TOP_P}}
                        )

        else:
            for index, row in winogrande_data.iterrows():
                # Get the actual fine-tuning dataset row
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

                # Prompt TRIALS many times
                for i in range(TRIALS):
                    # If the language had a lift of at least threshold on the given train/eval benchmark combination, add prompts to do quality assessments using the fine-tuning dataset row
                    if language in winogrande_on_mmlu_ck:
                        batch.append(
                            {
                                "custom_id": f"{MODEL}-on-{language}-{benchmark}-train_s-{index}_vs_mmlu-clinical_knowledge_{i}",
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": {"model": MODEL,
                                         "messages": [{"role": "user",
                                                       "content": f"I am trying to curate a supervised fine-tuning dataset that will be used to improve the performance of my LLM on the clinical knowledge section of MMLU in {full_language} (I have translated MMLU from English into many other languages). Please rate the following fine-tuning dataset sample on a scale of 1-10. Your rating should reflect the anticipated usefulness of the sample if included in the fine-tuning dataset for improving my LLM's performance on the clinical knowledge section of MMLU (with 1 implying least useful and 10 implying most useful). Do not include any additional explanation; just provide the number rating:\n\n{transformed_entry['messages']}"}],
                                         "max_tokens": MAX_TOKENS,
                                         "temperature": TEMPERATURE,
                                         "top_p": TOP_P}}
                        )

                    if language in winogrande_on_mmlu_v:
                        batch.append(
                            {"custom_id": f"{MODEL}-on-{language}-{benchmark}-train_s-{index}_vs_mmlu-virology_{i}",
                             "method": "POST",
                             "url": "/v1/chat/completions",
                             "body": {"model": MODEL,
                                      "messages": [{"role": "user",
                                                    "content": f"I am trying to curate a supervised fine-tuning dataset that will be used to improve the performance of my LLM on the virology section of MMLU in {full_language} (I have translated MMLU from English into many other languages). Please rate the following fine-tuning dataset sample on a scale of 1-10. Your rating should reflect the anticipated usefulness of the sample if included in the fine-tuning dataset for improving my LLM's performance on the virology section of MMLU (with 1 implying least useful and 10 implying most useful). Do not include any additional explanation; just provide the number rating:\n\n{transformed_entry['messages']}"}],
                                      "max_tokens": MAX_TOKENS,
                                      "temperature": TEMPERATURE,
                                      "top_p": TOP_P}}
                        )

                    if language in winogrande_on_winogrande:
                        batch.append(
                            {"custom_id": f"{MODEL}-on-{language}-{benchmark}-train_s-{index}_vs_winogrande_{i}",
                             "method": "POST",
                             "url": "/v1/chat/completions",
                             "body": {"model": MODEL,
                                      "messages": [{"role": "user",
                                                    "content": f"I am trying to curate a supervised fine-tuning dataset that will be used to improve the performance of my LLM on Winogrande in {full_language} (I have translated Winogrande from English into many other languages). Please rate the following fine-tuning dataset sample on a scale of 1-10. Your rating should reflect the anticipated usefulness of the sample if included in the fine-tuning dataset for improving my LLM's performance on Winogrande (with 1 implying least useful and 10 implying most useful). Do not include any additional explanation; just provide the number rating:\n\n{transformed_entry['messages']}"}],
                                      "max_tokens": MAX_TOKENS,
                                      "temperature": TEMPERATURE,
                                      "top_p": TOP_P}}
                        )

                    if language in winogrande_on_belebele:
                        batch.append(
                            {"custom_id": f"{MODEL}-on-{language}-{benchmark}-train_s-{index}_vs_belebele_{i}",
                             "method": "POST",
                             "url": "/v1/chat/completions",
                             "body": {"model": MODEL,
                                      "messages": [{"role": "user",
                                                    "content": f"I am trying to curate a supervised fine-tuning dataset that will be used to improve the performance of my LLM on Belebele, a diversity-focused evaluation benchmark designed to assess the performance of large language models (LLMs) across various African languages, in {full_language}. Please rate the following fine-tuning dataset sample on a scale of 1-10. Your rating should reflect the anticipated usefulness of the sample if included in the fine-tuning dataset for improving my LLM's performance on Belebele (with 1 implying least useful and 10 implying most useful). Do not include any additional explanation; just provide the number rating:\n\n{transformed_entry['messages']}"}],
                                      "max_tokens": MAX_TOKENS,
                                      "temperature": TEMPERATURE,
                                      "top_p": TOP_P}}
                        )

split_index = len(batch) // 2
# Write to JSONL file with proper encoding (split into two pieces to abide by 50,000 requests per batch limit on OpenAI's Batch API)
with open(os.path.join(output_dir, 'gpt-4o_quality_batch_0.jsonl'), 'w', encoding='utf-8') as f:
    for item in batch[:split_index]:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open(os.path.join(output_dir, 'gpt-4o_quality_batch_1.jsonl'), 'w', encoding='utf-8') as f:
    for item in batch[split_index:]:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
