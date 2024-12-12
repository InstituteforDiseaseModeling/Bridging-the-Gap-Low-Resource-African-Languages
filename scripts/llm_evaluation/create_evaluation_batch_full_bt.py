import pandas as pd
import os
import json
import sys
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import african_languages, suffix_map

"""
This script creates a JSONL file that contains every single evaluation question from our available benchmarks
(USING THE MACHINE-BACKTRANSLATED TO ENGLISH VERSIONS; 
note that the -on-{language}- denotes the initial source language prior to backtranslation)
in a format that can be filtered on by custom_id (to select specific subsets/questions to test on) and with 
evaluation prompts/few-shots added. The format is identical to the one used by OpenAI's Batch API: 
https://platform.openai.com/docs/guides/batch/overview
"""

# Remove English from suffixes since machine translation does not apply for English
del suffix_map['en']

# Set evaluation hyperparameters
MAX_TOKENS = 3
TEMPERATURE = 0.7
TOP_P = 0.9

# Set up paths and MMLU section names
sections = [
    'clinical_knowledge',
    'college_medicine',
    'virology',
]
input_dir_mmlu = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/mmlu_cm_ck_vir_bt'))
input_dir_winogrande = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/winogrande_s_bt'))
input_dir_belebele = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/evaluation_benchmarks_afr_release/belebele_bt'))
output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm_evaluation'))

# Ensure the same Winogrande dev questions are used no matter the order of the qIDs in the files and regardless of language,
# since the same dev set qIDs will be used no matter what
winogrande_en = pd.read_csv(os.path.join(input_dir_winogrande.replace("_bt", ""), 'winogrande.csv'))
dev_qIDs = set(winogrande_en[winogrande_en['Split'] == 'dev'].sample(n=5, random_state=42)['qID'].tolist())

# Prepare for creating Belebele questions
belebele_correct_answer_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
belebele_base_prompt = """Given the following passage, query, and answer choices, output only the letter corresponding to the correct answer. Do not add any explanation.
###
Passage:
{passage}
###
Query:
{query}
###
Choices:
(A) {a}
(B) {b}
(C) {c}
(D) {d}
###
Answer:"""

# Placeholder where the model name will be substituted. This should never appear anywhere else.
model_placeholder = '<|MODEL|>'

# Initialize empty JSONL batch
batch = []

# Go through each language and add questions from MMLU, Winogrande, and Belebele
for lang_code, suffix in suffix_map.items():

    if os.path.exists(os.path.join(input_dir_mmlu, f'clinical_knowledge_test{suffix}.csv')):
        # Add MMLU questions (5-shot)
        for section in sections:
            # Question source
            test_section = pd.read_csv(
                os.path.join(
                    input_dir_mmlu,
                    f'{section}_test{suffix}.csv'),
                header=None)

            # 5-shot source
            dev_section = pd.read_csv(
                os.path.join(
                    input_dir_mmlu,
                    f'{section}_dev{suffix}.csv'),
                header=None)

            # Work question into evaluation prompt
            for index, row in test_section.iterrows():
                this_prompt = f"""The following are multiple choice questions (with answers) about {section.replace("_", " ")}.

Question 1: {dev_section.iloc[0, 0]}
A. {dev_section.iloc[0, 1]}
B. {dev_section.iloc[0, 2]}
C. {dev_section.iloc[0, 3]}
D. {dev_section.iloc[0, 4]}
Answer: {dev_section.iloc[0, 5]}

Question 2: {dev_section.iloc[1, 0]}
A. {dev_section.iloc[1, 1]}
B. {dev_section.iloc[1, 2]}
C. {dev_section.iloc[1, 3]}
D. {dev_section.iloc[1, 4]}
Answer: {dev_section.iloc[1, 5]}

Question 3: {dev_section.iloc[2, 0]}
A. {dev_section.iloc[2, 1]}
B. {dev_section.iloc[2, 2]}
C. {dev_section.iloc[2, 3]}
D. {dev_section.iloc[2, 4]}
Answer: {dev_section.iloc[2, 5]}

Question 4: {dev_section.iloc[3, 0]}
A. {dev_section.iloc[3, 1]}
B. {dev_section.iloc[3, 2]}
C. {dev_section.iloc[3, 3]}
D. {dev_section.iloc[3, 4]}
Answer: {dev_section.iloc[3, 5]}

Question 5: {dev_section.iloc[4, 0]}
A. {dev_section.iloc[4, 1]}
B. {dev_section.iloc[4, 2]}
C. {dev_section.iloc[4, 3]}
D. {dev_section.iloc[4, 4]}
Answer: {dev_section.iloc[4, 5]}

Now, given the following question and answer choices, output only the letter corresponding to the correct answer. Do not add any explanation.

Question: {row.iloc[0]}
A. {row.iloc[1]}
B. {row.iloc[2]}
C. {row.iloc[3]}
D. {row.iloc[4]}
Answer:
"""

                batch.append(
                    {"custom_id": f"{model_placeholder}-on-{lang_code}-mmlu-{section}-{index}-answer-{row.iloc[5]}",
                     "method": "POST",
                     "url": "/v1/chat/completions",
                     "body": {"model": model_placeholder,
                              "messages": [{"role": "user",
                                            "content": this_prompt}],
                              "max_tokens": MAX_TOKENS,
                              "temperature": TEMPERATURE,
                              "top_p": TOP_P}}
                )

    else:
        print(f"MMLU not found for {lang_code}. Skipping...")

    if os.path.exists(os.path.join(input_dir_winogrande, f'winogrande{suffix}.csv')):
        # Add Winogrande Questions (5-shot)
        language = african_languages[lang_code]
        this_winogrande = pd.read_csv(os.path.join(input_dir_winogrande, f'winogrande{suffix}.csv'))

        # Get dev (for few-shot learning) and test sets
        this_dev_set = this_winogrande[this_winogrande['qID'].isin(dev_qIDs)].sort_values(by=['qID']).reset_index(drop=True)
        this_test_set = this_winogrande[this_winogrande['Split'] == 'test'].sort_values(by=['qID']).reset_index(drop=True)

        # Work question into evaluation prompt
        for index, row in this_test_set.iterrows():
            this_prompt = f"""The following are sentences that are missing a word or a few words (denoted with an underscore), each followed by two options to fill in the missing word or words. The correct option is given for each sentence:

Sentence 1: {this_dev_set.iloc[0][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[0][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[0][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[0]["Answer"]}

Sentence 2: {this_dev_set.iloc[1][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[1][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[1][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[1]["Answer"]}

Sentence 3: {this_dev_set.iloc[2][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[2][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[2][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[2]["Answer"]}

Sentence 4: {this_dev_set.iloc[3][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[3][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[3][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[3]["Answer"]}

Sentence 5: {this_dev_set.iloc[4][f"{language} Sentence"]}
Option1: {this_dev_set.iloc[4][f"{language} Option 1"]}
Option2: {this_dev_set.iloc[4][f"{language} Option 2"]}
Correct Option: {this_dev_set.iloc[4]["Answer"]}

Now, given the following sentence and options, output only the number corresponding to the correct option. Do not add any explanation.

Sentence: {this_test_set.iloc[index][f"{language} Sentence"]}
Option1: {this_test_set.iloc[index][f"{language} Option 1"]}
Option2: {this_test_set.iloc[index][f"{language} Option 2"]}
Correct Option:
"""

            batch.append(
                {"custom_id": f"{model_placeholder}-on-{lang_code}-winogrande-{index}-answer-{this_test_set.iloc[index]['qID'][-1]}",
                 "method": "POST",
                 "url": "/v1/chat/completions",
                 "body": {"model": model_placeholder,
                          "messages": [{"role": "user",
                                        "content": this_prompt}],
                          "max_tokens": MAX_TOKENS,
                          "temperature": TEMPERATURE,
                          "top_p": TOP_P}}
            )

    else:
        print(f"Winogrande not found for {lang_code}. Skipping...")

    # Add Belebele questions (0-shot)
    with open(os.path.join(input_dir_belebele, f'belebele{suffix}.jsonl'), 'r', encoding='utf-8') as fp:
        this_belebele = pd.read_json(fp, lines=True)
        this_belebele = this_belebele.sort_values(by=['link', 'question_number']).reset_index(drop=True)  # Ensure consistent order across langs

    # Work question into evaluation prompt
    for index, row in this_belebele.iterrows():
        this_prompt = belebele_base_prompt \
            .replace("{passage}", row['flores_passage']) \
            .replace("{query}", row['question']) \
            .replace("{a}", row['mc_answer1']) \
            .replace("{b}", row['mc_answer2']) \
            .replace("{c}", row['mc_answer3']) \
            .replace("{d}", row['mc_answer4'])
        batch.append(
            {
                "custom_id": f"{model_placeholder}-on-{lang_code}-belebele-{index}-answer-{belebele_correct_answer_map[row['correct_answer_num']]}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model_placeholder,
                         "messages": [{"role": "user",
                                       "content": this_prompt}],
                         "max_tokens": MAX_TOKENS,
                         "temperature": TEMPERATURE,
                         "top_p": TOP_P}}
        )

# Write the batch to a JSONL file, ensuring that special characters remain
with open(os.path.join(output_dir, 'evaluation_batch_full_bt.jsonl'), 'w', encoding='utf-8') as f:
    for item in batch:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
