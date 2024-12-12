#!/usr/bin/env python
# coding: utf-8

# # BLOOMZ 7b1 Out-of-the-Box Evaluation
# This script contains the code to evaluate BLOOMZ 7b1 out-of-the-box on our evaluation benchmarks.

# ## Installation

# In[ ]:


# Install Python packages
get_ipython().system('pip install transformers torch accelerate')
get_ipython().system('pip install -i https://pypi.org/simple/ bitsandbytes')
get_ipython().system('pip install flash-attn --no-build-isolation')
get_ipython().system('pip install pandas tqdm')


# In[ ]:


import torch
torch.version.cuda, torch.__version__


# In[ ]:


# Check installation status
get_ipython().system('nvcc')
get_ipython().system('python -m xformers.info')
get_ipython().system('python -m bitsandbytes')


# In[ ]:


# Import helpers
import sys
import os
# sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
sys.path.append('../../utils')
from useful_variables import languages
from useful_functions import check_mc_answer, check_winogrande_answer


# ## Load model

# In[ ]:


# Download and load BLOOMZ 7b1 (with some optimizations)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "bigscience/bloomz-7b1"
output_name = model_id[model_id.rfind('/')+1:]  # Take just the part after the '/'

quant_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# ## Inference

# In[ ]:


# Define inference function that accepts a row in an OpenAI Batch API-formatted JSONL and produces a response
def infer(jsonl_row):
    messages = jsonl_row['body']['messages']
    input_ids = tokenizer.encode(
        messages[0]['content'],
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=jsonl_row['body']['max_tokens'],
        do_sample=False,
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

test_row = {"custom_id": "<|MODEL|>-on-en-mmlu-clinical_knowledge-0-answer-A", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "<|MODEL|>", "messages": [{"role": "user", "content": "The following are multiple choice questions (with answers) about clinical knowledge.\n\nQuestion 1: The energy for all forms of muscle contraction is provided by:\nA. ATP.\nB. ADP.\nC. phosphocreatine.\nD. oxidative phosphorylation.\nAnswer: A\n\nQuestion 2: What is the difference between a male and a female catheter?\nA. Male and female catheters are different colours.\nB. Male catheters are longer than female catheters.\nC. Male catheters are bigger than female catheters.\nD. Female catheters are longer than male catheters.\nAnswer: B\n\nQuestion 3: In the assessment of the hand function which of the following is true?\nA. Abduction of the thumb is supplied by spinal root T2\nB. Opposition of the thumb by opponens policis is supplied by spinal root T1\nC. Finger adduction is supplied by the median nerve\nD. Finger abduction is mediated by the palmar interossei\nAnswer: B\n\nQuestion 4: How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: C\n\nQuestion 5: Glycolysis is the name given to the pathway involving the conversion of:\nA. glycogen to glucose-1-phosphate.\nB. glycogen or glucose to fructose.\nC. glycogen or glucose to pyruvate or lactate.\nD. glycogen or glucose to pyruvate or acetyl CoA.\nAnswer: C\n\nNow, given the following question and answer choices, output only the letter corresponding to the correct answer. Do not add any explanation.\n\nQuestion: What size of cannula would you use in a patient who needed a rapid blood transfusion (as of 2020 medical knowledge)?\nA. 18 gauge.\nB. 20 gauge.\nC. 22 gauge.\nD. 24 gauge.\nAnswer:\n"}], "max_tokens": 3, "temperature": 0.7, "top_p": 0.9}}
infer(test_row)


# ## Evaluation

# In[ ]:


# Run on all questions
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
import json

output_folder = '../../results/out-of-the-box/'

# Infer on all prompts
generations_map = {}
with open(f'../../results/llm_evaluation/evaluation_batch_full.jsonl', 'r', encoding='utf-8') as fp:
    all_prompts_jsonl = pd.read_json(fp, lines=True)

for index, row in tqdm(all_prompts_jsonl.iterrows(), total=all_prompts_jsonl.shape[0]):
    try:
        generations_map[row['custom_id'].replace('<|MODEL|>', output_name)] = infer({'custom_id': row['custom_id'], 'body': row['body']})
    except Exception as e:
        print(f"Exception \"{e}\" occurred on \"{row['custom_id'].replace('<|MODEL|>', output_name)}\". Skipping...")

# Save generations so they never have to be run again
with open(f'../../results/out-of-the-box/generations_{output_name}.json', 'w') as fp:
    json.dump(generations_map, fp, indent=2)


# In[ ]:


import re
import json
import pandas as pd

with open(f'../../results/out-of-the-box/generations_{output_name}.json', 'r') as fp:
    generations_map = json.load(fp)

# Get college medicine performance (including more sections means taking the average perf. across the sections
sections = [
    'college_medicine',
]

# initialize matrix of results
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    for section in sections:

        # Construct the pattern
        pattern = re.compile(rf".*-on-{lang}-mmlu-{section}.*")

        # Filter keys by pattern
        matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

        for (c_id, gen) in matching_generations:
            # Check correctness
            if check_mc_answer(c_id, gen):
                total_score += 1
            q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

# Save to CSV
matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_mmlu-college_medicine.csv'))

# Get clinical knowledge performance (including more sections means taking the average perf. across the sections
sections = [
    'clinical_knowledge',
]

# initialize matrix of results
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    for section in sections:

        # Construct the pattern
        pattern = re.compile(rf".*-on-{lang}-mmlu-{section}.*")

        # Filter keys by pattern
        matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

        for (c_id, gen) in matching_generations:
            # Check correctness
            if check_mc_answer(c_id, gen):
                total_score += 1
            q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

# Save to CSV
matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_mmlu-clinical_knowledge.csv'))

# Get virology performance (including more sections means taking the average perf. across the sections
sections = [
    'virology',
]

# initialize matrix of results
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    for section in sections:

        # Construct the pattern
        pattern = re.compile(rf".*-on-{lang}-mmlu-{section}.*")

        # Filter keys by pattern
        matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

        for (c_id, gen) in matching_generations:
            # Check correctness
            if check_mc_answer(c_id, gen):
                total_score += 1
            q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

# Save to CSV
matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_mmlu-virology.csv'))

# Get Winogrande performance
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    # Construct the pattern
    pattern = re.compile(rf".*-on-{lang}-winogrande.*")

    # Filter keys by pattern
    matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

    for (c_id, gen) in matching_generations:
        # Check correctness
        if check_winogrande_answer(c_id, gen):
            total_score += 1
        q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

# Save to CSV
matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_winogrande.csv'))

# Get Belebele performance
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    # Construct the pattern
    pattern = re.compile(rf".*-on-{lang}-belebele.*")

    # Filter keys by pattern
    matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

    for (c_id, gen) in matching_generations:
        # Check correctness
        if check_mc_answer(c_id, gen):
            total_score += 1
        q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_belebele.csv'))


# In[ ]:





# In[ ]:




