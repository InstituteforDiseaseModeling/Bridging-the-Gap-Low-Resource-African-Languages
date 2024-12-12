#!/usr/bin/env python
# coding: utf-8

# # Fine-Tuning Llama 3 70B IT
# This script fine-tunes and evaluates Llama 3 70B IT on a desired subset of fine-tuning datasets from the selection available in the repository (under "results/fine-tuning_datasets"), allowing for the recreation of results from the paper to be split up across multiple compute instances to speed things up.

# ## Setup
# You must select files you want to fine-tune and evaluate one-after-the-other on this compute instance.

# In[ ]:


import os
# Define functions that facilitate the selection process
def list_files_recursive(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return sorted(file_list)

def display_files(file_list):
    for i, file_path in enumerate(file_list):
        print(f"{i + 1}: {os.path.basename(file_path)}")

def parse_selection(selection_str, num_files):
    selected_indices = set()
    try:
        parts = selection_str.split(',')
        for part in parts:
            if '-' in part:
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                if start > end:
                    start, end = end, start
                selected_indices.update(range(start-1, end))
            else:
                index = int(part.strip()) - 1
                selected_indices.add(index)
    except ValueError:
        print("Invalid input. Please enter numbers and ranges correctly.")
    
    # Filter out indices that are out of range
    selected_indices = sorted(i for i in selected_indices if 0 <= i < num_files)
    return selected_indices

def get_user_selection(file_list):
    selected_files = []
    num_files = len(file_list)
    selections = input("Enter the numbers or ranges of the files you want to select (e.g., 1-3,5,7-9): ")
    selected_indices = parse_selection(selections, num_files)
    
    for index in selected_indices:
        selected_files.append(file_list[index])
    
    return selected_files


# In[ ]:


evaluation_runs = 3  # Number of evaluation runs/trials to do after each model is fine-tuned
directory = '../../results/fine-tuning_datasets'
file_list = list_files_recursive(directory)
display_files(file_list)

if file_list:
    selected_files = get_user_selection(file_list)
    print("\nYou selected the following files:")
    for file in selected_files:
        print(os.path.basename(file))
else:
    selected_files = []
    print("No files found in the specified directory.")


# ## Installation

# In[ ]:


# Install Python packages
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton --index-url https://download.pytorch.org/whl/cu121')
get_ipython().system('pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"')
get_ipython().system('pip install pandas')
get_ipython().system('pip install tqdm')


# In[ ]:


from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import re
import gc
# Import helpers
import sys
import os
import json
# sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
sys.path.append('../../utils')
from useful_variables import languages, fine_tuning_eval_benchmarks
from useful_functions import check_mc_answer, check_winogrande_answer
evaluation_batch_path = '../../results/llm_evaluation/evaluation_batch_full.jsonl'


# ## Fine-Tuning Evaluation Loop
# This is the main loop that will iterate over the fine-tuning datasets and fine-tune Llama 3 70B IT on them, followed by evaluation in necessary benchmarks determined automatically from the fine-tuning dataset.

# In[ ]:


# Iterate over files
for fine_tuning_path in selected_files:
    fine_tune_name = os.path.basename(fine_tuning_path)[:-6]  # create name that is distinct using FT dataset name
    # Get language of tuning (and evaluation if quality/quantity is involved)
    if 'language-is' in fine_tune_name:
        this_language = fine_tune_name.split('_')[1]
    else:
        this_language = fine_tune_name.split('_')[-1]

    # Get target eval benchmark if applicable
    if 'test-is' in fine_tune_name:
        this_eval_benchmark = fine_tune_name[fine_tune_name.find('test-is_')+len('test-is_'):fine_tune_name.find('_quality-is_')]
    else:
        this_eval_benchmark = None  # We will evaluate on all fine-tuning eval benchmarks in this case
    print(f"Now fine-tuning on \"{fine_tune_name}\", which is in language \"{this_language}\"...")

    # Load model fresh
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-70b-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Convert to PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 8,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 42,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # Load dataset
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3",
    )
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    train_df = pd.read_json(fine_tuning_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    train_ds = train_ds.map(formatting_prompts_func, batched = True,)

    # Fine-Tune Llama 3 70B IT with SFT
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 10,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            # gradient_accumulation_steps = 4,
            # warmup_steps = 5,
            # max_steps = 60,
            num_train_epochs = 3,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs",
        ),
    )
    # Print memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Actually fine-tune
    trainer_stats = trainer.train()

    print("Tuning finished!")

    # Print memory stats after tuning
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Test inference
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3",
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    messages = [
        {"role": "user", "content": "What is 2+2?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(input_ids = inputs, max_new_tokens = 512, use_cache = True, temperature = 0.7, top_p = 0.9, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.batch_decode(outputs))  # Should say something like "2+2=4"

    # Set output name prefix
    output_name = f"llama3-70b-instruct_{fine_tune_name}"

    # Define inference function that accepts a row in an OpenAI Batch API-formatted JSONL and produces a response
    def infer(jsonl_row):
        messages = jsonl_row['body']['messages']
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    
        outputs = model.generate(
            input_ids,
            max_new_tokens=jsonl_row['body']['max_tokens'],
            eos_token_id=terminators,
            do_sample=True,
            temperature=jsonl_row['body']['temperature'],
            top_p=jsonl_row['body']['top_p'],
            pad_token_id=tokenizer.eos_token_id,
        )
    
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response

    # Run small inference function test
    test_row = {"custom_id": "<|MODEL|>-on-en-mmlu-clinical_knowledge-0-answer-A", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "<|MODEL|>", "messages": [{"role": "user", "content": "The following are multiple choice questions (with answers) about clinical knowledge.\n\nQuestion 1: The energy for all forms of muscle contraction is provided by:\nA. ATP.\nB. ADP.\nC. phosphocreatine.\nD. oxidative phosphorylation.\nAnswer: A\n\nQuestion 2: What is the difference between a male and a female catheter?\nA. Male and female catheters are different colours.\nB. Male catheters are longer than female catheters.\nC. Male catheters are bigger than female catheters.\nD. Female catheters are longer than male catheters.\nAnswer: B\n\nQuestion 3: In the assessment of the hand function which of the following is true?\nA. Abduction of the thumb is supplied by spinal root T2\nB. Opposition of the thumb by opponens policis is supplied by spinal root T1\nC. Finger adduction is supplied by the median nerve\nD. Finger abduction is mediated by the palmar interossei\nAnswer: B\n\nQuestion 4: How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: C\n\nQuestion 5: Glycolysis is the name given to the pathway involving the conversion of:\nA. glycogen to glucose-1-phosphate.\nB. glycogen or glucose to fructose.\nC. glycogen or glucose to pyruvate or lactate.\nD. glycogen or glucose to pyruvate or acetyl CoA.\nAnswer: C\n\nNow, given the following question and answer choices, output only the letter corresponding to the correct answer. Do not add any explanation.\n\nQuestion: What size of cannula would you use in a patient who needed a rapid blood transfusion (as of 2020 medical knowledge)?\nA. 18 gauge.\nB. 20 gauge.\nC. 22 gauge.\nD. 24 gauge.\nAnswer:\n"}], "max_tokens": 3, "temperature": 0.7, "top_p": 0.9}}
    print(infer(test_row))

    
    output_folder = '../../results/fine-tuning_experiments/'
    os.makedirs(output_folder, exist_ok=True)

    # Run evaluation_runs many times
    for i in range(evaluation_runs):
        # Infer on necessary prompts
        generations_map = {}
        with open(evaluation_batch_path, 'r', encoding='utf-8') as fp:
            all_prompts_jsonl = pd.read_json(fp, lines=True)
            # Fine-tuned models are never evaluated on MMLU college medicine since it is used in training
            all_prompts_jsonl = all_prompts_jsonl[~all_prompts_jsonl['custom_id'].str.contains('college_medicine')]
            # Filter more if needed (target benchmark and language for quality x quantity versions)
            if this_eval_benchmark is not None:
                all_prompts_jsonl = all_prompts_jsonl[all_prompts_jsonl['custom_id'].str.contains(f'-on-{this_language}-{this_eval_benchmark}-')]
                print(f"Evaluating on {this_eval_benchmark} in {this_language} ({all_prompts_jsonl.shape[0]} questions)")
            else:
                print(f"Evaluating on everything except MMLU college medicine ({all_prompts_jsonl.shape[0]} questions)")
        
        for index, row in tqdm(all_prompts_jsonl.iterrows(), total=all_prompts_jsonl.shape[0]):
            try:
                generations_map[row['custom_id'].replace('<|MODEL|>', output_name)] = infer({'custom_id': row['custom_id'], 'body': row['body']})
            except Exception as e:
                print(f"Exception \"{e}\" occurred on \"{row['custom_id'].replace('<|MODEL|>', output_name)}\". Skipping...")
        
        # Save generations so they never have to be run again
        with open(os.path.join(output_folder, f'generations_{output_name}_{i}.json'), 'w') as fp:
            json.dump(generations_map, fp, indent=2)

    # Iterate over runs and place results in results folder
    for i in range(evaluation_runs):
        with open(os.path.join(output_folder, f'generations_{output_name}_{i}.json'), 'r') as fp:
            generations_map = json.load(fp)

        # Only get results if we actually tested on the benchmark for this fine-tuned model
        if this_eval_benchmark is None or this_eval_benchmark == 'mmlu-clinical_knowledge':
            # Get clinical knowledge performance (including more sections means taking the average perf. across the sections)
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
            matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_mmlu-clinical_knowledge_{i}.csv'))

        if this_eval_benchmark is None or this_eval_benchmark == 'mmlu-virology':
            # Get virology performance (including more sections means taking the average perf. across the sections)
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
            matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_mmlu-virology_{i}.csv'))

        if this_eval_benchmark is None or this_eval_benchmark == 'winogrande':
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
            matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_winogrande_{i}.csv'))

        if this_eval_benchmark is None or this_eval_benchmark == 'belebele':
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
            
            matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_belebele_{i}.csv'))
        
    # Clear model from memory to make room for next fine-tuning process
    get_ipython().system('rm -rf outputs')
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:





# In[ ]:




