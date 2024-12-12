import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '..')))
from config import Config
from openai import OpenAI
import pandas as pd
import json
import concurrent.futures
from tqdm import tqdm
import threading
file_lock = threading.Lock()

# Set parameters of evaluation
MAX_WORKERS = 16  # How many API calls at once?
MODEL = 'o1-mini-2024-09-12'
batch_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm_evaluation/evaluation_batch_full.jsonl'))
output_folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/o1-related'))
output_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/o1-related/generations_{MODEL}.jsonl'))

# Get already completed custom_ids in case of failure so don't have to restart
if os.path.exists(output_path):
    existing_outputs_df = pd.read_json(output_path, lines=True, encoding='utf-8')
    existing_outputs = set(existing_outputs_df['custom_id'].tolist())
    print(f"{len(existing_outputs)} existing completions loaded that will not be redone!")
else:
    existing_outputs = set()

# Set up OpenAI credentials
openai_api_key = Config.gpt_api_key

client = OpenAI(api_key=openai_api_key)

# Load data
os.makedirs(output_folder, exist_ok=True)
eval_df = pd.read_json(batch_path, lines=True, encoding='utf-8')
eval_df['custom_id'] = eval_df['custom_id'].str.replace('<|MODEL|>', MODEL)
eval_df['messages'] = eval_df['body'].apply(lambda x: x['messages'])


# Evaluates a single question using o1
def evaluate_o1(arg_tup):
    custom_id, messages = arg_tup
    if custom_id in existing_outputs:
        return

    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages=messages
        ).choices[0].message.content

        result = {'custom_id': custom_id, 'content': result}

        # Write the result to the output file immediately
        with file_lock:
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                f.flush()

    except Exception as e:
        print(f"An error occurred: {e}")


args_list = []
for index, row in eval_df.iterrows():
    args_list.append((row['custom_id'], row['messages']))

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks to the executor
    futures = [executor.submit(evaluate_o1, arg) for arg in args_list]

    # Use as_completed to get futures as they complete
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass
