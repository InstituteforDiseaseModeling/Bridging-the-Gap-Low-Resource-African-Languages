import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../..')))
from config import Config
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
random.seed(42)
output_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/o1-related/cost_estimations.txt'))

# TODO: Set parameters of test
trials = 3
questions_to_sample = 3
# Prices from https://openai.com/api/pricing/
o1_input_per_million = 15
o1_output_per_million = 60
o1_mini_input_per_million = 3
o1_mini_output_per_million = 12


# Make the prices directly usable
o1_input_per_million /= 1e6
o1_output_per_million /= 1e6
o1_mini_input_per_million /= 1e6
o1_mini_output_per_million /= 1e6

# For reproducibility
random_seeds = random.sample(range(1, 101), trials)

# Set up OpenAI credentials
openai_api_key = Config.gpt_api_key

client = OpenAI(api_key=openai_api_key)

# Load data, create columns that will be used to stratify data
batch_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm_evaluation/evaluation_batch_full.jsonl'))
batch_df = pd.read_json(batch_path, encoding='utf-8', lines=True)
batch_df['mmlu?'] = batch_df['custom_id'].apply(lambda x: '-mmlu-' in x)
batch_df['winogrande?'] = batch_df['custom_id'].apply(lambda x: '-winogrande-' in x)
batch_df['belebele?'] = batch_df['custom_id'].apply(lambda x: '-belebele-' in x)
batch_df['lang'] = batch_df['custom_id'].apply(lambda x: x[x.find('-on-')+4:x.find('-on-')+6])
batch_df['dataset'] = batch_df.apply(lambda row: [col[:-1] for col in ['mmlu?', 'winogrande?', 'belebele?'] if row[col]][0], axis=1)

# Randomly sample test data; run test
input_token_usages = []
output_token_usages = []
for i in range(trials):
    sampled_df = batch_df.groupby(['dataset', 'lang']).apply(lambda x: x.sample(questions_to_sample, random_state=random_seeds[i])).reset_index(drop=True)
    total_input = 0
    total_output = 0
    errors = 0
    for index, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0]):
        input_content = row['body']['messages']
        try:
            response = client.chat.completions.create(  # o1 models do not currently accept parameters like temperature
                model="o1-mini",  # test using o1-mini, assuming similar to o1-preview in token usage (much cheaper to use o1-mini for testing)
                messages=input_content,
            )
        except Exception as e:
            print(e)
            print(input_content)
            errors += 1
            continue
        print(response.choices[0].message.content)
        input_tokens, output_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
        total_input += input_tokens
        total_output += output_tokens
    # Place averages from the trial
    input_token_usages.append(total_input / (sampled_df.shape[0] - errors))
    output_token_usages.append(total_output / (sampled_df.shape[0] - errors))
    print(f"Errors during Trial {i+1}: {errors}")

# Get estimated total costs for entire dataset using average token usage from sampled questions
o1_costs = [(x*o1_input_per_million+y*o1_output_per_million)*batch_df.shape[0] for x,y in zip(input_token_usages, output_token_usages)]
o1_mini_costs = [(x*o1_mini_input_per_million+y*o1_mini_output_per_million)*batch_df.shape[0] for x,y in zip(input_token_usages, output_token_usages)]

# Write to output file (reader-friendly)
with open(output_path, 'w', encoding='utf-8') as fp:

    # Write each trial's values
    for i in range(trials):
        fp.write(f'Trial {i + 1}:\n')
        fp.write(f'  Average input token usage per question: {input_token_usages[i]:.2f}\n')
        fp.write(f'  Average output token usage per question: {output_token_usages[i]:.2f}\n')
        fp.write(f'  Est. o1-preview cost for full evaluation (USD): {o1_costs[i]:.4f}\n')
        fp.write(f'  Est. o1-mini cost for full evaluation (USD): {o1_mini_costs[i]:.4f}\n\n')

    # Lists and their names for statistics
    lists = [input_token_usages, output_token_usages, o1_costs, o1_mini_costs]
    list_names = ['input_token_usages', 'output_token_usages', 'o1_costs', 'o1_mini_costs']

    # Compute and write statistics for each list
    for data, name in zip(lists, list_names):
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        ci_lower = mean - 2 * std
        ci_upper = mean + 2 * std

        fp.write(f'Statistics for {name}:\n')
        fp.write(f'  Mean: {mean:.2f}\n')
        fp.write(f'  Standard Deviation: {std:.2f}\n')
        fp.write(f'  95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]\n\n')
