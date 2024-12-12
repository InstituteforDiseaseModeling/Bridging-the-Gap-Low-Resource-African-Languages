import pandas as pd
import re
import numpy as np
import json
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import languages, fine_tuning_eval_benchmarks

TRIALS = 3

# Load quality responses from GPT-4o
input_path1 = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator/quality_generations_gpt-4o_0.jsonl'))
input_path2 = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator/quality_generations_gpt-4o_1.jsonl'))
model_generations = pd.concat([pd.read_json(input_path1, lines=True, encoding='utf-8'), pd.read_json(input_path2, lines=True, encoding='utf-8')])
model_generations['content'] = model_generations['response'].apply(lambda x: x['body']['choices'][0]['message']['content'])

# Get mapping of custom_id to LLM response
generations_map = dict(zip(model_generations['custom_id'], model_generations['content']))
train_benchmarks = ['mmlu-college_medicine', 'winogrande-train_s']

# Map benchmark IDs to quality score results from GPT-4o
ids_to_results = {}
for language in languages:
    for tb in train_benchmarks:
        for eb in fine_tuning_eval_benchmarks:
            # Construct the pattern to match train and eval benchmarks
            pattern = re.compile(rf"gpt-4o-on-{language}-{tb}-.*_vs_{eb}_.*")

            # Filter keys based on matching pattern to custom_id
            matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]
            if (len(matching_generations)) == 0:
                # print(f"Skipping {language}: {tb} vs {eb}...")
                continue
            else:
                print(f"{language}: {tb} vs {eb} size is {len(matching_generations)}")

            # Compute average and STD scores among the trials
            for (c_id, gen) in matching_generations:
                results = []
                this_id = c_id[len(f"gpt-4o-on-{language}-{tb}-"):c_id.find('_', c_id.find('_')+1)]
                for run in range(TRIALS):
                    results.append(int(generations_map[f"gpt-4o-on-{language}-{tb}-{this_id}_vs_{eb}_{run}"]))

                ids_to_results[f"gpt-4o-on-{language}-{tb}-{this_id}_vs_{eb}"] = {
                    f'Run #{i+1}': results[i] for i in range(len(results))
                }
                ids_to_results[f"gpt-4o-on-{language}-{tb}-{this_id}_vs_{eb}"]['Mean'] = np.mean(results)
                ids_to_results[f"gpt-4o-on-{language}-{tb}-{this_id}_vs_{eb}"]['STD'] = np.std(results, ddof=1)

# Save output to JSON
with open(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator/benchmarks_to_quality_scores.json')), 'w', encoding='utf-8') as fp:
    json.dump(ids_to_results, fp, indent=2, ensure_ascii=False)
