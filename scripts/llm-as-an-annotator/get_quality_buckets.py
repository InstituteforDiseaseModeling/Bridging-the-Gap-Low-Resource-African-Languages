import json
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import languages, fine_tuning_eval_benchmarks

# Get mapping of IDs to scores from before
with open(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator/benchmarks_to_quality_scores.json')), 'r', encoding='utf-8') as fp:
    ids_to_results = json.load(fp)

# set up various paths and variables
train_benchmarks = ['mmlu-college_medicine', 'winogrande-train_s']
output_folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator/'))
output_path = os.path.join(output_folder, 'quality_buckets.json')
output_dict = {language: {tb: {} for tb in train_benchmarks} for language in languages}

# Go through IDs/scores and map to quality buckets (low, med, high thirds)
for language in languages:
    for tb in train_benchmarks:
        for eb in fine_tuning_eval_benchmarks:
            # Construct the pattern to match train and eval benchmarks
            pattern = re.compile(rf"gpt-4o-on-{language}-{tb}-.*_vs_{eb}")

            # Filter keys based on matching pattern to custom_id
            matches = [(c_id, results) for c_id, results in ids_to_results.items() if pattern.match(c_id)]
            if (len(matches)) == 0:
                # print(f"Skipping {language}: {tb} vs {eb}...")
                continue
            else:
                print(f"{language}: {tb} vs {eb} size is {len(matches)}")

            # Sort by mean then STD from least to greatest
            matches.sort(key=lambda x: (x[1]['Mean'], x[1]['STD']))

            # Split into three equal parts
            splits = np.array_split([match[0][len(f"gpt-4o-on-{language}-{tb}-"):match[0].find('_', match[0].find('_')+1)] for match in matches], 3)
            all_means = [match[1]['Mean'] for match in matches]
            score_splits = np.array_split(all_means, 3)

            # Map splits into bucket names
            output_dict[language][tb][eb] = {
                'low': [x for x in splits[0].tolist()],  # higher score = higher quality
                'med': [x for x in splits[1].tolist()],
                'high': [x for x in splits[2].tolist()],
            }

            print(f"\n{language} {tb} vs {eb} score splits (L, M, H): "
                  f"[{score_splits[0].tolist()[0]}, {score_splits[0].tolist()[-1]}]; "
                  f"[{score_splits[1].tolist()[0]}, {score_splits[1].tolist()[-1]}]; "
                  f"[{score_splits[2].tolist()[0]}, {score_splits[2].tolist()[-1]}]")

            # Calculate tertiles
            lower_tertile = np.percentile(all_means, 100/3)
            upper_tertile = np.percentile(all_means, 200/3)

            # Create histogram data
            hist, bins = np.histogram(all_means, bins=[i/3 for i in range(3, 31)])

            # Define colors based on tertiles
            colors = []
            for bin_edge in bins[:-1]:
                if bin_edge < lower_tertile:
                    colors.append('red')  # Lower tertile
                elif bin_edge < upper_tertile:
                    colors.append('green')  # Middle tertile
                else:
                    colors.append('blue')  # Upper tertile

            # Plot histogram with colored bins
            plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black', color=colors, align='edge')
            plt.title(f"GPT-4o utility scores of {tb} on {eb} in {language}")
            plt.ylabel("Count")
            plt.xlabel("Score")
            plt.xlim(1, 10)
            # Save figure
            plots_dir = os.path.join(output_folder, 'plots/')
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, f'{language}_{tb}_vs_{eb}.png'), bbox_inches='tight', dpi=300)
            plt.close()

# Save buckets
with open(output_path, 'w', encoding='utf-8') as fp:
    json.dump(output_dict, fp, indent=2, ensure_ascii=False)
