import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # This is needed to create custom legend handles
import matplotlib.lines as mlines
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils/')))
from useful_variables import llm_responses, baselines_ft, languages, african_languages, runs, fine_tuning_eval_benchmarks, fine_tuning_train_benchmarks
from useful_functions import flatten_list

figure_name = 'figure_2.pdf'

# Filter to get only cross-lingual experiments (i.e. fine-tuning, but no quality or quantity)
llm_responses = llm_responses[(llm_responses['Model.Was Fine-Tuned']) &
                              (pd.isnull(llm_responses['Fine-Tuning.Data Partition.Data Quality']))]

assert llm_responses.shape[0] == 2676672  # 3 runs * (1767 Wino Q's + 265 MMLU CK Q's + 166 MMLU Vir Q's + 900 Belebele Q's) * (12 models tuned on Wino + 12 models tuned on MMLU CM) * 12 languages to evaluate on

# Get comprehensive performance gains above baselines across African languages
afr_lang_gains = {}
for train_benchmark in fine_tuning_train_benchmarks:
    train_responses = llm_responses[llm_responses['Fine-Tuning.Data'] == train_benchmark]  # Filter incrementally to speed things up
    for eval_benchmark in fine_tuning_eval_benchmarks:
        eval_responses = train_responses[train_responses['Evaluation.Data'] == eval_benchmark]
        for train_lang in african_languages:
            train_lang_responses = eval_responses[eval_responses['Fine-Tuning.Language'] == train_lang]
            for eval_lang in african_languages:
                run_accuracies = []
                relevant_responses = train_lang_responses[train_lang_responses['Evaluation.Target Language'] == eval_lang]
                for run in runs:
                    more_relevant_responses = relevant_responses[relevant_responses['Evaluation.Trial Number'] == run]

                    run_accuracy = sum(more_relevant_responses['Evaluation.Model Response Was Correct'].tolist()) / more_relevant_responses.shape[0]
                    run_accuracies.append(run_accuracy)
                mean_accuracy = np.mean(run_accuracies)
                afr_lang_gains[(train_benchmark, eval_benchmark, train_lang, eval_lang)] = mean_accuracy - baselines_ft[(eval_benchmark, eval_lang)]

# Get distributions of cross-lingual performances for African languages
afr_lang_dists = defaultdict(list)
for train_benchmark in fine_tuning_train_benchmarks:
    for eval_benchmark in fine_tuning_eval_benchmarks:
        for train_lang in african_languages:
            for eval_lang in african_languages:
                if train_lang == eval_lang:
                    afr_lang_dists[(train_benchmark, eval_benchmark, 'mono-lingual')].append(afr_lang_gains[(train_benchmark, eval_benchmark, train_lang, eval_lang)]*100)
                else:
                    afr_lang_dists[(train_benchmark, eval_benchmark, 'cross-lingual')].append(afr_lang_gains[(train_benchmark, eval_benchmark, train_lang, eval_lang)]*100)

# Get distributions in desired order for displaying
ordered_dists = []
ordered_train_benchmarks = ['winogrande', 'mmlu-college_medicine']
ordered_eval_benchmarks = ['winogrande', 'mmlu-clinical_knowledge', 'mmlu-virology', 'belebele']
lingualities = ['mono-lingual', 'cross-lingual']
for train_benchmark in ordered_train_benchmarks:
    for eval_benchmark in ordered_eval_benchmarks:
        for linguality in lingualities:
            ordered_dists.append(afr_lang_dists[(train_benchmark, eval_benchmark, linguality)])

# Get performance gains above baselines for English
eng_lang_gains = {}
for train_benchmark in fine_tuning_train_benchmarks:
    train_responses = llm_responses[llm_responses['Fine-Tuning.Data'] == train_benchmark]  # Filter incrementally to speed things up
    for eval_benchmark in fine_tuning_eval_benchmarks:
        eval_responses = train_responses[(train_responses['Evaluation.Data'] == eval_benchmark) &
                                         (train_responses['Evaluation.Target Language'] == 'en')]
        for train_lang in languages:
            run_accuracies = []
            relevant_responses = eval_responses[eval_responses['Fine-Tuning.Language'] == train_lang]
            for run in runs:
                more_relevant_responses = relevant_responses[relevant_responses['Evaluation.Trial Number'] == run]

                run_accuracy = sum(more_relevant_responses['Evaluation.Model Response Was Correct'].tolist()) / more_relevant_responses.shape[0]
                run_accuracies.append(run_accuracy)
            mean_accuracy = np.mean(run_accuracies)
            eng_lang_gains[(train_benchmark, eval_benchmark, train_lang)] = mean_accuracy - baselines_ft[(eval_benchmark, 'en')]

# Get distributions of cross-lingual performances for English
eng_lang_dists = defaultdict(list)
for train_benchmark in fine_tuning_train_benchmarks:
    for eval_benchmark in fine_tuning_eval_benchmarks:
        for train_lang in languages:
            if 'en' == train_lang:
                eng_lang_dists[(train_benchmark, eval_benchmark, 'mono-lingual')] = eng_lang_gains[(train_benchmark, eval_benchmark, train_lang)]*100
            else:
                eng_lang_dists[(train_benchmark, eval_benchmark, 'cross-lingual')].append(eng_lang_gains[(train_benchmark, eval_benchmark, train_lang)]*100)

        # Take median of distribution to display
        eng_lang_dists[(train_benchmark, eval_benchmark, 'cross-lingual')] = np.median(eng_lang_dists[(train_benchmark, eval_benchmark, 'cross-lingual')])

# Order English points of data
ordered_eng_mono = []
ordered_eng_cross = []
for train_benchmark in ordered_train_benchmarks:
    for eval_benchmark in ordered_eval_benchmarks:
        ordered_eng_mono.append(eng_lang_dists[(train_benchmark, eval_benchmark, 'mono-lingual')])
        ordered_eng_cross.append(eng_lang_dists[(train_benchmark, eval_benchmark, 'cross-lingual')])


# Creating the boxplot
fig, ax = plt.subplots(figsize=(5, 4))
bp = ax.boxplot(
    ordered_dists,
    tick_labels=list(range(len(afr_lang_dists))),
    patch_artist=True,
    whis=(0, 100),
    whiskerprops=dict(color='gray', linewidth=1),
    medianprops=dict(color='gray', linewidth=1),
    boxprops=dict(color='gray', linewidth=1)
    # flierprops=dict(marker='o', color='red', markersize=8, markeredgecolor='white')
)

# Customizing colors (blue, green)
colors = ['#85C1E9', '#82E0AA']
for i in range(len(afr_lang_dists) // 2):
  colors.extend(colors[:2])

# Set the colors for the boxes
for patch, color in zip(bp['boxes'], colors):
  patch.set_facecolor(color)

# Plotting English baselines
# plotting English baseline - mono-lingual
ax.plot(range(1, 16, 2),
      ordered_eng_mono,
      'o',
      markerfacecolor='white',
      markeredgecolor='#F48FB1',
      label='En (mono)')

# plotting English baseline - cross-lingual
ax.plot(range(2, 18, 2),
      ordered_eng_cross,
      's',
      markerfacecolor='#F48FB1',
      markeredgecolor='#F48FB1',
      label='En (cross - median)')

# Adding vertical line between the 4th and 5th boxplot
# Position can be 4.5 if boxes are centered at 1, 2, 3, ..., N
ax.axvline(x=8.5, color='grey', linestyle='--', linewidth=1)

# Flatten the list of all values
all_values = flatten_list(ordered_dists)

# Find the minimum value across all lists
# This will allow us to position the text on the figure
y_min = min(all_values)
y_max = max(all_values)

fontsize = 12

# Define two plot splits with fine-tuning data type
# print(y_min)
ax.text(0.65,
        y_min - 10,
        'Fine-tuning Data:\nWinogrande',
        ha='left',
        va='bottom',
        fontsize=10,
        color='black')

ax.text(8.65,
        y_min - 10,
        'Fine-tuning Data: \nMMLU (college med.)',
        ha='left',
        va='bottom',
        fontsize=10,
        color='black')

plt.ylim([y_min - 10, y_max + 20])

# Adding labels and title
# Defining labels on xticks
labels = [''] + ['Wino', 'MMLU\n(ck)', 'MMLU\n(vir)', 'Bele']*2

# Create the sequence starting at 1.5, incrementing by 2, and for 1.5 times the length of the list
tick_index = [1.5] + [1.5 + 2 * i for i in range(len(labels) - 1)]
# print(len(tick_index))

plt.xticks(ticks=tick_index, labels=labels, fontsize=10)

# Create legend
legend_handles = [
    mpatches.Patch(color=c, label=l)
    for c, l in zip(colors[:2], ['Mono-ling.', 'Cross-ling.'])
]

# Add a circle legend handle (mono-lingual circle)
circle_handle = mlines.Line2D([], [],
                              markeredgecolor='#F48FB1',
                              markerfacecolor='white',
                              marker='o',
                              markersize=8,
                              label='En (mono-ling)',
                              linestyle='None')

legend_handles.append(circle_handle)

# Add second (cross-lingual square) to legend
circle_handle = mlines.Line2D([], [],
                              markeredgecolor='#F48FB1',
                              markerfacecolor='#F48FB1',
                              marker='s',
                              markersize=8,
                              label='En (cross-ling, median)',
                              linestyle='None')

legend_handles.append(circle_handle)

# format legend
ax.legend(
    handles=legend_handles,
    frameon=True,
    fancybox=True,
    facecolor='white',
    edgecolor='white',
    loc='upper left',
    # bbox_to_anchor=(0.5, 1.35),
    ncol=2,
    handletextpad=0.5,  # Adjust space between the handle and the text
    columnspacing=1.0,  # Adjust space between columns
    fontsize=10,
)

# axes labels
plt.xlabel('Evaluation Data', fontweight='bold', fontsize=fontsize)
plt.ylabel('Performance Gain (%)', fontweight='bold', fontsize=fontsize)

# save plot
plt.tight_layout()
plt.savefig(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/figures/{figure_name}')), format='pdf')
