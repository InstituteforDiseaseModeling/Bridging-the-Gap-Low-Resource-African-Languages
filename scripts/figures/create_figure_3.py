import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # This is needed to create custom legend handles
import matplotlib.lines as mlines
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils/')))
from useful_variables import llm_responses, african_languages, runs, quantities, qualities, baselines_ft

figure_name = 'figure_3.pdf'

# Filter LLM responses to get quality experiments for MMLU
llm_responses = llm_responses[(~pd.isnull(llm_responses['Fine-Tuning.Data Partition.Data Quality'])) &
                              (llm_responses['Fine-Tuning.Data'] == 'mmlu-college_medicine') &
                              (llm_responses['Evaluation.Data'] == 'mmlu-clinical_knowledge')]

# Get performance distributions across African languages for the quality x quantity experiments
afr_dists = {}
for quantity in quantities:
    for quality in qualities:
        dist = []
        for language in african_languages:
            run_accuracies = []
            for run in runs:
                relevant_responses = llm_responses[(llm_responses['Fine-Tuning.Data Partition.Data Quality.Percent Used'] == quantity) &
                                                   (llm_responses['Fine-Tuning.Data Partition.Data Quality'] == quality) &
                                                   (llm_responses['Evaluation.Target Language'] == language) &
                                                   (llm_responses['Evaluation.Trial Number'] == run)]
                assert relevant_responses.shape[0] == 265  # number of MMLU questions in MMLU clinical knowledge
                run_accuracy = sum(relevant_responses['Evaluation.Model Response Was Correct'].tolist()) / relevant_responses.shape[0]
                run_accuracies.append(run_accuracy)
            mean_accuracy = np.mean(run_accuracies)*100
            dist.append(mean_accuracy)
        afr_dists[(quantity, quality)] = dist

eng_dists = {}
for quantity in quantities:
    for quality in qualities:
        dist = []
        run_accuracies = []
        for run in runs:
            relevant_responses = llm_responses[(llm_responses['Fine-Tuning.Data Partition.Data Quality.Percent Used'] == quantity) &
                                               (llm_responses['Fine-Tuning.Data Partition.Data Quality'] == quality) &
                                               (llm_responses['Evaluation.Target Language'] == 'en') &
                                               (llm_responses['Evaluation.Trial Number'] == run)]
            assert relevant_responses.shape[0] == 265  # number of MMLU questions in MMLU clinical knowledge
            run_accuracy = sum(relevant_responses['Evaluation.Model Response Was Correct'].tolist()) / relevant_responses.shape[0]
            run_accuracies.append(run_accuracy)
        mean_accuracy = np.mean(run_accuracies)*100
        dist.append(mean_accuracy)
        eng_dists[(quantity, quality)] = dist

# Get distributions in desired order
ordered_dists = []
# Begin with baselines
for i in range(2):
    ordered_dists.append([baseline*100 for (benchmark, language), baseline in baselines_ft.items() if benchmark == 'mmlu-clinical_knowledge' and language != 'en'])

# Then finish with quality x quantity results
for quantity in quantities:
    for quality in qualities:
        ordered_dists.append(afr_dists[(quantity, quality)])

# Creating the boxplot
fig, ax = plt.subplots(figsize=(5, 4))
bp = ax.boxplot(
    ordered_dists,
    tick_labels=list(range(len(ordered_dists))),
    patch_artist=True,
    whis=(0, 100),
    whiskerprops=dict(color='gray', linewidth=1),
    medianprops=dict(color='gray', linewidth=1),
    boxprops=dict(color='gray', linewidth=1)
    # flierprops=dict(marker='o', color='red', markersize=8, markeredgecolor='white')
)

colors = ['#FCE5CD', '#B5D7A8'] * (len(ordered_dists) // 2)
# Set the colors for the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

english_baseline = baselines_ft[('mmlu-clinical_knowledge', 'en')]*100

# Plotting English baselines
ax.plot(range(1, 10, 2),
        [english_baseline] + [eng_dists[(quantity, 'low')][0] for quantity in quantities],
        's',
        markerfacecolor='#F5CBCC',
        markeredgecolor='#F5CBCC',
        label='En (low quality)')

ax.plot(range(2, 12, 2),
        [english_baseline] + [eng_dists[(quantity, 'high')][0] for quantity in quantities],
        's',
        markerfacecolor='#990001',
        markeredgecolor='#990001',
        label='En (high quality)')

fontsize = 12

# Set desired ranges
plt.ylim([27, 102])

# Adding labels and title
# Defining labels on xticks
labels = ['', '0%\n(N/A)', '25%\n(17)', '50%\n(33)', '75%\n(50)', '100%\n(66)']

# Create the sequence starting at 1.5, incrementing by 2, and for 1.5 times the length of the list
tick_index = [1.5] + [1.5 + 2 * i for i in range(len(labels) - 1)]
# print(len(tick_index))

plt.xticks(ticks=tick_index, labels=labels, fontsize=10)

# Create legend
legend_handles = [
    mpatches.Patch(color=c, label=l)
    for c, l in zip(colors[:2], ['Low Quality', 'High Quality'])
]

# Add a circle legend handle (mono-lingual circle)
# You can specify the marker as 'o' for circle, color, and other properties
circle_handle = mlines.Line2D([], [],
                              markerfacecolor='#F5CBCC',
                              markeredgecolor='#F5CBCC',
                              marker='s',
                              markersize=8,
                              label='En (low quality)',
                              linestyle='None')

legend_handles.append(circle_handle)

# Add second (cross-lingual square) to legend
circle_handle = mlines.Line2D([], [],
                              markerfacecolor='#990001',
                              markeredgecolor='#990001',
                              marker='s',
                              markersize=8,
                              label='En (high quality)',
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
plt.xlabel('Percent of Data Quality Tertile Used\n(samples)', fontweight='bold', fontsize=fontsize)
plt.ylabel('Performance (%)', fontweight='bold', fontsize=fontsize)

# save plot
plt.tight_layout()
plt.savefig(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/figures/{figure_name}')), format='pdf')
