import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # This is needed to create custom legend handles
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils/')))
from useful_variables import evaluation_data, african_languages, winogrande_data
from useful_functions import rouge_score_single

figure_name = 'figure_A12.pdf'

# Filter out Belebele since we do not care about translation performance on it
evaluation_data = evaluation_data[evaluation_data['Evaluation.Data'] != 'belebele']

# Get human-translated rows
evaluation_data_human = evaluation_data[(evaluation_data['Evaluation.Translation Approach'].str.contains('Human')) &
                                        (evaluation_data['Evaluation.Target Language'] != 'en')]

# Get machine-translated rows (excluding backtranslations)
evaluation_data_mt = evaluation_data[(evaluation_data['Evaluation.Question ID'].str.contains('-gt-')) &
                                     (evaluation_data['Evaluation.Target Language'] != 'en')]

# Get English rows (excluding backtranslations)
evaluation_data_en = evaluation_data[(evaluation_data['Evaluation.Translation Approach'].str.contains('Human')) &
                                     (evaluation_data['Evaluation.Target Language'] == 'en')]

# Compute ROUGE-1 scores between machine translations and original English
rouge1_vs_mt = []
rouge1_vs_original = []
for index, row in tqdm(evaluation_data_human.iterrows(), total=evaluation_data_human.shape[0]):
    this_id = row['Evaluation.Question ID']
    this_lang = row['Evaluation.Target Language']

    # Adjust IDs to map to machine-translated or English versions
    english_row = evaluation_data_en[evaluation_data_en['Evaluation.Question ID'] == this_id.replace(f"-{this_lang}-", f"-en-")]
    mt_row = evaluation_data_mt[evaluation_data_mt['Evaluation.Question ID'] == this_id.replace(f"-{this_lang}-", f"-gt-{this_lang}-")]
    assert english_row.shape[0] == mt_row.shape[0] and mt_row.shape[0] == 1  # check that only 1 row matches

    english_row = english_row.iloc[0]
    mt_row = mt_row.iloc[0]

    rouge1_vs_mt.append(rouge_score_single(row['Evaluation.Question'], mt_row['Evaluation.Question']))
    rouge1_vs_original.append(rouge_score_single(row['Evaluation.Question'], english_row['Evaluation.Question']))

# Store ROUGE-1 scores
evaluation_data_human['ROUGE-1 vs Machine-Translation'] = rouge1_vs_mt
evaluation_data_human['ROUGE-1 vs Original English'] = rouge1_vs_original

# Get distributions of ROUGE-1s (after corrections)
distributions_mt_wino = {}
distributions_original_wino = {}
for language in african_languages:
    distribution_wino = evaluation_data_human[(evaluation_data_human['Evaluation.Target Language'] == language) & (evaluation_data_human['Evaluation.Data'] == 'winogrande')]
    assert distribution_wino.shape[0] == 3674  # Assert expected size of Winogrande
    distributions_mt_wino[language] = distribution_wino['ROUGE-1 vs Machine-Translation'].tolist()
    distributions_original_wino[language] = distribution_wino['ROUGE-1 vs Original English'].tolist()

# Get distributions of ROUGE-1s (before corrections)
distributions_mt_round1 = {}
distributions_original_round1 = {}
for language in african_languages:
    relevant_rows = winogrande_data[winogrande_data['Target Language'] == african_languages[language]]
    distribution_mt_round1 = []
    distribution_original_round1 = []
    for index, row in relevant_rows.iterrows():
        distribution_mt_round1.append(rouge_score_single(row['Question - Translation by Initial Translator'], row['Question - Translation by Machine Translation Tool']))
        distribution_original_round1.append(rouge_score_single(row['Question - Translation by Initial Translator'], row['Question - Original English']))

    assert len(distribution_mt_round1) == 3674 and len(distribution_original_round1) == 3674  # Check that size is full Winogrande size
    distributions_mt_round1[language] = distribution_mt_round1
    distributions_original_round1[language] = distribution_original_round1

# Creating the boxplot
fig, ax = plt.subplots(figsize=(11, 4))

# Organize distributions into list of lists in desired order
data_values = []
for language in african_languages:
    data_values.append(distributions_mt_round1[language])
    data_values.append(distributions_mt_wino[language])
    data_values.append(distributions_original_round1[language])
    data_values.append(distributions_original_wino[language])
bp = ax.boxplot(
    data_values,
    tick_labels=list(range(len(african_languages)*4)),  # Four distributions for 11 African languages
    patch_artist=True,
    whis=(0, 100),
    whiskerprops=dict(color='gray', linewidth=1),
    medianprops=dict(color='gray', linewidth=1),
    boxprops=dict(color='gray', linewidth=1)
    # flierprops=dict(marker='o', color='red', markersize=8, markeredgecolor='white')
)

colors = ['#85C1E9', '#82E0AA', '#FB8F7F', '#F5E14F'] * len(african_languages)
# Set the colors for the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add dividing lines between languages
for i in range(len(african_languages)-1):
    ax.axvline(x=i*4+4.5, color='grey', linestyle='--', linewidth=1)

# Set y-range and fontsize
y_min = 0
y_max = 1
fontsize = 12

plt.ylim([y_min - .05, y_max + 0.19])

# Adding labels and title
# Defining labels on xticks
labels = [''] + [lang for lang in african_languages]

# Set tick positions
tick_index = [2.5] + [2.5 + 4 * i for i in range(len(labels) - 1)]

plt.xticks(ticks=tick_index, labels=labels, fontsize=10)

# Create legend
legend_handles = [
    mpatches.Patch(color=c, label=l)
    for c, l in zip(colors[:4], ['Round 1 Sim. to MT', 'Round 2 Sim. to MT', 'Round 1 Sim. to English', 'Round 2 Sim. to English'])
]

# format legend
ax.legend(
    handles=legend_handles,
    frameon=True,
    fancybox=True,
    facecolor='white',
    edgecolor='white',
    loc='upper left',
    # bbox_to_anchor=(0.5, 1.35),
    ncol=4,
    handletextpad=0.5,  # Adjust space between the handle and the text
    columnspacing=1.0,  # Adjust space between columns
    fontsize=10,
)

# axes labels
plt.xlabel('Target Language', fontweight='bold', fontsize=fontsize)
plt.ylabel('ROUGE-1', fontweight='bold', fontsize=fontsize)

# save plot
plt.tight_layout()
plt.savefig(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/figures/{figure_name}')), format='pdf')
