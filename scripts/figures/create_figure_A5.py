import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches  # This is needed to create custom legend handles
import os

figure_name = 'figure_A5.pdf'

# load data
df = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/2. Winogrande Data.csv')))

# create column indicating if the translation required correction
df['Corrected'] = df['Question - Correction by Verifying Translator'].apply(lambda x: not pd.isna(x))

# columns
df = df[["Target Language","Winogrande Question ID","Corrected"]]

# pivot so that each language has a column
df = df.pivot(index='Winogrande Question ID', columns='Target Language', values='Corrected')

# get the percentage of corrections required for each language
corrected = df.sum() / df.count() * 100
not_corrected = 100 - corrected

# sort by corrected
corrected = corrected.sort_values(ascending=False)
not_corrected = not_corrected[corrected.index]

blue = "#4A86E8"
yellow = "#FFE599"

# Plot a stacked bar chart showing the percentage of questions that required correction and those that did not
plt.figure(figsize=(12, 5))
plt.bar(corrected.index, corrected, bottom=not_corrected, color=yellow, label='Corrected', width=0.7)
plt.bar(not_corrected.index, not_corrected, color=blue, label='Not Corrected', width=0.7)

# add legend
yellow_patch = mpatches.Patch(color=yellow, label='Corrected')
blue_patch = mpatches.Patch(color=blue, label='Not Corrected')
plt.legend(handles=[yellow_patch, blue_patch])

# move legend to the top of the plot
plt.legend(handles=[yellow_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=12)

# show the values on top of the bars
for i, v in enumerate(corrected):
    plt.text(i, v + not_corrected[i]-4.8, f'{v:.1f}%', ha='center', va='bottom')

for i, v in enumerate(not_corrected):
    plt.text(i, v-3.3, f'{v:.1f}%', ha='center', va='center', color='white')

# add labels
plt.xlabel('Language', fontsize=14)
plt.ylabel('Percentage of Winogrande Questions (%)', fontsize=14)
#plt.title('Percentage of Winogrande Questions Requiring Correction by Language')

# make y axis percentages
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])

# increase font size of ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# save plot
plt.tight_layout()
plt.savefig(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/figures/{figure_name}')), format='pdf')
