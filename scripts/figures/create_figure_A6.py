import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the global font size for the figure
sns.set_context("talk", font_scale=0.75)  # Adjust font_scale as needed

columns = ['bm', 'am', 'ig', 'sn', 'nso', 'ts', 'tn', 'st', 'xh', 'zu', 'af', 'en']

data = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__),
                                    '../../results/tables/table_A3.csv')))
data = data[columns]
spearman_corr = data.corr(method='spearman')

print(data.columns)

# Create a mask for the upper triangle since the correlations are symmetric
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and custom color map
sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='coolwarm', vmin=0, vmax=1,
            square=True, linewidths=.5, cbar_kws={"label": "Spearman Correlation"})

plt.savefig(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__),
                                          '../../results/figures/figure_A6.pdf')),
            format='pdf', dpi=1200, bbox_inches='tight')
