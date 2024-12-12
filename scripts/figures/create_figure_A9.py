from sklearn.metrics import cohen_kappa_score
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import os

# import the data in data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv
import pandas as pd

# read the data
data = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv')))

# create a column that is true if both annotators agree that the translation quality is "Good translation"
data['good_translation'] = (data['Translation Quality per Annotator 1'] == 'Good translation') & (data['Translation Quality per Annotator 2'] == 'Good translation')

# create a column that is true if either annotator thinks the sentence is inappropriate
data['inappropriate'] = ((data['Cultural Appropriateness per Annotator 1'] != 'No, the sentence is typical') | (data['Cultural Appropriateness per Annotator 2'] != 'No, the sentence is typical'))
data.head()

data['appropriate'] = data['inappropriate'] == False

#Exclude the rows where the translation quality is not "Good translation" for both annotators
data_gt = data[data['good_translation']]

#Generate a list of Target Languages
languages = data_gt['Target Language'].unique()

#initizlize a dataframe where the rows are the target languages and the columns are the target languages, and each cell is the cohen's kappa
result = pd.DataFrame(index=languages, columns=languages)

#For each target langauge
for language_a in languages:
    #for each target language, calculate the Cohen's Kappa for the Cultural Appropriateness
    for language_b in languages:

        #Filter the data for the target language
        data_a = data_gt[data_gt['Target Language'] == language_a]
        data_b = data_gt[data_gt['Target Language'] == language_b]

        #join the two dataframes on the Winogrande ID
        data_ab = data_a.merge(data_b, on='Winogrande Question ID')

        #compute the cohen's kappa between the inappropriateness columns of the two languages
        kappa = cohen_kappa_score(data_ab['appropriate_x'], data_ab['appropriate_y'])

        #round kappa to 2 decimal places
        kappa = round(kappa, 2)
        if kappa == -0.0:
            kappa = 0.0

        #store the kappa in the result dataframe
        result.loc[language_a, language_b] = kappa


# generate a figure from this result that shows the cohen's kappa between each pair of languages, shade the cells based on the value of the kappa


#convert the result to a numeric dataframe
result = result.apply(pd.to_numeric)

# Define the custom colormap using a dictionary for exact segment control
cdict = {
    'red':   [(0.0, 0.0, 0.0),  # Blue
              (0.2, 0.0, 0.0),
              (0.4, 0.0, 0.0),
              (0.6, 1.0, 1.0),
              (0.8, 1.0, 1.0),
              (1.0, 1.0, 1.0)],  # Red

    'green': [(0.0, 0.0, 0.0),  # Blue
              (0.2, 0.0, 0.0),
              (0.4, 1.0, 1.0),
              (0.6, 1.0, 1.0),
              (0.8, 0.0, 0.0),
              (1.0, 0.0, 0.0)],  # Red

    'blue':  [(0.0, 1.0, 1.0),  # Blue
              (0.2, 1.0, 1.0),
              (0.4, 1.0, 1.0),
              (0.6, 0.0, 0.0),
              (0.8, 0.0, 0.0),
              (1.0, 0.0, 0.0)]   # Red
}

# Create the colormap
cmap = LinearSegmentedColormap('custom_cmap', cdict)

# Adjust font sizes for x and y axis labels
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Plot the heatmap
ax = sns.heatmap(result, annot=True, cmap=cmap, cbar_kws={'label': 'Cohen\'s Kappa'}, annot_kws={'fontsize': 8})
ax.figure.axes[-1].yaxis.label.set_size(8)
cbar = ax.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=8)
plt.savefig(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/figures/figure_A9.pdf')), format='pdf', dpi=1200, bbox_inches='tight')
