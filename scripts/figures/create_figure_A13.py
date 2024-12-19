# ## Figure A.13
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import random
random.seed(42)

data = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv')))
eval = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/6. Evaluation Data.csv')), low_memory=False)
resp = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/5. LLM Responses.csv')), low_memory=False)

# left join data to eval on the column 'Winogrande Question ID' and "Evaluation.Winogrande Question ID" and Target Language columns
data2 = pd.merge(data, eval, left_on=['Winogrande Question ID', 'Target Language'],
                 right_on=['Evaluation.Winogrande Question ID', 'Evaluation.Target Language'], how='left')

# left join data2 to resp on the Evaluation.Question ID columns
data2 = pd.merge(data2, resp, left_on='Evaluation.Question ID', right_on='Evaluation.Question ID', how='left')

# keep only the rows in data2 where Evaluation.Translation Approach_y contains the string "Human"
data2 = data2[data2['Evaluation.Translation Approach_y'] == 'Human - Upwork.com']

# Keep only the out-of-the-box results
data2 = data2[data2['Model.Was Fine-Tuned'] == False]

# Keep only the GPT 4o out-of-the-box results
data2 = data2[data2['Model.Unique Identifier'] == 'gpt-4o-2024-05-13']

data2['good_translation'] = (data2['Translation Quality per Annotator 1'] != 'Completely wrong') & (
            data2['Translation Quality per Annotator 2'] != 'Completely wrong')

data2['appropriate'] = ((data2['Cultural Appropriateness per Annotator 1'] == 'No, the sentence is typical') & (
            data2['Cultural Appropriateness per Annotator 2'] == 'No, the sentence is typical'))

# keep only these columns: Target Language, Winogrande Question ID, Translation Quality per Annotator 1, Translation Quality per Annotator 2, Cultural Appropriateness per Annotator 1, Cultural Appropriateness per Annotator 2, Evaluation.Model Response Was Correct
data3 = data2[['Target Language', 'Winogrande Question ID', 'good_translation', 'appropriate',
               'Evaluation.Model Response Was Correct']]

# keep only the good transaltions
data3 = data3[data3['good_translation']]

# Generate a list of Target Languages
languages = data3['Target Language'].unique()

# initialize a dict to store results for each langauge
results = {}

# For each langage in the list
for lang in languages:
    # generate a dataframe for the language
    data_lang = data3[data3['Target Language'] == lang]

    # compute the count of the rows that are "appropriate" and "inappropriate"
    count = data_lang.groupby('appropriate').count()

    # store the appropriate count in a variable
    appropriate = count.loc[True, 'Target Language']

    # store the inappropriate count in a variable
    inappropriate = count.loc[False, 'Target Language']

    # Count row many rows were correct when the translation was appropriate
    correct_appropriate = data_lang[data_lang['appropriate'] == True]['Evaluation.Model Response Was Correct'].sum()

    # Count row many rows were correct when the translation was inappropriate
    correct_inappropriate = data_lang[data_lang['appropriate'] == False]['Evaluation.Model Response Was Correct'].sum()

    # compute the percentage of correct responses when the translation was appropriate and when the translation was inappropriate
    correct_appropriate_percentage = correct_appropriate / appropriate
    correct_inappropriate_percentage = correct_inappropriate / inappropriate

    # initialize a list to store the results
    appropriate_rand_results = []
    inappropriate_rand_results = []
    for i in range(100):
        # draw a random sample of data the same size as the appropriate count
        random_sample = data_lang.sample(appropriate, random_state=random.randint(0, 2**32-1))

        # draw a random sample of data the same size as the inappropriate count
        random_sample2 = data_lang.sample(inappropriate, random_state=random.randint(0, 2**32-1))

        # Count row many rows were correct when the translation was appropriate
        correct_appropriate_rand = random_sample['Evaluation.Model Response Was Correct'].sum()

        # Count row many rows were correct when the translation was inappropriate
        correct_inappropriate_rand = random_sample2['Evaluation.Model Response Was Correct'].sum()

        # compute the percentage of correct responses using the random sample
        correct_appropriate_rand_percentage = correct_appropriate_rand / appropriate
        correct_inappropriate_rand_percentage = correct_inappropriate_rand / inappropriate

        # append the results to the list
        appropriate_rand_results.append(correct_appropriate_rand_percentage)
        inappropriate_rand_results.append(correct_inappropriate_rand_percentage)

    # store the results in the results dict
    results[lang] = (correct_appropriate_percentage, correct_inappropriate_percentage, appropriate_rand_results,
                     inappropriate_rand_results)

# draw a box and whisker plot of the results for all languages, make sure inappropriate is on the left and appropriate is on the right and ensure that you plot the languages in the following order: xh,ig,ts,bm,am,tn,st,zu,af,nso,sn
languages = ['xh', 'ig', 'ts', 'bm', 'am', 'tn', 'st', 'zu', 'af', 'nso', 'sn']

plt.figure(figsize=(10, 5))

plt.boxplot([results[lang][3] for lang in languages], positions=range(0, len(languages) * 2, 2), widths=0.6,
            patch_artist=True, boxprops=dict(facecolor='none'), showfliers=False)
plt.boxplot([results[lang][2] for lang in languages], positions=range(1, len(languages) * 2, 2), widths=0.6,
            patch_artist=True, boxprops=dict(facecolor='none'), showfliers=False)
plt.xticks(range(0, len(languages) * 2, 2), languages)
plt.xlabel('Target Language')
plt.ylabel('Percentage of Correct Responses')

# now draw the correct appropriate percentage and the correct inappropriate percentage on the plot as two points
# ensure that you plot the languages in the following order: xh,ig,ts,bm,am,tn,st,zu,af,nso,sn
languages = ['xh', 'ig', 'ts', 'bm', 'am', 'tn', 'st', 'zu', 'af', 'nso', 'sn']
for i, lang in enumerate(languages):
    plt.scatter(i * 2, results[lang][1], color='blue')
    plt.scatter(i * 2 + 1, results[lang][0], color='red')

# Create the custom legend for points
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Inappropriate',
                          markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Appropriate',
                          markerfacecolor='red', markersize=10)]

plt.legend(handles=legend_elements, loc='upper right')
plt.savefig(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/figures/figure_A13.pdf')), format='pdf', dpi=1200, bbox_inches='tight')
