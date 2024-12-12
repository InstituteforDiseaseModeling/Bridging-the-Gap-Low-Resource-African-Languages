import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import runs, llm_responses, baselines_ft, african_languages, fine_tuning_eval_benchmarks, qualities, quantities

output_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/statistics/gain_statistics.txt'))
with open(output_path, 'w') as f:
    # Redirect sys.stdout to the file
    sys.stdout = f

    print("For the 11 African languages (excluding English)...")


    # Function to get key-wise averages from a dictionary of tuples to floats, where the first value in the tuple is considered as the key
    def calculate_averages(data):
        # Dictionary to accumulate sums and counts
        sums = defaultdict(float)
        counts = defaultdict(int)

        # Iterate through the dictionary and accumulate sums and counts
        for (first, second), value in data.items():
            sums[first] += value
            counts[first] += 1

        # Calculate and print averages
        for key in sums:
            average = sums[key] / counts[key]
            assert counts[key] == 11
            print(f"Average gains when evaluating on {key}: {round(average*100,1)}%")

    # Get mono-lingual performances
    mono_lingual = llm_responses[(llm_responses['Evaluation.Target Language'] == llm_responses['Fine-Tuning.Language']) &
                                 (pd.isnull(llm_responses['Fine-Tuning.Data Partition.Data Quality'])) &
                                 (llm_responses['Evaluation.Target Language'] != 'en')]

    # Split by MMLU and Winogrande as fine-tuning dataset
    mono_lingual_mmlu = mono_lingual[mono_lingual['Fine-Tuning.Data'].str.contains('mmlu')]
    mono_lingual_wino = mono_lingual[mono_lingual['Fine-Tuning.Data'].str.contains('wino')]

    # Get mono-lingual gains by subtracting means from baselines
    monolingual_gains_mmlu = {}
    for benchmark in fine_tuning_eval_benchmarks:
        for language in african_languages:
            relevant_rows = mono_lingual_mmlu[(mono_lingual_mmlu['Evaluation.Target Language'] == language) &
                                              (mono_lingual_mmlu['Evaluation.Data'] == benchmark)]
            scores = []
            for run in runs:
                more_relevant_rows = relevant_rows[relevant_rows['Evaluation.Trial Number'] == run]
                assert more_relevant_rows.shape[0] == relevant_rows.shape[0] // 3
                scores.append(sum(more_relevant_rows['Evaluation.Model Response Was Correct'].tolist()) / more_relevant_rows.shape[0])

            mean = np.mean(scores)
            monolingual_gains_mmlu[(benchmark, language)] = mean - baselines_ft[(benchmark, language)]
    monolingual_gains_wino = {}
    for benchmark in fine_tuning_eval_benchmarks:
        for language in african_languages:
            relevant_rows = mono_lingual_wino[(mono_lingual_wino['Evaluation.Target Language'] == language) &
                                              (mono_lingual_wino['Evaluation.Data'] == benchmark)]
            scores = []
            for run in runs:
                more_relevant_rows = relevant_rows[relevant_rows['Evaluation.Trial Number'] == run]
                assert more_relevant_rows.shape[0] == relevant_rows.shape[0] // 3
                scores.append(sum(more_relevant_rows['Evaluation.Model Response Was Correct'].tolist()) / more_relevant_rows.shape[0])

            mean = np.mean(scores)
            monolingual_gains_wino[(benchmark, language)] = mean - baselines_ft[(benchmark, language)]

    # Take average of mono-lingual gains
    average_monolingual_gain = 0
    cnt = 0
    for benchmark in fine_tuning_eval_benchmarks:
        for language in african_languages:
            average_monolingual_gain += monolingual_gains_wino[(benchmark, language)] + monolingual_gains_mmlu[(benchmark, language)]
            cnt += 2

    print(f"Average Mono-Lingual Gain (total number of gains considered): {round(100*average_monolingual_gain/cnt, 1)}% ({cnt})")

    print("\nMMLU-tuned model mono-lingual gains:")
    calculate_averages(monolingual_gains_mmlu)
    print("\nWinogrande-tuned model mono-lingual gains:")
    calculate_averages(monolingual_gains_wino)

    # Get cross-lingual performance
    cross_lingual = llm_responses[(llm_responses['Evaluation.Target Language'] != llm_responses['Fine-Tuning.Language']) &
                                  (llm_responses['Model.Was Fine-Tuned']) &
                                  (pd.isnull(llm_responses['Fine-Tuning.Data Partition.Data Quality'])) &
                                  (llm_responses['Evaluation.Target Language'] != 'en')]

    # Split by MMLU and Winogrande as fine-tuning dataset
    cross_lingual_mmlu = cross_lingual[cross_lingual['Fine-Tuning.Data'].str.contains('mmlu')]
    cross_lingual_wino = cross_lingual[cross_lingual['Fine-Tuning.Data'].str.contains('wino')]

    # Get cross-lingual gains by subtracting means from baselines
    cross_lingual_gains_mmlu = {}
    for benchmark in fine_tuning_eval_benchmarks:
        for language in african_languages:
            for inner_language in african_languages:
                if language != inner_language:
                    relevant_rows = cross_lingual_mmlu[(cross_lingual_mmlu['Evaluation.Target Language'] == inner_language) &
                                                       (cross_lingual_mmlu['Fine-Tuning.Language'] == language) &
                                                       (cross_lingual_mmlu['Evaluation.Data'] == benchmark)]
                    scores = []
                    for run in runs:
                        more_relevant_rows = relevant_rows[relevant_rows['Evaluation.Trial Number'] == run]

                        assert more_relevant_rows.shape[0] == relevant_rows.shape[0] // 3
                        scores.append(sum(more_relevant_rows['Evaluation.Model Response Was Correct'].tolist()) /
                                      more_relevant_rows.shape[0])

                    mean = np.mean(scores)
                    cross_lingual_gains_mmlu[(benchmark, language, inner_language)] = mean - baselines_ft[(benchmark, inner_language)]
    cross_lingual_gains_wino = {}
    for benchmark in fine_tuning_eval_benchmarks:
        for language in african_languages:
            for inner_language in african_languages:
                if language != inner_language:
                    relevant_rows = cross_lingual_wino[(cross_lingual_wino['Evaluation.Target Language'] == inner_language) &
                                                       (cross_lingual_wino['Fine-Tuning.Language'] == language) &
                                                       (cross_lingual_wino['Evaluation.Data'] == benchmark)]
                    scores = []
                    for run in runs:
                        more_relevant_rows = relevant_rows[relevant_rows['Evaluation.Trial Number'] == run]

                        assert more_relevant_rows.shape[0] == relevant_rows.shape[0] // 3
                        scores.append(sum(more_relevant_rows['Evaluation.Model Response Was Correct'].tolist()) /
                                      more_relevant_rows.shape[0])

                    mean = np.mean(scores)
                    cross_lingual_gains_wino[(benchmark, language, inner_language)] = mean - baselines_ft[(benchmark, inner_language)]

    # Take average of cross-lingual gains
    average_cross_lingual_gain = 0
    cnt = 0
    for benchmark in fine_tuning_eval_benchmarks:
        for language in african_languages:
            for inner_language in african_languages:
                if language != inner_language:
                    average_cross_lingual_gain += cross_lingual_gains_wino[(benchmark, language, inner_language)] + cross_lingual_gains_mmlu[(benchmark, language, inner_language)]
                    cnt += 2

    # Take max of cross-lingual gains
    max_cross_lingual_gain = -1
    for benchmark in fine_tuning_eval_benchmarks:
        for language in african_languages:
            for inner_language in african_languages:
                if language != inner_language:
                    max_cross_lingual_gain = max(max_cross_lingual_gain, cross_lingual_gains_wino[(benchmark, language, inner_language)], cross_lingual_gains_mmlu[(benchmark, language, inner_language)])

    print(f"\nAverage Cross-Lingual Gain (total number of gains considered): {round(100*average_cross_lingual_gain/cnt, 1)}% ({cnt})")
    print(f"Maximum Cross-Lingual Gain (total number of gains considered): {round(100*max_cross_lingual_gain, 1)}% ({cnt})")

    # Get the quality x quantity experiments where the fine-tuning dataset was MMLU college medicine and the eval dataset was MMLU clinical knowledge (the combination with the most consistent lifts using the full fine-tuning datasets)
    mmlu_quality = llm_responses[(llm_responses['Evaluation.Target Language'] == llm_responses['Fine-Tuning.Language']) &
                                 (~pd.isnull(llm_responses['Fine-Tuning.Data Partition.Data Quality'])) &
                                 (llm_responses['Evaluation.Target Language'] != 'en') &
                                 (llm_responses['Evaluation.Data'] == 'mmlu-clinical_knowledge') &
                                 (llm_responses['Fine-Tuning.Data'] == 'mmlu-college_medicine')]

    # Get gains from 33 to 66 samples (50% to 100%)
    quantity_gains = {}
    for language in african_languages:
        for quality in qualities:
            relevant_rows_100 = mmlu_quality[(mmlu_quality['Evaluation.Target Language'] == language) &
                                             (mmlu_quality['Fine-Tuning.Data Partition.Data Quality.Percent Used'] == '100%') &
                                             (mmlu_quality['Fine-Tuning.Data Partition.Data Quality'] == quality)]
            relevant_rows_50 = mmlu_quality[(mmlu_quality['Evaluation.Target Language'] == language) &
                                             (mmlu_quality[
                                                  'Fine-Tuning.Data Partition.Data Quality.Percent Used'] == '50%') &
                                             (mmlu_quality['Fine-Tuning.Data Partition.Data Quality'] == quality)]
            scores_100 = []
            scores_50 = []
            for run in runs:
                more_relevant_rows_100 = relevant_rows_100[relevant_rows_100['Evaluation.Trial Number'] == run]
                more_relevant_rows_50 = relevant_rows_50[relevant_rows_50['Evaluation.Trial Number'] == run]

                assert more_relevant_rows_100.shape[0] == more_relevant_rows_50.shape[0] and more_relevant_rows_100.shape[0] == 265  # expected size of MMLU clinical knowledge test
                scores_100.append(sum(more_relevant_rows_100['Evaluation.Model Response Was Correct'].tolist()) /
                                  more_relevant_rows_100.shape[0])
                scores_50.append(sum(more_relevant_rows_50['Evaluation.Model Response Was Correct'].tolist()) /
                                 more_relevant_rows_50.shape[0])

            mean_100 = np.mean(scores_100)
            mean_50 = np.mean(scores_50)
            quantity_gains[(language, quality)] = mean_100 - mean_50

    # Take average of gains due to doubling quantity
    average_quantity_gain = 0
    cnt = 0
    for language in african_languages:
        for quality in qualities:
            average_quantity_gain += quantity_gains[(language, quality)]
            cnt += 1

    print(f"\nAverage 33 to 66 Quantity MMLU Gain (total number of gains considered): {round(100*average_quantity_gain/cnt, 1)}% ({cnt})")

    # Get gains due to low -> high quality
    quality_gains = {}
    for language in african_languages:
        for quantity in quantities:
            relevant_rows_high = mmlu_quality[(mmlu_quality['Evaluation.Target Language'] == language) &
                                              (mmlu_quality['Fine-Tuning.Data Partition.Data Quality.Percent Used'] == quantity) &
                                              (mmlu_quality['Fine-Tuning.Data Partition.Data Quality'] == 'high')]
            relevant_rows_low = mmlu_quality[(mmlu_quality['Evaluation.Target Language'] == language) &
                                             (mmlu_quality[
                                                  'Fine-Tuning.Data Partition.Data Quality.Percent Used'] == quantity) &
                                             (mmlu_quality['Fine-Tuning.Data Partition.Data Quality'] == 'low')]
            scores_high = []
            scores_low = []
            for run in runs:
                more_relevant_rows_high = relevant_rows_high[relevant_rows_high['Evaluation.Trial Number'] == run]
                more_relevant_rows_low = relevant_rows_low[relevant_rows_low['Evaluation.Trial Number'] == run]

                assert more_relevant_rows_high.shape[0] == more_relevant_rows_low.shape[0] and more_relevant_rows_high.shape[0] == 265
                scores_high.append(sum(more_relevant_rows_high['Evaluation.Model Response Was Correct'].tolist()) /
                                   more_relevant_rows_high.shape[0])
                scores_low.append(sum(more_relevant_rows_low['Evaluation.Model Response Was Correct'].tolist()) /
                                  more_relevant_rows_low.shape[0])

            mean_high = np.mean(scores_high)
            mean_low = np.mean(scores_low)
            quality_gains[(language, quantity)] = mean_high - mean_low

    # Take average of gains due to increasing quality
    average_quality_gain = 0
    cnt = 0
    for language in african_languages:
        for quantity in quantities:
            average_quality_gain += quality_gains[(language, quantity)]
            cnt += 1

    # Get max of gains due to increasing quality
    max_quality_gain = -1
    for language in african_languages:
        for quantity in quantities:
            max_quality_gain = max(max_quality_gain, quality_gains[(language, quantity)])

    print(f"\nAverage low to high quality MMLU Gain (total number of gains considered): {round(100*average_quality_gain/cnt, 1)}% ({cnt})")
    print(f"Maximum low to high quality MMLU Gain (total number of gains considered): {round(100*max_quality_gain, 1)}% ({cnt})")
