import pandas as pd
import re
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import languages
from useful_functions import check_mc_answer, check_winogrande_answer

# Use the example generations
input_path = '../../results/out-of-the-box/generations_gpt-3.5-turbo-1106_EXAMPLE.jsonl'  # TODO: Change this to desired Batch API JSONL
output_name = os.path.basename(input_path)[12:-6]  # Drop .jsonl and generations_
output_folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/out-of-the-box'))

# Get the model generations
input_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), input_path))
model_generations = pd.read_json(input_path, lines=True, encoding='utf-8')
model_generations['content'] = model_generations['response'].apply(lambda x: x['body']['choices'][0]['message']['content'])

# Get mapping of custom_id to LLM response
generations_map = dict(zip(model_generations['custom_id'], model_generations['content']))

# Get college medicine performance (including more sections means taking the average perf. across the sections
sections = [
    'college_medicine',
]

# initialize matrix of results
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    for section in sections:

        # Construct the pattern
        pattern = re.compile(rf".*-on-{lang}-mmlu-{section}.*")

        # Filter keys by pattern
        matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

        for (c_id, gen) in matching_generations:
            # Check correctness
            if check_mc_answer(c_id, gen):
                total_score += 1
            q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

# Save to CSV
matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_mmlu-college_medicine.csv'))

# Get clinical knowledge performance (including more sections means taking the average perf. across the sections
sections = [
    'clinical_knowledge',
]

# initialize matrix of results
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    for section in sections:

        # Construct the pattern
        pattern = re.compile(rf".*-on-{lang}-mmlu-{section}.*")

        # Filter keys by pattern
        matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

        for (c_id, gen) in matching_generations:
            # Check correctness
            if check_mc_answer(c_id, gen):
                total_score += 1
            q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

# Save to CSV
matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_mmlu-clinical_knowledge.csv'))

# Get virology performance (including more sections means taking the average perf. across the sections
sections = [
    'virology',
]

# initialize matrix of results
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    for section in sections:

        # Construct the pattern
        pattern = re.compile(rf".*-on-{lang}-mmlu-{section}.*")

        # Filter keys by pattern
        matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

        for (c_id, gen) in matching_generations:
            # Check correctness
            if check_mc_answer(c_id, gen):
                total_score += 1
            q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

# Save to CSV
matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_mmlu-virology.csv'))

# Get Winogrande performance
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    # Construct the pattern
    pattern = re.compile(rf".*-on-{lang}-winogrande.*")

    # Filter keys by pattern
    matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

    for (c_id, gen) in matching_generations:
        # Check correctness
        if check_winogrande_answer(c_id, gen):
            total_score += 1
        q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

# Save to CSV
matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_winogrande.csv'))

# Get Belebele performance
matrix = pd.DataFrame(
    data=0.0,
    index=[output_name],
    columns=languages
)
matrix.index.name = "Model"

for lang in languages:
    # Accumulate scores and counts for average
    total_score = 0
    q_cnt = 0

    # Construct the pattern
    pattern = re.compile(rf".*-on-{lang}-belebele.*")

    # Filter keys by pattern
    matching_generations = [(c_id, gen) for c_id, gen in generations_map.items() if pattern.match(c_id)]

    for (c_id, gen) in matching_generations:
        # Check correctness
        if check_mc_answer(c_id, gen):
            total_score += 1
        q_cnt += 1

    # Report error score if no matches found
    if q_cnt == 0:
        final_score = -1
    else:
        final_score = total_score / q_cnt
    matrix.at[output_name, lang] = round(final_score*100, 1)

matrix.to_csv(os.path.join(output_folder, f'{output_name}_on_belebele.csv'))
