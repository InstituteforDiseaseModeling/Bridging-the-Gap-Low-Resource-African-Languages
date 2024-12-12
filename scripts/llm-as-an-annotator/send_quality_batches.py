import sys
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from create_openai_batch import create_openai_batch

# Set up input and output paths as well as GPT model name
batch_path_0 = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__),
                                             '../../results/llm-as-an-annotator/gpt-4o_quality_batch_0.jsonl'))
batch_path_1 = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__),
                                             '../../results/llm-as-an-annotator/gpt-4o_quality_batch_1.jsonl'))
output_folder = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator'))
desired_model = 'gpt-4o-2024-05-13'

# Run batches in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(create_openai_batch, batch_path_0, output_folder, desired_model=desired_model),
        executor.submit(create_openai_batch, batch_path_1, output_folder, desired_model=desired_model)
    ]

    # Wait for all the futures to complete
    for future in as_completed(futures):
        future.result()  # This will raise an exception if one occurred during the function call

# Rename files to remove timestamps
resulting_files = glob(os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm-as-an-annotator/generations_gpt-4o*')))
resulting_files.sort()
for index, filename in enumerate(resulting_files):
    os.rename(filename, filename.replace(os.path.basename(filename), f'quality_generations_gpt-4o_{index}.jsonl'))
