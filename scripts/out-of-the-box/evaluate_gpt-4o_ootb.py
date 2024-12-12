import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from create_openai_batch import create_openai_batch

batch_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/llm_evaluation/evaluation_batch_full.jsonl'))
output_folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/out-of-the-box'))
desired_model = 'gpt-4o-2024-05-13'

create_openai_batch(batch_path, output_folder, desired_model=desired_model)
