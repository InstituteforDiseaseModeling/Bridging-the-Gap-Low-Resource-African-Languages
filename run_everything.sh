#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Running all scripts that do not incur a monetary cost"
echo "Recreating tables"
python3 scripts/tables/create_table_2.py > /dev/null 2>&1
python3 scripts/tables/create_tables_A20-A27.py > /dev/null 2>&1
find scripts/tables -name "*.py" | xargs -n 1 -P 8 python3 > /dev/null 2>&1

echo "Recreating figures"
find scripts/figures -name "*.py" | xargs -n 1 -P 8 python3 > /dev/null 2>&1

echo "Recreating statistics"
find scripts/statistics -name "*.py" | xargs -n 1 -P 8 python3 > /dev/null 2>&1

echo "Recreating LLM evaluation batches"
find scripts/llm_evaluation -name "*.py" | xargs -n 1 -P 8 python3 > /dev/null 2>&1

echo "Recreating LLM-as-an-annotator script results"
find scripts/llm-as-an-annotator -name "*.py" ! -name "*send*" | xargs -n 1 -P 8 python3 > /dev/null 2>&1

echo "Recreating GPT performance results"
find scripts/gpt_performance -name "*.py" | xargs -n 1 -P 8 python3 > /dev/null 2>&1

echo "Recreating fine-tuning datasets"
find scripts/fine-tuning_datasets -name "*.py" | xargs -n 1 -P 8 python3 > /dev/null 2>&1

echo "Recreating GPT-3.5 out-of-the-box example results"
python3 scripts/out-of-the-box/process_gpt_generations.py > /dev/null 2>&1

echo "Recreating o1-related results"
python3 scripts/o1-related/process_o1_generations.py > /dev/null 2>&1
python3 scripts/o1-related/add_o1-mini_to_table_2.py > /dev/null 2>&1

echo "Everything has completed successfully"
