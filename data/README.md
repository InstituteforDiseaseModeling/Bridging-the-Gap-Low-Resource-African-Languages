# Data

This directory contains all data files used for the experiments and analysis in the paper. The data is organized into the following subdirectories:


[`data/evaluation_benchmarks_afr_release/`](evaluation_benchmarks_afr_release): Contains the full translations of [Winogrande](https://github.com/allenai/winogrande), [Belebele](https://github.com/facebookresearch/belebele), and [MMLU](https://github.com/hendrycks/test) ("college medicine", "clinical knowledge", and "virology"). 
This is the folder one should use if they just want the benchmarks.

[`data/gpt_performance/`](gpt_performance): Contains raw [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) response `.jsonl` files of the GPT-3.5, GPT-4, and GPT-4o out-of-the-box responses for [Winogrande](https://github.com/allenai/winogrande), [Belebele](https://github.com/facebookresearch/belebele), and [MMLU](https://github.com/hendrycks/test). Each model was run 3 times on each benchmark, and each file named according to the following format: `<model>_generations_<run_number>.jsonl`,
where `<model>` is one of `gpt-3.5`, `gpt-4`, or `gpt-4o` and `<run_number>` indicates the trial number for the experiment, beginning at `0` (Run #1) and ending at `2` (Run #3).

[//]: # (The data in this subdirectory are stored in the following format:)

[//]: # (```json lines)

[//]: # ({)

[//]: # (  "id": "<OpenAI_batch_id>",)

[//]: # (  "custom_id": "<model-id>-on-<language_code>-<benchmark>-<benchmark_subsection>-<question_number>-answer-<correct_answer>",)

[//]: # (  "response": {)

[//]: # (    "status_code": 200,)

[//]: # (    "request_id": "<OpenAI_request_id>", )

[//]: # (    "body": {)

[//]: # (      "id": "<OpenAI_completion_id>", )

[//]: # (      "object": "chat.completion",)

[//]: # (      "created": <timestamp>,)

[//]: # (      "model": "<model_id>",)

[//]: # (      "choices": [{)

[//]: # (        "index": 0,)

[//]: # (        "message": {)

[//]: # (          "role": "assistant",)

[//]: # (          "content": "<model_output>")

[//]: # (        },)

[//]: # (        "logprobs": null,)

[//]: # (        "finish_reason": "<OpenAI_finish_reason>")

[//]: # (      }],)

[//]: # (      "usage": {)

[//]: # (        "prompt_tokens": <num_input_tokens>,)

[//]: # (        "completion_tokens": <num_output_tokens>,)

[//]: # (        "total_tokens": <num_input_and_output_tokens>)

[//]: # (      },)

[//]: # (      "system_fingerprint": "<OpenAI_system_fingerprint>")

[//]: # (    })

[//]: # (  },)

[//]: # (  "error": null)

[//]: # (})

[//]: # (```)

[`data/translations_and_llm_responses/`](translations_and_llm_responses) contains the following data files:
- [`1. Data Dictionary.pdf`](translations_and_llm_responses/1.%20Data%20Dictionary.pdf): A data dictionary describing the contents of each data file within this subdirectory and the meaning of each column within each `.csv` file.
- [`2. Winogrande Data.csv`](translations_and_llm_responses/2.%20Winogrande%20Data.csv): The raw human translation results for the Winogrande dataset.
- [`3. Winogrande Cultural Surveys.csv`](translations_and_llm_responses/3.%20Winogrande%20Cultural%20Surveys.csv): The raw human survey results for quality and cultural appropriateness assessment.
- [`4. Winogrande Upworker Profiles.csv`](translations_and_llm_responses/4.%20Winogrande%20Upworker%20Profiles.csv): The anonymized profiles and qualifications of each Upworker who was hired to do any of the Winogrande translation/assessment tasks. This is what was used to make Appendix Table 25.
- [`5. LLM Responses.csv`](translations_and_llm_responses/5.%20LLM%20Responses.csv): The complete set of all LLM responses to every single question asked to an LLM during the experiments conducted for the AAAI paper released with this code repository. This includes LLM responses from out-of-the-box experiments, fine-tuning experiments using the full fine-tuning datasets, and fine-tuning experiments using quality x quantity sampling on the fine-tuning datasets.
- [`6. Evaluation Data.csv`](translations_and_llm_responses/6.%20Evaluation%20Data.csv): The complete set of all evaluation benchmark questions, including machine-translated versions.
- [`7. Fine-Tuning Datasets.csv`](translations_and_llm_responses/7.%20Fine-Tuning%20Datasets.csv): The set of the actual fine-tuning datasets used for our experiments, given as lists of evaluation benchmark IDs that match those given in [`6. Evaluation Data.csv`](translations_and_llm_responses/6.%20Evaluation%20Data.csv). Note that the quality x quantity fine-tuning datasets reproduced for this repository (i.e. in [`results/fine-tuning_datasets/quality_x_quantity/`](../results/fine-tuning_datasets/quality_x_quantity)) may not match those given in this CSV file (due to the randomness of GPT-4o responses). As such, this CSV file should be used to select fine-tuning dataset rows if one wanted to use the exact same *rows* that we used, instead of just the same *method* to generate the rows that we used.
- [`8. MMLU Data.csv`](translations_and_llm_responses/8.%20MMLU%20Data.csv): The raw human translation results for the MMLU dataset. 
- [`9. Belebele Data.csv`](translations_and_llm_responses/9.%20Belebele%20Data.csv): The raw human translation results for the Belebele dataset. 

Note that this is the folder that should be used if one wanted to conduct additional analyses using our translation results or raw LLM responses (e.g. perhaps there is a correlation between ROUGE-1 score and LLM performance).

[`data/parquet_ready_release/`](parquet_ready_release) contains a version of [`data/evaluation_benchmarks_afr_release/`](evaluation_benchmarks_afr_release) with just our human-translated Winogrande and MMLU contributions formatted in a way suitable for HuggingFace's Dataset Viewer (i.e. it can be automatically converted to Parquet by HuggingFace). It was generated with [`create_parquet_ready_release.py`](create_parquet_ready_release.py).
