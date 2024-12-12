import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '..')))
from config import Config
from openai import OpenAI
from time import sleep
import io
import datetime

# Set up OpenAI credentials
openai_api_key = Config.gpt_api_key

client = OpenAI(api_key=openai_api_key)


# Function to create an OpenAI Batch API job and hang until the batch finishes
def create_openai_batch(batch_path, output_folder, desired_model='gpt-4o-2024-05-13', model_placeholder="<|MODEL|>"):
    base_batch_path = os.path.basename(batch_path)
    os.makedirs(output_folder, exist_ok=True)

    # Read eval file
    with open(batch_path, 'r', encoding='utf-8') as fp:
        eval_template = fp.read()
        # Replace placeholder with actual model name
        eval_template = eval_template.replace(model_placeholder, desired_model)

    # In case of too many requests per minute, wait and try again
    while True:
        try:
            # Create Batch job
            this_eval = io.BytesIO(eval_template.encode())
            this_id = client.files.create(
                file=this_eval,
                purpose="batch"
            ).id
            batch_id = client.batches.create(
                input_file_id=this_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            ).id

            print(f"Batch job created using file: {base_batch_path}\nBatch ID is: {batch_id}")

            break
        except Exception as e:
            print(f"An exception occurred when trying to create a batch job using file ({base_batch_path}):", e, "Waiting for 1 minute...", sep='\n')
            sleep(60)

    # Wait for batch to finish
    while True:
        try:
            this_batch = client.batches.retrieve(batch_id)
            if not this_batch.status == "completed":
                print(f"Batch ID {batch_id} created from file {base_batch_path} not yet completed. Progress: {this_batch.request_counts.completed}/{this_batch.request_counts.total}")
                print("Checking again in one minute...")
                sleep(60)
            else:
                break

        except Exception as e:
            print(f"An exception occurred while waiting for the batch job using file ({base_batch_path}) to finish:", e, "Waiting for 1 minute...", sep='\n')
            sleep(60)

    print("Batch completed! Downloading the results...")
    # Retrieve batch output
    while True:
        try:
            this_batch = client.batches.retrieve(batch_id)
            output_file_id = this_batch.output_file_id
            file_content = client.files.content(output_file_id)
            # Save the outputs
            # Get the current date and time
            now = datetime.datetime.now()

            # Format the date and time as a string
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(output_folder, f'generations_{desired_model}_{timestamp}.jsonl'), 'w', encoding='utf-8') as fp:
                fp.write(file_content.text)
            break

        except Exception as e:
            print(f"An exception occurred while trying to download the results of the batch job created using ({base_batch_path}):", e, "Waiting for 1 minute...", sep='\n')
            sleep(60)
