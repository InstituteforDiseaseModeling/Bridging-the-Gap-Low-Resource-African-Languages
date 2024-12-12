import pandas as pd
import json
import os

table_name = "table_A11.csv"

# Load the annotator data
annotator_data = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv')))

# Filter to QA pairs where either annotator marked it as "Good translation" or "Understandable"
annotator_data = annotator_data[(annotator_data["Translation Quality per Annotator 1"] != "Completely wrong") | (annotator_data["Translation Quality per Annotator 2"] != "Completely wrong")]

# QA pairs considered appropriate if both annotators marked it as "No, the sentence is typical"
annotator_data["Appropriate"] = annotator_data.apply(lambda x: x["Cultural Appropriateness per Annotator 1"] == "No, the sentence is typical" and x["Cultural Appropriateness per Annotator 2"] == "No, the sentence is typical", axis=1)

# Now make each language get its own column, with the "Appropriate" column as the value
annotator_data = annotator_data[["Winogrande Question ID", "Target Language", "Appropriate"]]
annotator_data = annotator_data.pivot(index="Winogrande Question ID", columns="Target Language", values="Appropriate")


# load the winogrande ID map to convert the number IDs to the winogrande IDs
wino_id_map = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "../../data/gpt_performance/qids.txt")))
wino_id_map["id"] = wino_id_map.index
wino_id_map = wino_id_map.set_index("id")
wino_id_map = wino_id_map.to_dict()["qID"]


def parse_jsonl(file_name):
    """ Parse the GPT evaluation jsonl file into a dataframe """
    wino_data = {}
    llm = file_name.split('_')[0]
    file_name = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f"../../data/gpt_performance/{file_name}"))

    # load jsonl file into list of dictionaries
    lines = []
    with open(file_name, "r") as f:
        for line in f:
            lines.append(json.loads(line))

    # parse the response
    for line in lines:
        custom_id = line['custom_id']
        answer = line["response"]["body"]["choices"][0]["message"]["content"]
        info = custom_id.split('on-')[1].split('-')
        language = info[0]
        dataset = info[1]

        if dataset != "winogrande": continue

        # looks like
        # gpt-3.5-turbo-1106-on-xh-winogrande-399-answer-1

        # get the number id
        number_id = int(info[2])

        # convert number id to winogrande ID
        id = wino_id_map[number_id]

        # the correct answer
        actual_answer = custom_id[-1]

        # account for GPT sometimes giving extra information (e.g. "1. " or "Option 1.")
        if "1" in answer:
            answer = "1"
        elif "2" in answer:
            answer = "2"

        if id not in wino_data:
            wino_data[id] = {}

        # whether it got the answer correct
        wino_data[id][language] = 1 if answer == actual_answer else 0

    # convert to dataframe
    wino_data = pd.DataFrame(wino_data).T
    return wino_data


# Load all 3 runs of the GPT model evaluations
gpt_35_data_0 = parse_jsonl("gpt-3.5_generations_0.jsonl")
gpt_35_data_1 = parse_jsonl("gpt-3.5_generations_1.jsonl")
gpt_35_data_2 = parse_jsonl("gpt-3.5_generations_2.jsonl")

gpt_4_data_0 = parse_jsonl("gpt-4_generations_0.jsonl")
gpt_4_data_1 = parse_jsonl("gpt-4_generations_1.jsonl")
gpt_4_data_2 = parse_jsonl("gpt-4_generations_2.jsonl")

gpt_4o_data_0 = parse_jsonl("gpt-4o_generations_0.jsonl")
gpt_4o_data_1 = parse_jsonl("gpt-4o_generations_1.jsonl")
gpt_4o_data_2 = parse_jsonl("gpt-4o_generations_2.jsonl")

# for each model, get the average accuracy
gpt_35_data = (gpt_35_data_0 + gpt_35_data_1 + gpt_35_data_2) / 3
gpt_4_data = (gpt_4_data_0 + gpt_4_data_1 + gpt_4_data_2) / 3
gpt_4o_data = (gpt_4o_data_0 + gpt_4o_data_1 + gpt_4o_data_2) / 3

# filter the annotator data to only the questions that were assessed by the GPT models
annotator_data = annotator_data.loc[gpt_35_data.index]

# final results
results = []

# For each model, get the accuracy for each language given the "Appropriate" flag
for language in annotator_data.columns:
    # filter only the rows that are appropriate/inappropriate
    appropriate = annotator_data[annotator_data[language] == 1]
    inappropriate = annotator_data[annotator_data[language] == 0]

    # get the sample size for each
    total_appropriate = len(appropriate)
    total_inappropriate = len(inappropriate)

    # filter each model's data to only the appropriate/inappropriate questions
    gpt_35_data_appropriate = gpt_35_data.loc[appropriate.index]
    gpt_4_data_appropriate = gpt_4_data.loc[appropriate.index]
    gpt_4o_data_appropriate = gpt_4o_data.loc[appropriate.index]

    gpt_35_data_inappropriate = gpt_35_data.loc[inappropriate.index]
    gpt_4_data_inappropriate = gpt_4_data.loc[inappropriate.index]
    gpt_4o_data_inappropriate = gpt_4o_data.loc[inappropriate.index]

    # get the accuracy for each model for appropriate/inappropriate questions
    gpt_35_accuracy = 100*gpt_35_data_appropriate[language].sum() / total_appropriate
    gpt_4_accuracy = 100*gpt_4_data_appropriate[language].sum() / total_appropriate
    gpt_4o_accuracy = 100*gpt_4o_data_appropriate[language].sum() / total_appropriate

    gpt_35_accuracy_inappropriate = 100*gpt_35_data_inappropriate[language].sum() / total_inappropriate
    gpt_4_accuracy_inappropriate = 100*gpt_4_data_inappropriate[language].sum() / total_inappropriate
    gpt_4o_accuracy_inappropriate = 100*gpt_4o_data_inappropriate[language].sum() / total_inappropriate

    # now get the english performance on the appropriate questions
    gpt_35_accuracy_english = 100*gpt_35_data_appropriate["en"].sum() / total_appropriate
    gpt_4_accuracy_english = 100*gpt_4_data_appropriate["en"].sum() / total_appropriate
    gpt_4o_accuracy_english = 100*gpt_4o_data_appropriate["en"].sum() / total_appropriate

    # now get the english performance on the inappropriate questions
    gpt_35_accuracy_english_inappropriate = 100*gpt_35_data_inappropriate["en"].sum() / total_inappropriate
    gpt_4_accuracy_english_inappropriate = 100*gpt_4_data_inappropriate["en"].sum() / total_inappropriate
    gpt_4o_accuracy_english_inappropriate = 100*gpt_4o_data_inappropriate["en"].sum() / total_inappropriate

    # append the results
    results.append([
        language,
        total_appropriate, gpt_4o_accuracy, gpt_4_accuracy, gpt_35_accuracy,
        total_inappropriate, gpt_4o_accuracy_inappropriate, gpt_4_accuracy_inappropriate, gpt_35_accuracy_inappropriate,
        total_appropriate, gpt_4o_accuracy_english, gpt_4_accuracy_english, gpt_35_accuracy_english,
        total_inappropriate, gpt_4o_accuracy_english_inappropriate, gpt_4_accuracy_english_inappropriate, gpt_35_accuracy_english_inappropriate,
        gpt_4o_accuracy - gpt_4o_accuracy_inappropriate, gpt_4_accuracy - gpt_4_accuracy_inappropriate, gpt_35_accuracy - gpt_35_accuracy_inappropriate,
        gpt_4o_accuracy_english - gpt_4o_accuracy_english_inappropriate, gpt_4_accuracy_english - gpt_4_accuracy_english_inappropriate, gpt_35_accuracy_english - gpt_35_accuracy_english_inappropriate,
        (gpt_4o_accuracy - gpt_4o_accuracy_inappropriate) - (gpt_4o_accuracy_english - gpt_4o_accuracy_english_inappropriate),
        (gpt_4_accuracy - gpt_4_accuracy_inappropriate) - (gpt_4_accuracy_english - gpt_4_accuracy_english_inappropriate),
        (gpt_35_accuracy - gpt_35_accuracy_inappropriate) - (gpt_35_accuracy_english - gpt_35_accuracy_english_inappropriate),
    ])

# Create a dataframe from the results
results_df = pd.DataFrame(results, columns=[
    "language",
    "Total Appropriate", "GPT-4.0 App. Accuracy", "GPT-4 App. Accuracy", "GPT-3.5 App. Accuracy",
    "Total Inappropriate", "GPT-4.0 Inapp. Accuracy", "GPT-4 Inapp. Accuracy", "GPT-3.5 Inapp. Accuracy",
    "Total Appropriate", "GPT-4.0 En App. Accuracy", "GPT-4 En App. Accuracy", "GPT-3.5 En App. Accuracy",
    "Total Inappropriate", "GPT-4.0 En Inapp. Accuracy", "GPT-4 En Inapp. Accuracy", "GPT-3.5 En Inapp. Accuracy",
    "GPT-4.0 Accuracy Diff", "GPT-4 Accuracy Diff", "GPT-3.5 Accuracy Diff",
    "GPT-4.0 En Accuracy Diff", "GPT-4 En Accuracy Diff", "GPT-3.5 En Accuracy Diff",
    "GPT-4.0 Accuracy Diff - En", "GPT-4 Accuracy Diff - En", "GPT-3.5 Accuracy Diff - En"
])

# set the index to the language and transpose
results_df = results_df.set_index("language").T
results_df = results_df[["xh", "zu", "af", "ig", "sn", "ts", "st", "nso", "tn", "bm", "am"]]  # reorder columns

print(results_df)

# save to CSV
results_df = results_df.round(1)
results_df.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../results/tables/{table_name}')))

