import json
import pandas as pd
import os


"""
This script processes the results of the GPT evaluations for Winogrande.
The output is 3 CSV files (directories listed in relation to root directory of the repository):
- results/gpt_performance/wino_evaluation_results_0.csv
- results/gpt_performance/wino_evaluation_results_1.csv
- results/gpt_performance/wino_evaluation_results_2.csv

These files map the Winogrande IDs to a 1 or 0, indicating whether the model got the answer correct.
"""

# load the winogrande ID map to convert the number IDs to the winogrande IDs
wino_id_map = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "../../data/gpt_performance/qids.txt")))
wino_id_map["id"] = wino_id_map.index
wino_id_map = wino_id_map.set_index("id")
wino_id_map = wino_id_map.to_dict()["qID"]

def parse(file_name, wino_data=None):
    if wino_data is None:
        wino_data = {}  # {ID: {llm_xh_answer, llm_zu_answer, llm_af_answer}}

    llm = file_name.split('_')[0]
    file_name = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f"../../data/gpt_performance/{file_name}"))

    # load jsonl file into list if dictionaries
    lines = []
    with open(file_name, "r") as f:
        for line in f:
            lines.append(json.loads(line))

    print(f"{llm}: Loaded lines: ", len(lines))

    # parse the response
    for line in lines:
        custom_id = line['custom_id']
        answer = line["response"]["body"]["choices"][0]["message"]["content"]
        info = custom_id.split('on-')[1].split('-')
        language = info[0]
        dataset = info[1]

        if dataset == "winogrande":
            # looks like
            # gpt-3.5-turbo-1106-on-xh-winogrande-399-answer-1

            # get the number id
            number_id = int(info[2])

            # convert number id to winogrande ID
            id = wino_id_map[number_id]

            actual_answer = custom_id[-1]

            # account for GPT giving extra information
            if "1" in answer:
                answer = "1"
            elif "2" in answer:
                answer = "2"

            if id not in wino_data:
                wino_data[id] = {
                    "id": id,
                    "answer": actual_answer
                }

            # whether it got the answer correct
            wino_data[id][f"{llm}_{language}"] = 1 if answer == actual_answer else 0

        else:
            continue

        #print(id, language, answer, actual_answer)

    return wino_data



if __name__ == "__main__":
    col_order = [
        'id', 'answer',
        'gpt-4o_en', 'gpt-4o_xh', 'gpt-4o_zu', 'gpt-4o_af', 'gpt-4o_ig', 'gpt-4o_sn', 'gpt-4o_ts', 'gpt-4o_st', 'gpt-4o_nso', 'gpt-4o_tn', 'gpt-4o_bm', 'gpt-4o_am',
        'gpt-4_en', 'gpt-4_xh', 'gpt-4_zu', 'gpt-4_af', 'gpt-4_ig', 'gpt-4_sn', 'gpt-4_ts', 'gpt-4_st', 'gpt-4_nso', 'gpt-4_tn', 'gpt-4_bm', 'gpt-4_am',
        'gpt-3.5_en', 'gpt-3.5_xh', 'gpt-3.5_zu', 'gpt-3.5_af', 'gpt-3.5_ig', 'gpt-3.5_sn', 'gpt-3.5_ts', 'gpt-3.5_st', 'gpt-3.5_nso', 'gpt-3.5_tn', 'gpt-3.5_bm', 'gpt-3.5_am'
    ]

    # Winogrande Run 0
    wino_data_0 = parse("gpt-4o_generations_0.jsonl")
    parse("gpt-4_generations_0.jsonl", wino_data_0)
    parse("gpt-3.5_generations_0.jsonl", wino_data_0)
    wino_df_0 = pd.DataFrame(wino_data_0.values())  # convert to dataframe
    wino_df_0 = wino_df_0[col_order]  # reorder columns
    wino_df_0.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "../../results/gpt_performance/wino_evaluation_results_0.csv")), index=False)  # save to CSV

    # Winogrande Run 1
    wino_data_0 = parse("gpt-4o_generations_1.jsonl")
    parse("gpt-4_generations_1.jsonl", wino_data_0)
    parse("gpt-3.5_generations_1.jsonl", wino_data_0)
    wino_df_0 = pd.DataFrame(wino_data_0.values())  # convert to dataframe
    wino_df_0 = wino_df_0[col_order]  # reorder columns
    wino_df_0.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "../../results/gpt_performance/wino_evaluation_results_1.csv")), index=False)  # save to CSV

    # Winogrande Run 2
    wino_data_0 = parse("gpt-4o_generations_2.jsonl")
    parse("gpt-4_generations_2.jsonl", wino_data_0)
    parse("gpt-3.5_generations_2.jsonl", wino_data_0)
    wino_df_0 = pd.DataFrame(wino_data_0.values())  # convert to dataframe
    wino_df_0 = wino_df_0[col_order]  # reorder columns
    wino_df_0.to_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "../../results/gpt_performance/wino_evaluation_results_2.csv")), index=False)  # save to CSV
    