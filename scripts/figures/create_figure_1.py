import pandas as pd
import json
from bokeh.plotting import figure, show, save, output_file, gridplot
from bokeh.models import ColumnDataSource, FixedTicker, CustomJSTickFormatter
from bokeh.io import export_svg
import os

figure_name = "figure_1.svg"

# Load the annotator data
annotator_data = pd.read_csv(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../data/translations_and_llm_responses/3. Winogrande Cultural Surveys.csv')))

# Filter to QA pairs where BOTH annotators marked it as "Good translation" or "Understandable"
annotator_data = annotator_data[(annotator_data["Translation Quality per Annotator 1"] != "Completely wrong") & (annotator_data["Translation Quality per Annotator 2"] != "Completely wrong")]

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

# load the GPT-4o data
print("Loading data...")
gpt_4o_data_0 = parse_jsonl("gpt-4o_generations_0.jsonl")
gpt_4o_data_1 = parse_jsonl("gpt-4o_generations_1.jsonl")
gpt_4o_data_2 = parse_jsonl("gpt-4o_generations_2.jsonl")

# Get the average accuracy
gpt_4o_data = (gpt_4o_data_0 + gpt_4o_data_1 + gpt_4o_data_2) / 3

# filter the annotator data to only the questions that were assessed by the GPT models
annotator_data = annotator_data.loc[gpt_4o_data.index]

# final results
data = []

# get the accuracy for each language given the "Appropriate" flag
for language in annotator_data.columns:
    # filter only the rows that are appropriate/inappropriate
    appropriate = annotator_data[annotator_data[language] == 1]
    inappropriate = annotator_data[annotator_data[language] == 0]

    # get the sample size for each
    total_appropriate = len(appropriate)
    total_inappropriate = len(inappropriate)

    # filter to only the appropriate/inappropriate questions
    gpt_4o_data_appropriate = gpt_4o_data.loc[appropriate.index]
    gpt_4o_data_inappropriate = gpt_4o_data.loc[inappropriate.index]

    # get the accuracy for appropriate/inappropriate questions
    gpt_4o_accuracy = 100*gpt_4o_data_appropriate[language].sum() / total_appropriate
    gpt_4o_accuracy_inappropriate = 100*gpt_4o_data_inappropriate[language].sum() / total_inappropriate

    # now get the english performance on the appropriate/inappropriate questions
    gpt_4o_accuracy_english = 100*gpt_4o_data_appropriate["en"].sum() / total_appropriate
    gpt_4o_accuracy_english_inappropriate = 100*gpt_4o_data_inappropriate["en"].sum() / total_inappropriate

    # append the results
    data.append({
        "language": language,
        "data": [gpt_4o_accuracy, gpt_4o_accuracy_inappropriate],
        "english": [gpt_4o_accuracy_english, gpt_4o_accuracy_english_inappropriate]
    })

# sort by the difference
data = sorted(data, key=lambda x: x["data"][0] - x["data"][1])

categories = [item["language"] for item in data]
labels = ["Appropriate", "Inappropriate"]  # legend labels for the data
colors = ["mediumseagreen", "green"]  # colors for the inappropriate and appropriate data

x_positions = []
for i in range(len(categories)):
    x_positions.append(i)

x_repeated = [x_positions[i] for i in range(len(x_positions)) for _ in range(2)]
y_values = [v for item in data for v in item["data"]]

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(
    x=x_repeated,
    y=y_values,
    legend_labels=[labels[j] for i in range(len(categories)) for j in range(len(data[i]["data"]))],
    colors=[colors[i] for j in range(len(data)) for i in range(len(data[j]["data"]))],
))

# Create a figure
p = figure(
    # title=title,
    y_range=(49, 84),
    toolbar_location=None, tools="",
    width=400, height=250
)

# Add segments
for i, cat in enumerate(categories):
    cat_x = x_positions[i]
    p.segment(x0=cat_x, y0=data[i]["data"][0], x1=cat_x, y1=data[i]["data"][1], line_width=2, color="black")

# Add circles
if labels is None:
    c = p.circle(x='x', y='y', size=15, source=source, color="blue", alpha=1, fill_color='colors', line_color=None)
else:
    c = p.circle(x='x', y='y', size=15, source=source, color="blue", alpha=1, fill_color='colors', line_color=None, legend_field='legend_labels')

# remove grid lines
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# add legend
# p.legend.label_text_font_size = '8pt'
p.legend.location = (0, 152)
p.legend.label_height = 10
p.legend.label_width = 10
p.legend.glyph_height = 15
p.legend.glyph_width = 15
p.legend.spacing = 0
p.legend.background_fill_alpha = 0.0
p.legend.border_line_alpha = 0.0

# remove x axis
p.xaxis.visible = False

# make y axis percentages
p.yaxis.formatter = CustomJSTickFormatter(code="return tick + '%'")
p.yaxis.axis_label = "Performance"
p.yaxis.major_label_text_font_size = '10pt'
p.yaxis.minor_tick_line_color = None
p.yaxis.axis_label_standoff = -1

#########
# now the bar chart underneath the lollipop chart

x_positions = []
for i in range(len(categories)):
    x_positions.append(i-0.13)
    x_positions.append(i+0.13)

target_vals = [item["data"][0] - item["data"][1] for item in data]
english_vals =  [item["english"][0] - item["english"][1] for item in data]
vals = []
for i in range(len(categories)):
    vals.append(target_vals[i])
    vals.append(english_vals[i])

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(
    x=x_positions, y=vals,
    legend_labels=["Target Language", "English" ] * len(data),
    colors=["green", "darkgrey" ] *len(data),
))

# Create a figure
p2 = figure(
    y_range=(min(vals )-1, max(vals )+1),
    x_range=(min(x_positions )-0.5, max(x_positions )+0.5),
    toolbar_location=None, tools="",
    width=400, height=120
)
p2.vbar(x='x', top='y', width=0.25, source=source, color='colors', legend_field='legend_labels')
p2.yaxis.axis_label = "Lift"

p2.xaxis.ticker = FixedTicker(ticks=list(range(len(categories))))
p2.xaxis.major_label_overrides = {pos: label for pos, label in zip(range(len(categories)), categories)}
# p2.xaxis.major_label_orientation = 0.5
p2.xaxis.major_label_text_font_size = '10pt'
p2.xaxis.axis_label = "Language"
p2.xaxis.axis_label_standoff = -5

p2.xgrid.grid_line_color = None
p2.ygrid.grid_line_color = None
p2.yaxis.formatter = CustomJSTickFormatter(code="return tick + '%'")
p2.yaxis.major_label_text_font_size = '10pt'
p2.yaxis.minor_tick_line_color = None
p2.yaxis.axis_label_standoff = -8  # adjust y axis label position

p2.legend.label_text_font_size = '8pt'
p2.legend.label_height = 10
p2.legend.label_width = 10
p2.legend.glyph_height = 10
p2.legend.glyph_width = 10
p2.legend.location = (5, 30)
p2.legend.background_fill_alpha = 0.0
p2.legend.border_line_alpha = 0.0

# set font to dejavu sans mono to match matplotlib default
font = "DejaVu Sans Mono, monospace"
p.title.text_font = font
p.yaxis.axis_label_text_font = font
p2.yaxis.axis_label_text_font = font
p2.xaxis.major_label_text_font = font
p.legend.label_text_font = font
p2.legend.label_text_font = font
p2.xaxis.axis_label_text_font = font

# set backend output to svg
p.output_backend = "svg"
p2.output_backend = "svg"

# grid layout
plot = gridplot([[p], [p2]], toolbar_location=None)
export_svg(plot, filename=os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f"../../results/figures/{figure_name}")))
