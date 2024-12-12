# Function to flatten a nested list of any dimensions recursively
def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


# Function to compute ROUGE-1 score between two texts
def rouge_score_single(text1, text2):
    if not isinstance(text1, str) or not isinstance(text2, str):
        raise ValueError('Both inputs must be strings.')

    words1 = text1.strip().split()
    words2 = text2.strip().split()

    if len(words1) == 0 or len(words2) == 0:
        return 0

    word_count1 = {}
    word_count2 = {}

    for word in words1:
        word_count1[word] = word_count1.get(word, 0) + 1

    for word in words2:
        word_count2[word] = word_count2.get(word, 0) + 1

    overlap = 0

    for word in word_count1:
        if word in word_count2:
            overlap += min(word_count1[word], word_count2[word])

    precision = overlap / len(words1)
    recall = overlap / len(words2)

    if precision + recall == 0:
        return 0

    rouge1 = 2 * ((precision * recall) / (precision + recall))
    return rouge1


# Create a function to extract initials from names
def get_initials(name):
    if isinstance(name, str):
        name_pieces = name.split('-')
        return f"{name_pieces[0]}.{name_pieces[1]}."


# Define response-to-correctness functions
# For MMLU and Belebele, this function is used
def check_mc_answer(custom_id, generation):
    parsed_gen = generation.strip().replace('(', '').replace(')', '').upper()
    return len(parsed_gen) > 0 and parsed_gen[0] == custom_id[-1]  # answer is stored in last number of custom_id


# For Winogrande, this function is used
def check_winogrande_answer(custom_id, generation):
    correct_number = custom_id[-1]  # answer is stored in the last character of the custom_id
    incorrect_number = str(3 - int(correct_number))  # maps 1 to 2 and 2 to 1
    correct = correct_number in generation and incorrect_number not in generation
    return correct
