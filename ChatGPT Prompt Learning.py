import openai
import pandas as pd
import time
import spacy
import random
import re
nlp = spacy.load('en_core_web_md')

# API key
openai.api_key = ''

##################################################################################
#                             Zero-shot Simple                                   #
##################################################################################

# ChatGPT Zero-shot API function
def zero_shot_simple(note, category):
    if category == 'RAREDISEASE':
        prompt = '''Extract the exact name or names of rare diseases, which are diseases that affect a small number of people compared to the general population, from the following passage and provide them in a list'''
    elif category == 'DISEASE':
        prompt = '''Extract the exact name or names of medical diseases, which are abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both, from the following passage and provide them in a list'''
    elif category == 'SYMPTOM':
        prompt = '''Extract the exact name or names of medical symptoms, which are physical or mental problems that cannot be measured from tests or observed by a doctor, from the following passage and provide them in a list'''
    elif category == 'SIGN':
        prompt = '''Extract the exact name or names of medical signs, which are physical or mental problems that can be measured from tests or observed by a doctor, from the following passage and provide them in a list'''
    message = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = 0,
      messages=[
            {"role": "user", "content": f"{prompt}: {note}"}
        ]

    )
    return message['choices'][0]['message']['content']

# Read in test data
input_file_test = 'test_dat_category_all.xlsx'
complete_df_test = pd.read_excel(input_file_test, keep_default_na = False, na_values = '')

# Zero-shot learning
for idx, row in list(complete_df_test.iterrows()):
    note = row['note']
    category = row['category']
    if category == 'RAREDISEASE':
        prompt = '''Extract the exact name or names of rare diseases, which are diseases that affect a small number of people compared to the general population, from the following passage and provide them in a list'''
    elif category == 'DISEASE':
        prompt = '''Extract the exact name or names of medical diseases, which are abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both, from the following passage and provide them in a list'''
    elif category == 'SYMPTOM':
        prompt = '''Extract the exact name or names of medical symptoms, which are physical or mental problems that cannot be measured from tests or observed by a doctor, from the following passage and provide them in a list'''
    elif category == 'SIGN':
        prompt = '''Extract the exact name or names of medical signs, which are physical or mental problems that can be measured from tests or observed by a doctor, from the following passage and provide them in a list'''
    complete_df_test.at[idx, 'prompt'] = ("").join(prompt)

    out = zero_shot_simple(note, category)
    complete_df_test.at[idx, 'output'] = ("").join(out)
    print(idx)


complete_df_test.to_excel('zero_shot_simple_output.xlsx', sheet_name = 'Sheet1')

##################################################################################
#                             Zero-shot Structured                               #
##################################################################################

# ChatGPT Zero-shot API function
def zero_shot_structured(note, category):
    if category == 'RAREDISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of rare diseases from the input text and output them in a list.
                    ### Definition:
                    Rare diseases are defined as diseases that affect a small number of people compared to the general population.
                    ### Input Text: 
                    '''
    elif category == 'DISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of diseases from the input text and output them in a list.
                    ### Definition:
                    Diseases are defined as abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both.
                    ### Input Text: 
                    '''
    elif category == 'SYMPTOM':
        prompt = '''### Task: 
                    Extract the exact name or names of symptoms from the input text and output them in a list.
                    ### Definition:
                    Symptoms are defined as physical or mental problems that cannot be measured from tests or observed by a doctor.
                    ### Input Text: 
                    '''
    elif category == 'SIGN':
        prompt = '''### Task: 
                    Extract the exact name or names of signs from the input text and output them in a list.
                    ### Definition:
                    Signs are defined as physical or mental problems that can be measured from tests or observed by a doctor.
                    ### Input Text: 
                    '''
    message = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = 0,
      messages=[
            {"role": "user", "content": f"{prompt}: {note} \n ### Output:"}
        ]

    )
    return message['choices'][0]['message']['content']

# Read in test data
input_file_test = 'test_dat_category_all.xlsx'
complete_df_test = pd.read_excel(input_file_test, keep_default_na = False, na_values = '')

# Zero-shot learning
for idx, row in list(complete_df_test.iterrows()):
    note = row['note']
    category = row['category']
    if category == 'RAREDISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of rare diseases from the input text and output them in a list.
                    ### Definition:
                    Rare diseases are defined as diseases that affect a small number of people compared to the general population.
                    ### Input Text: 
                    '''
    elif category == 'DISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of diseases from the input text and output them in a list.
                    ### Definition:
                    Diseases are defined as abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both.
                    ### Input Text: 
                    '''
    elif category == 'SYMPTOM':
        prompt = '''### Task: 
                    Extract the exact name or names of symptoms from the input text and output them in a list.
                    ### Definition:
                    Symptoms are defined as physical or mental problems that cannot be measured from tests or observed by a doctor.
                    ### Input Text: 
                    '''
    elif category == 'SIGN':
        prompt = '''### Task: 
                    Extract the exact name or names of signs from the input text and output them in a list.
                    ### Definition:
                    Signs are defined as physical or mental problems that can be measured from tests or observed by a doctor.
                    ### Input Text: 
                    '''
    complete_df_test.at[idx, 'prompt'] = ("").join(prompt)

    out = zero_shot_structured(note, category)
    complete_df_test.at[idx, 'output'] = ("").join(out)
    print(idx)

complete_df_test.to_excel('zero_shot_structured_output.xlsx', sheet_name = 'Sheet1')

##################################################################################
#                               One-shot Simple + Random                         #
##################################################################################

# Read in training data
input_file_train = 'train_dat_category.xlsx'
complete_df_train = pd.read_excel(input_file_train, keep_default_na = False, na_values = '')

# Read in test data
input_file_test = 'test_dat_category_all.xlsx'
complete_df_test = pd.read_excel(input_file_test, keep_default_na = False, na_values = '')

def one_shot_simple_random(train_text, category, train_gold, test_text):
    if category == 'RAREDISEASE':
        category_new = "rare diseases"
        extra = "which are diseases that affect a small number of people compared to the general population,"
    elif category == 'DISEASE':
        category_new = "medical diseases"
        extra = "which are abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both,"
    elif category == 'SYMPTOM':
        category_new = "medical symptoms"
        extra = "which are physical or mental problems that cannot be measured from tests or observed by a doctor,"
    elif category == 'SIGN':
        category_new = "medical signs"
        extra = "which are physical or mental problems that can be measured from tests or observed by a doctor,"
    message = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = 0,
      messages=[
            {"role": "user", "content": f"Passage: {train_text}"},
            {"role": "user", "content": f"Extract the exact name or names of {category_new}, {extra} from this passage separated by commas: {train_gold}"},
            {"role": "user", "content": f"Passage: {test_text}"},
            {"role": "user", "content": f"Extract the exact name or names of {category_new}, {extra} from this passage separated by commas:"}
        ]
    )
    return message['choices'][0]['message']['content']

for idx, row in list(complete_df_test.iterrows()):
    test_text = row['note']

    category = row['category']

    # Obtain a list of file candidates that have this category
    train_file_candidates = list(set(complete_df_train.loc[complete_df_train['category'] == category]['file_name']))

    # Randomly choose 1 training example with this category
    train_file = random.sample(list(set(complete_df_train.loc[complete_df_train['category'] == category]['file_name'])), 1)
    train_file = ('').join(train_file)

    # Obtain the training examples
    train_text = complete_df_train.loc[(complete_df_train['file_name'] == train_file) & (complete_df_train['category'] == category)]['note']
    train_text = ''.join(train_text)

    # Obtain the training gold labels
    train_gold = complete_df_train.loc[(complete_df_train['file_name'] == train_file) & (complete_df_train['category'] == category)]['gold_All']
    train_gold = ','.join(train_gold)
    train_gold = re.sub('\n', ', ', train_gold)

    # Run few-shot API

    out = one_shot_simple_random(train_text, category, train_gold, test_text)
    out_separate = re.sub(', ', '\n', out)
    complete_df_test.at[idx, 'output'] = ("").join(out_separate)
    print(idx)

complete_df_test.to_excel('one_shot_simple_random_output.xlsx', sheet_name = 'Sheet1')


#############################################################################################
#                               One-shot Structured Random                                  #
#############################################################################################

# Read in training data
input_file_train = 'train_dat_category.xlsx'
complete_df_train = pd.read_excel(input_file_train, keep_default_na = False, na_values = '')

# Read in test data
input_file_test = 'test_dat_category_all.xlsx'
complete_df_test = pd.read_excel(input_file_test, keep_default_na = False, na_values = '')

def one_shot_structured_random(train_text, category, train_gold, test_text):
    if category == 'RAREDISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of rare diseases from the input text and output them in a list.
                    ### Definition:
                    Rare diseases are defined as diseases that affect a small number of people compared to the general population.
                    '''
    elif category == 'DISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of diseases from the input text and output them in a list.
                    ### Definition:
                    Diseases are defined as abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both.
                    '''
    elif category == 'SYMPTOM':
        prompt = '''### Task: 
                    Extract the exact name or names of symptoms from the input text and output them in a list.
                    ### Definition:
                    Symptoms are defined as physical or mental problems that cannot be measured from tests or observed by a doctor.
                    '''
    elif category == 'SIGN':
        prompt = '''### Task: 
                    Extract the exact name or names of signs from the input text and output them in a list.
                    ### Definition:
                    Signs are defined as physical or mental problems that can be measured from tests or observed by a doctor. 
                    '''
    message = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = 0,
      messages=[
            {"role": "user", "content": f"{prompt}"},
            {"role": "user", "content": f"### Input Text: {train_text}"},
            {"role": "user", "content": f"### Output: {train_gold}"},
            {"role": "user", "content": f"### Input Text: {test_text}"},
            {"role": "user", "content": f"### Output:"}
        ]
    )
    return message['choices'][0]['message']['content']

for idx, row in list(complete_df_test.iterrows()):
    test_text = row['note']

    category = row['category']

    # Obtain a list of file candidates that have this category
    train_file_candidates = list(set(complete_df_train.loc[complete_df_train['category'] == category]['file_name']))

    # Randomly choose 1 training example with this category
    train_file = random.sample(list(set(complete_df_train.loc[complete_df_train['category'] == category]['file_name'])), 1)
    train_file = ('').join(train_file)

    # Obtain the training examples
    train_text = complete_df_train.loc[(complete_df_train['file_name'] == train_file) & (complete_df_train['category'] == category)]['note']
    train_text = ''.join(train_text)

    # Obtain the training gold labels
    train_gold = complete_df_train.loc[(complete_df_train['file_name'] == train_file) & (complete_df_train['category'] == category)]['gold_All']
    train_gold = ','.join(train_gold)
    train_gold = re.sub('\n', ', ', train_gold)


    out = one_shot_structured_random(train_text, category, train_gold, test_text)
    out_separate = re.sub(', ', '\n', out)
    complete_df_test.at[idx, 'output'] = ("").join(out_separate)
    print(idx)

complete_df_test.to_excel('one_shot_structured_random_output.xlsx', sheet_name = 'Sheet1')


##################################################################################
#                             One-shot Simple Similarity                         #
##################################################################################

# Read in training data
input_file_train = 'train_dat_category.xlsx'
complete_df_train = pd.read_excel(input_file_train, keep_default_na = False, na_values = '')

# Read in test data
input_file_test = 'test_dat_category_all.xlsx'
complete_df_test = pd.read_excel(input_file_test, keep_default_na = False, na_values = '')

def one_shot_simple_similarity(train_text, category, train_gold, test_text):
    if category == 'RAREDISEASE':
        category_new = "rare diseases"
        extra = "which are diseases that affect a small number of people compared to the general population,"
    elif category == 'DISEASE':
        category_new = "medical diseases"
        extra = "which are abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both,"
    elif category == 'SYMPTOM':
        category_new = "medical symptoms"
        extra = "which are physical or mental problems that cannot be measured from tests or observed by a doctor,"
    elif category == 'SIGN':
        category_new = "medical signs"
        extra = "which are physical or mental problems that can be measured from tests or observed by a doctor,"
    message = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = 0,
      messages=[
            {"role": "user", "content": f"Passage: {train_text}"},
            {"role": "user", "content": f"Extract the exact name or names of {category_new}, {extra} from this passage separated by commas: {train_gold}"},
            {"role": "user", "content": f"Passage: {test_text}"},
            {"role": "user", "content": f"Extract the exact name or names of {category_new}, {extra} from this passage separated by commas:"}
        ]
    )
    return message['choices'][0]['message']['content']

for idx, row in list(complete_df_test.iterrows()):
    test_text = row['note']
    test_text_doc = nlp(test_text)

    category = row['category']

    # Obtain a list of file candidates that have this category
    train_file_candidates = list(set(complete_df_train.loc[complete_df_train['category'] == category]['file_name']))

    # Find the abstract that maximizes the similarity score
    sim_score = list()
    for file in train_file_candidates:
        tmp = complete_df_train.loc[(complete_df_train['file_name'] == file) & (complete_df_train['category'] == category)]['note']
        tmp = ''.join(tmp)
        train_text_doc = nlp(tmp)
        sim_score.append(test_text_doc.similarity(train_text_doc))

    # The training abstract is the one that is the most similar to the test abstract
    train_file = train_file_candidates[pd.Series(sim_score).idxmax()]

    # Obtain the training examples
    train_text = complete_df_train.loc[(complete_df_train['file_name'] == train_file) & (complete_df_train['category'] == category)]['note']
    train_text = ''.join(train_text)

    # Obtain the training gold labels
    train_gold = complete_df_train.loc[(complete_df_train['file_name'] == train_file) & (complete_df_train['category'] == category)]['gold_All']
    train_gold = ','.join(train_gold)
    train_gold = re.sub('\n', ', ', train_gold)

    out = one_shot_simple_similarity(train_text, category, train_gold, test_text)
    out_separate = re.sub(', ', '\n', out)
    complete_df_test.at[idx, 'output'] = ("").join(out_separate)
    print(idx)

complete_df_test.to_excel('one_shot_simple_similarity_output.xlsx', sheet_name = 'Sheet1')


#############################################################################################
#                            One-shot Structured Similarity                                 #
#############################################################################################

# Read in training data
input_file_train = 'train_dat_category.xlsx'
complete_df_train = pd.read_excel(input_file_train, keep_default_na = False, na_values = '')

# Read in test data
input_file_test = 'test_dat_category_all.xlsx'
complete_df_test = pd.read_excel(input_file_test, keep_default_na = False, na_values = '')

def one_shot_structured_similarity(train_text, category, train_gold, test_text):
    if category == 'RAREDISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of rare diseases from the input text and output them in a list.
                    ### Definition:
                    Rare diseases are defined as diseases that affect a small number of people compared to the general population.
                    '''
    elif category == 'DISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of medical diseases from the input text and output them in a list.
                    ### Definition:
                    Diseases are defined as abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both.
                    '''
    elif category == 'SYMPTOM':
        prompt = '''### Task: 
                    Extract the exact name or names of medical symptoms from the input text and output them in a list.
                    ### Definition:
                    Symptoms are defined as physical or mental problems that cannot be measured from tests or observed by a doctor.
                    '''
    elif category == 'SIGN':
        prompt = '''### Task: 
                    Extract the exact name or names of medical signs from the input text and output them in a list.
                    ### Definition:
                    Signs are defined as physical or mental problems that can be measured from tests or observed by a doctor. 
                    '''
    message = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = 0,
      messages=[
            {"role": "user", "content": f"{prompt}"},
            {"role": "user", "content": f"### Input Text: {train_text}"},
            {"role": "user", "content": f"### Output: {train_gold}"},
            {"role": "user", "content": f"### Input Text: {test_text}"},
            {"role": "user", "content": f"### Output:"}
        ]
    )
    return message['choices'][0]['message']['content']

for idx, row in list(complete_df_test.iterrows()):
    test_text = row['note']
    test_text_doc = nlp(test_text)

    category = row['category']

    # Obtain a list of file candidates that have this category
    train_file_candidates = list(set(complete_df_train.loc[complete_df_train['category'] == category]['file_name']))

    # Find the abstract that maximizes the similarity score
    sim_score = list()
    for file in train_file_candidates:
        tmp = complete_df_train.loc[(complete_df_train['file_name'] == file) & (complete_df_train['category'] == category)]['note']
        tmp = ''.join(tmp)
        train_text_doc = nlp(tmp)
        sim_score.append(test_text_doc.similarity(train_text_doc))

    # The training abstract is the one that is the most similar to the test abstract
    train_file = train_file_candidates[pd.Series(sim_score).idxmax()]

    # Obtain the training examples
    train_text = complete_df_train.loc[(complete_df_train['file_name'] == train_file) & (complete_df_train['category'] == category)]['note']
    train_text = ''.join(train_text)

    # Obtain the training gold labels
    train_gold = complete_df_train.loc[(complete_df_train['file_name'] == train_file) & (complete_df_train['category'] == category)]['gold_All']
    train_gold = ','.join(train_gold)
    train_gold = re.sub('\n', ', ', train_gold)

    out = one_shot_structured_similarity(train_text, category, train_gold, test_text)
    out_separate = re.sub(', ', '\n', out)
    complete_df_test.at[idx, 'output'] = ("").join(out_separate)
    print(idx)

complete_df_test.to_excel('one_shot_structured_similarity_output', sheet_name = 'Sheet1')