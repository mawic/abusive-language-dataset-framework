import json


"""
Fixes our data imports for the already downloaded Waseem json in Mishras format
"""
data_path = "./data/Waseem_2016/from_mishra/"
d_full_hydrated = "./data/Waseem_2016/201116_waseem_hovy_twitter_hydrated.json"
d_racism = data_path + "racism.json"
d_sexism = data_path + "sexism.json"
d_neither = data_path + "neither.json"

def get_user_data():
    full_data = list()
    with open(d_full_hydrated) as json_file:
        data = json.load(json_file)
    return data

def get_data():
    full_data = list()
    for data in [d_racism, d_sexism, d_neither]:
        with open(data) as f:
            for line in f:
                f_data = json.loads(line)
                full_data.append(f_data)
    assert len(full_data) == 16907
    for entry in full_data:
        entry['label'] = entry.pop('Annotation')
    return full_data

def get_data_binary():
    full_data = list()
    for data in [d_racism, d_sexism, d_neither]:
        with open(data) as f:
            for line in f:
                f_data = json.loads(line)
                full_data.append(f_data)
    assert len(full_data) == 16907
    for entry in full_data:
        label = entry.pop('Annotation')
        if label == 'none':
            entry['label'] = 'neutral'
        else:
            entry['label'] = 'abusive'
    return full_data

# some relevant fields:
# label, text, created_at, user[id], user[created_at], user[verified]
# user[followers_count], user[friends_count], user[statuses_count]
