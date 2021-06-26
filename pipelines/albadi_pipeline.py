import json

def loadData():
    filename1 = './data/Albadi_2018/Hate_Speech/test.json'
    filename2 = './data/Albadi_2018/Hate_Speech/train.json'
    with open(filename1) as f:
        content = f.readlines()
    with open(filename2) as f:
        content2 = f.readlines()
    content.extend(content2)
    content = [json.loads(x.strip()) for x in content] 

    return content


def get_data_binary():
    full_data = list()
    data = loadData()
    for elem in data:
        label = 'neutral' if elem['hate'] != 'yes' else 'abusive'
        full_data.append({'text':elem['full_text'],'label': label})
    return full_data

def get_data():
    full_data = list()
    data = loadData()
    for elem in data:
        label = 'neutral' if elem['hate'] != 'yes' else 'hate'
        full_data.append({'text':elem['full_text'],
                          'label': label,
                          'id':str(elem['id_str']),
                          'user': {'id':str(elem['user']['id'])}})
    return full_data


def get_complete_data():
    full_data = list()
    data = loadData()
    for elem in data:
        label = 'neutral' if elem['hate'] != 'yes' else 'hate'
        full_data.append({'text':elem['full_text'],
                          'label': label,
                          'id':str(elem['id_str']),
                          'user': {'id':str(elem['user']['id'])}})
    return full_data


def get_available_data():
    full_data = list()
    data = loadData()
    lookup = dict()
    for elem in data:
        lookup[elem['id_str']] = 'neutral' if elem['hate'] != 'yes' else 'hate'

    # load available data
    with open('./data/Albadi_2018/Hate_Speech/210218_API_dump.json') as json_file:
        available_data = json.load(json_file)
        
    for elem in available_data:
        full_data.append({'text':elem['full_text'],
                          'label': lookup[elem['id_str']],
                          'id':str(elem['id_str']),
                          'user': {'id':str(elem['user']['id'])}})
    return full_data