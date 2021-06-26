import requests
import time
import json
from hatebase_credentials import api_key

auth_path = "https://api.hatebase.org/4-4/authenticate"
query_path = "https://api.hatebase.org/4-4/get_vocabulary"

#establish api connection and get token

r_auth = requests.post(auth_path, data = {'api_key': api_key})
assert r_auth.status_code == 200, r_auth.json()

token = r_auth.json()['result']['token']

#initialize object to write to
eng_vocab = []

#first query + get num pages
r_query = requests.post(query_path, data = {'token': token, 'page': '1', 'language':'ENG'})
assert r_query.status_code == 200, json.dumps(r_auth.json(), indent=4)
res_object = r_query.json()
num_pages = res_object['number_of_pages']
eng_vocab.extend(res_object['result'])

#go through pages and get vocab
for page in range(2, num_pages+1):
    print("Page %i of %i" % (page, num_pages))
    r_query = requests.post(query_path, data = {'token': token, 'page': str(page), 'language':'ENG'})
    assert r_query.status_code == 200, json.dumps(r_auth.json(), indent=4)
    res_object = r_query.json()
    eng_vocab.extend(res_object['result'])
    time.sleep(1)

with open('eng_vocab.json', 'w') as outfile:
    json.dump(eng_vocab, outfile, indent=4)
