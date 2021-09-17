import pandas as pd
import stanza
import requests
import json
from time import sleep
from pandas.io.json import json_normalize
from enums import *
from hebtokenizer import HebTokenizer
import yap_api

yap_token = "4c50c422eeaeeae2decdcb28f6548c7f"


def call_yap(text):
    url = f'https://www.langndata.com/api/heb_parser?token={yap_token}'
    _json = '{"data":"' + text.strip() + '"}'
    sleep(3)
    r = requests.post(url, data=_json.encode('utf-8'), headers={
        'Content-type': 'application/json; charset=utf-8'})
    json_obj = r.json()
    return json_obj


def parse_dep_tree(dep_tree):
    print("Parsing dep tree")
    words = []
    pos_tags = []
    suf_and_gen_info = []
    for node in dep_tree.values():
        words.append(node['word'])
        pos_tags.append(node['pos'])
        suf_and_gen_info.append(node['empty'])

    return words, pos_tags, suf_and_gen_info


def segment_and_tag_sentence(text):
    # text = text.replace(r'"', r'\"')
    json = call_yap(text)
    # print("json is:\n", json)
    tokenized_text = json['tokenized_text']
    segmented_text = json['segmented_text']
    lemmas = json['lemmas']
    dep_tree = json['dep_tree']
    # print("tokenized_text: \n", tokenized_text, "\n segmented_text:\n",
    #       segmented_text, "\nlemmas:\n", lemmas)
    words, pos_tags, suf_and_gen_info = parse_dep_tree(dep_tree)
    # print("words are: ", words, " \npos tags are ", pos_tags)
    return tokenized_text, words, pos_tags, suf_and_gen_info


