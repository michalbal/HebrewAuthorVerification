
import requests
from time import sleep


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
    json = call_yap(text)
    tokenized_text = json['tokenized_text']
    segmented_text = json['segmented_text']
    lemmas = json['lemmas']
    dep_tree = json['dep_tree']
    words, pos_tags, suf_and_gen_info = parse_dep_tree(dep_tree)
    return tokenized_text, words, pos_tags, suf_and_gen_info


