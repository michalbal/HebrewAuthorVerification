import os
import string

import pandas as pd
import regex as re
import numpy as np
from itertools import product

import yap_caller

punctuations = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

POS_TAGS = {'ABVERB','AT', 'BN', 'BNN', 'CC', 'CD', 'CDT', 'CONJ', 'COP',
            'COP_TOINFINITIVE', 'DEF', 'DT', 'DTT', 'DUMMY_AT', 'EX', 'IN',
            'INTJ', 'JJ', 'JJT', 'MD', 'NN', 'NN_S_PP', 'NNP', 'NNT', 'P',
            'POS', 'PREPOSITION', 'PRP', 'QW', 'S_PRN', 'TEMP', 'VB',
            'VB_TOINFINITIVE'}


function_words = ['ה', 'ו', 'ש', 'כש', 'ל', 'את', 'מתי', 'מתישהו', 'למשל',
                  'גם', 'אשר', 'לא', 'אין', 'אחר', 'אחרי', 'של', 'אני', 'על',
                  'זה', 'פי', 'כן', 'כאשר', 'איה', 'הגיע', 'בא', 'לכן', 'כי',
                  'בגלל', 'היית', 'עד', 'כאן', 'בי', 'בתוכי', 'מפני', 'האם',
                  'כך', 'זהו', 'היו', 'היה', 'אבל', 'עם', 'כל', 'הוא', 'היא',
                  'הם', 'אם', 'מה', 'אבל', 'אתכם', 'אותם', 'אותן', 'איתי', 'איתו',
                  'אתכן', 'עמו', 'עמהן', 'עמהם', 'זאת', 'זו', 'היכן', 'איפה',
                  'שכמותך', 'שכמוך', 'תהיה', 'חסל', 'חדל', 'צריך', 'צריכה',
                  'אמלא', 'אילו', 'ואילו', 'שהיה', 'שיהיה', 'חוץ', 'מזה',
                  'בפי', 'בלי', 'ללא', 'מבלי', 'אגב', 'בין', 'כה', 'אף', 'תם',
                  'נשלם', 'נגמר', 'הסתיים', 'כבר', 'איננו', 'איננה', 'אומרת',
                  'אומר', 'כשהוא', 'כשהיא', 'כשהם', 'פן', 'ההוא', 'ההיא', 'ההם',
                  'יהיו', 'יהיה', 'יהי', 'שבין', 'לבין', 'רז', 'רזי', 'רזים', 'אלה']
# TODO add charachter bigrams/ trigrams?

class Embedder:

    def __init__(self, pos_window_size: int):
        self.window_size = pos_window_size
        self.pos_combos = {}
        for index, pos_combo in enumerate(product(POS_TAGS, repeat=self.window_size)):
            self.pos_combos[pos_combo] = index

    def count_punctuations(self, text: str):
        # print("Counting punctuations")
        words = text.split()
        num_words = len(words)
        punctuation_features = np.zeros(len(string.punctuation))
        if num_words == 0:
            return punctuation_features
        for i in range(num_words):
            ch = text[i]
            if ch in string.punctuation:
                punct_i = string.punctuation.find(ch)
                punctuation_features[punct_i] += 1
        # Normalize count by number of words
        punctuation_features = punctuation_features / num_words.__float__()
        return punctuation_features

    def count_function_words(self, text):
        # print("Counting function words")
        num_of_words = len(function_words)
        features = np.zeros(num_of_words)
        words = text.split()
        for word in words:
            for i in range(num_of_words):
                if word.strip() == words[i]:
                    features[i] += 1
                    break
        features = features / len(words)
        return features

    def create_pos_tag_features(self, pos_tags_text):
        print("Creating pos tag combos")
        pos_combo_features = np.zeros(len(self.pos_combos))
        for sentence_pos in pos_tags_text:

            start_pos_combo = 0
            end_pos_combo = start_pos_combo + self.window_size
            while end_pos_combo < len(sentence_pos):
                combo = []
                for i in range(self.window_size):
                    combo.append(sentence_pos[i + start_pos_combo])

                pos_tuple = tuple(combo)
                if pos_tuple in self.pos_combos:
                    pos_combo_index = self.pos_combos[pos_tuple]
                    pos_combo_features[pos_combo_index] += 1
                start_pos_combo += 1
                end_pos_combo = start_pos_combo + self.window_size

        pos_combo_features = pos_combo_features / len(pos_tags_text)
        return pos_combo_features

    def create_document_feature_vector(self, line):
        text = line['Text']

        punctuation_features = self.count_punctuations(text)
        word_features = self.count_function_words(text)

        # Was used for pos tag features, but they take too ling
        # pos_tags = line['POS_Tags']
        # pos_features = self.create_pos_tag_features(pos_tags)
        # return np.concatenate([punctuation_features, word_features, pos_features])

        return np.concatenate([punctuation_features, word_features])


def clean_text(text):
    cleaned = text.replace("\xa0", '')
    cleaned = cleaned.replace("\xad", '')
    for punct in string.punctuation:
        cleaned = cleaned.replace(punct, ' ' + punct + ' ')
    cleaned = " ".join(cleaned.split())
    return cleaned


def split_text_to_sentences(text: str):
    sentences = []
    cleaned_text = clean_text(text)
    paragraphs = re.split('[\n|\r|\t]+', cleaned_text)
    for parag in paragraphs:
        if len(parag) > 2:
            parag_splitted = parag.split('.')
            if len(sentences) <= 1:
                sentences.append(parag)
            else:
                for sentence in parag_splitted:
                    if len(sentence) > 2:
                        sentence_full = sentence + '.'
                        sentences.append(sentence_full)
    return sentences




