import string

import pandas as pd
import regex as re
import numpy as np
from itertools import product

import enums
import yap_api
import yap_caller
from hebtokenizer import HebTokenizer

punctuations = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

function_words_for_segmented_text =\
    ['ה', 'ו', 'ש', 'כש', 'ל', 'את', 'מתי', 'מתישהו', 'למשל',
                  'גם', 'אשר', 'לא', 'אין', 'אחר', 'אחרי', 'של', 'אני', 'על',
                  'זה', 'פי', 'כן', 'כאשר', 'איה', 'הגיע', 'בא', 'לכן', 'כי',
                  'בגלל', 'היית', 'עד', 'כאן', 'בי', 'בתוכי', 'מפני', 'האם',
                  'כך', 'זהו', 'היו', 'היה', 'אבל', '', '']

# TODO: For now we send the text not segmented with ב ב and the likes.
#  We hopefully get that from the pos tags, if we don't, we will send the
#  segmented, but notice that we miss words like עליי


function_words = ['ה', 'ו', 'ש', 'כש', 'ל', 'את', 'מתי', 'מתישהו', 'למשל',
                  'גם', 'אשר', 'לא', 'אין', 'אחר', 'אחרי', 'של', 'אני', 'על',
                  'זה', 'פי', 'כן', 'כאשר', 'איה', 'הגיע', 'בא', 'לכן', 'כי',
                  'בגלל', 'היית', 'עד', 'כאן', 'בי', 'בתוכי', 'מפני', 'האם',
                  'כך', 'זהו', 'היו', 'היה', 'אבל', '', '']


class Parser:

    def __init__(self, pos_window_size: int):
        self.window_size = pos_window_size
        self.pos_tags = enums.PennTags.__members__
        self.pos_combos = {}
        for index, pos_combo in enumerate(product(self.pos_tags, repeat=self.window_size)):
            # TODO verify what the type of the combo is, and if it matches the
            #  pos tag combinations we get from the feature function
            self.pos_combos[pos_combo] = index
            print("Pos combo is: ", pos_combo, " of index ", index)

    def clean_text_and_reverse_order(self, text: str):
        # punctuations_regex = "(!|\"|#|$|%|&|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\]|\^|_|`|{|}|~|\t|\n|&nbsp;)+"
        # seperated_punctuations = re.sub(punctuations_regex, ' ', text)
        seperated_punctuations = text
        print("text is: ", text)
        for punct in string.punctuation:
            seperated_punctuations = seperated_punctuations.replace(punct,
                                                                    ' ' + punct + ' ')

        print("after seperating punctuations the text is: ")
        print(seperated_punctuations)
        splitted = seperated_punctuations.split()
        print(splitted)
        splitted.reverse()
        cleaned_text = " ".join(splitted)
        print("Cleaned text is: ")
        print(cleaned_text)
        return cleaned_text

    # YAPAPI clean text
    def clean_text(self, text:str):
        text=text.replace('\n', ' ').replace('\r', ' ')
        pattern= re.compile(r'[^א-ת\s.,!?a-zA-Z]')
        alnum_text =pattern.sub(' ', text)
        while(alnum_text.find('  ')>-1):
            alnum_text=alnum_text.replace('  ', ' ')
        return alnum_text

    def split_text_to_sentences(self, tokenized_text):
        """
        YAP better perform on sentence-by-sentence.
        Also, dep_tree is limited to 256 nodes.
        """
        max_len=150
        arr=tokenized_text.strip().split()
        sentences=[]
        # Finding next sentence break.
        while (True):
            stop_points=[h for h in [i for i, e in enumerate(arr) if re.match(r"[!|.|?]",e)] ]
            if len(stop_points)>0:
                stop_point=min(stop_points)
                # Keep several sentence breaker as 1 word, like "...." or "???!!!"
                while True:
                    stop_points.remove(stop_point)
                    if len(stop_points)>1 and min(stop_points)==(stop_point+1):
                        stop_point=stop_point+1
                    else:
                        break
                # Case there is no sentence break, and this split > MAX LEN:
                sntnc=arr[:stop_point+1]
                if len(sntnc) >max_len:
                    while(len(sntnc) >max_len):
                        sentences.append(" ".join(sntnc[:140]))
                        sntnc=sntnc[140:]
                    sentences.append(" ".join(sntnc))
                # Normal: sentence is less then 150 words...
                else:
                    sentences.append(" ".join(arr[:stop_point+1] ))
                arr=arr[stop_point+1:]
            else:
                break
        if len(arr)>0:
            sentences.append(" ".join(arr))
        return sentences

    def count_punctuations(self, text: str):
        # Could've used python count, didnt want to because this is more efficient
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
        # TODO maybe use different way to split into words, maybe mutual one between the functions
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

    def create_pos_tag_features(self, text):
        pos_combo_features = np.zeros(len(self.pos_combos))
        text = text.replace(r'"', r'\"')
        alnum_text = self.clean_text(text)
        tokenized_text = HebTokenizer().tokenize(alnum_text)
        tokenized_text = ' '.join([word for (part, word) in tokenized_text])
        print("Tokens: {}".format(len(tokenized_text.split())))
        sentences = self.split_text_to_sentences(tokenized_text)
        # sentences = text.split('.')  # TODO maybe use splitting of yap_api
        for sentence in sentences:
            tokenized_text, words, pos_tags, suf_and_gen_info = yap_caller.segment_and_tag_sentence(
                sentence)

            start_pos_combo = 0
            end_pos_combo = start_pos_combo + self.window_size
            while end_pos_combo < len(pos_tags):
                combo = []
                for i in range(self.window_size):
                    combo.append(pos_tags[i])

                pos_tuple = tuple(combo)
                if pos_tuple in self.pos_combos:
                    pos_combo_index = self.pos_combos[pos_tuple]
                    pos_combo_features[pos_combo_index] += 1
                start_pos_combo += 1
                end_pos_combo = start_pos_combo + self.window_size

        # TODO need to decide by what to normalize here -
        #  num of sent? num of words?
        pos_combo_features = pos_combo_features / len(sentences)
        return pos_combo_features

    def create_document_feature_vector(self, document_text):

        # TODO maybe clean and split sentence here
        punctuation_features = self.count_punctuations(document_text)
        word_features = self.count_function_words(document_text)
        pos_features = self.create_pos_tag_features(document_text)

        print("punctuation feature vector is: ", punctuation_features)
        print("word feature vector is: ", word_features)
        print("pos feature vector is: ", pos_features)
        print("POS feature vector len is: ", len(pos_features))

        return pd.concat([punctuation_features, word_features, pos_features])


if __name__ == '__main__':
    attempt = pd.read_csv("authors/דוד פרישמן/אהבה.csv")
    text = attempt["Text"][0]
    print("text is ", text)
    parser = Parser(2)
    features_vector = parser.create_document_feature_vector(text)
