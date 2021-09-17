import os
import string

import pandas as pd
import regex as re
import numpy as np
from itertools import product

import WorksRetrival
import yap_caller
from hebtokenizer import HebTokenizer

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, \
    roc_curve, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
import pickle

from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier().fit(X_train_tf, train_df.label)
from sklearn.linear_model import LogisticRegression
# clf=LogisticRegression().fit(X_train_tf, train_df.label)




punctuations = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

POS_TAGS = {'ABVERB','AT', 'BN', 'BNN', 'CC', 'CD', 'CDT', 'CONJ', 'COP',
            'COP_TOINFINITIVE', 'DEF', 'DT', 'DTT', 'DUMMY_AT', 'EX', 'IN',
            'INTJ', 'JJ', 'JJT', 'MD', 'NN', 'NN_S_PP', 'NNP', 'NNT', 'P',
            'POS', 'PREPOSITION', 'PRP', 'QW', 'S_PRN', 'TEMP', 'VB',
            'VB_TOINFINITIVE'}
# TODO consider adding the punctuations here as well

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
                  'כך', 'זהו', 'היו', 'היה', 'אבל', 'עם', 'כל', 'הוא', 'היא',
                  'הם', 'אם', 'מה', 'אבל', 'אתכם', 'אותם', 'אותן', 'איתי', 'איתו',
                  'אתכן', 'עמו', 'עמהן', 'עמהם', 'זאת', 'זו', 'היכן']


class Parser:

    def __init__(self, pos_window_size: int):
        self.window_size = pos_window_size
        self.pos_combos = {}
        for index, pos_combo in enumerate(product(POS_TAGS, repeat=self.window_size)):
            # TODO verify what the type of the combo is, and if it matches the
            #  pos tag combinations we get from the feature function
            self.pos_combos[pos_combo] = index
            print("Pos combo is: ", pos_combo, " of index ", index)

    def clean_text_and_reverse_order(self, text: str):
        # punctuations_regex = "(!|\"|#|$|%|&|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\]|\^|_|`|{|}|~|\t|\n|&nbsp;)+"
        # seperated_punctuations = re.sub(punctuations_regex, ' ', text)
        seperated_punctuations = text
        # print("text is: ", text)
        for punct in string.punctuation:
            seperated_punctuations = seperated_punctuations.replace(punct,
                                                                    ' ' + punct + ' ')
        cleaned_text = " ".join(seperated_punctuations.split())
        # print("after seperating punctuations the text is: ")
        # print(seperated_punctuations)
        # splitted = seperated_punctuations.split()
        # print(splitted)
        # splitted.reverse()
        # cleaned_text = " ".join(splitted)
        # print("Cleaned text is: ")
        # print(cleaned_text)
        return cleaned_text

    # YAPAPI clean text
    def clean_text(self, text:str):
        text=text.replace('\n', ' ').replace('\r', ' ')
        pattern= re.compile(r'[^א-ת\s.,!?a-zA-Z]')
        alnum_text =pattern.sub(' ', text)
        print("Cleaning the text")
        while(alnum_text.find('  ')>-1):
            print("In cleaning while loop")
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
            print("In split to sentences while loop")
            stop_points=[h for h in [i for i, e in enumerate(arr) if re.match(r"[!|.|?|;]",e)] ]
            if len(stop_points)>0:
                stop_point=min(stop_points)
                # Keep several sentence breaker as 1 word, like "...." or "???!!!"
                while True:
                    print("In split sentence inner while loop - True")
                    stop_points.remove(stop_point)
                    if len(stop_points)>1 and min(stop_points)==(stop_point+1):
                        stop_point=stop_point+1
                    else:
                        break
                # Case there is no sentence break, and this split > MAX LEN:
                sntnc=arr[:stop_point+1]
                if len(sntnc) >max_len:
                    while(len(sntnc) >max_len):
                        print("In split sentence inner while loop")
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
        print("Counting punctuations")
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
        print("Counting function words")
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
        print("Creating pos tag combos")
        pos_combo_features = np.zeros(len(self.pos_combos))
        sentences = self.split_text_to_sentences(text)
        # sentences = text.split('.')  # TODO maybe use splitting of yap_api
        for sentence in sentences:
            try:
                tokenized_text, words, pos_tags, suf_and_gen_info = yap_caller.segment_and_tag_sentence(
                    sentence)

                start_pos_combo = 0
                end_pos_combo = start_pos_combo + self.window_size
                while end_pos_combo < len(pos_tags):
                    print("In creating combos while..")
                    combo = []
                    for i in range(self.window_size):
                        combo.append(pos_tags[i + start_pos_combo])

                    pos_tuple = tuple(combo)
                    if pos_tuple in self.pos_combos:
                        print("Combo ", pos_tuple, " In combos dict")
                        pos_combo_index = self.pos_combos[pos_tuple]
                        pos_combo_features[pos_combo_index] += 1
                    start_pos_combo += 1
                    end_pos_combo = start_pos_combo + self.window_size
            except:
                print("Exception for sentence ", sentence)
                continue

        # TODO need to decide by what to normalize here -
        #  num of sent? num of words?
        pos_combo_features = pos_combo_features / len(sentences)
        return pos_combo_features

    def create_document_feature_vector(self, document_text):

        # text = document_text.replace(r'"', r'\"')
        text = self.clean_text_and_reverse_order(document_text)

        # tokenized_text = HebTokenizer().tokenize(document_text)
        # tokenized_text = ' '.join([word for (part, word) in tokenized_text])
        # print("Tokens: {}".format(len(tokenized_text.split())))
        # print("after tokenization text is: ", tokenized_text)
        # text = tokenized_text
        print("Cleaned text is: \n", text)


        # TODO maybe clean and split sentence here
        punctuation_features = self.count_punctuations(text)
        word_features = self.count_function_words(text)
        pos_features = self.create_pos_tag_features(text)

        print("punctuation feature vector is: ", punctuation_features)
        print("word feature vector is: ", word_features)
        print("pos feature vector is: ", pos_features)

        return np.concatenate([punctuation_features, word_features, pos_features])

        # return np.concatenate([punctuation_features, word_features])



def create_svm_model_and_split_data(samples_df, author_name):
    encodings = [parser.create_document_feature_vector(text) for text in samples_df['Text']]
    X_train, X_test, y_train, y_test = train_test_split(
        encodings, samples_df['Label'], test_size=0.2, random_state=42)

    svm_model_path = ".\models\\" + author_name + "_svm_model.pkl"
    if not os.path.exists(svm_model_path):
        svm_model = SGDClassifier()
        svm_model.fit(X_train, y_train)
        save_model(svm_model, svm_model_path)
    else:
        svm_model = load_model(svm_model_path)
    show_model_success_on_dataset(svm_model, svm_model_path, X_test, y_test, author_name)
    return  svm_model, X_train, X_test, y_train, y_test


def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def show_model_success_on_dataset(model, model_name, x_test, y_test, author_name):
    y_predicted = model.predict(x_test)
    print(model_name, " accuracy: ",
          accuracy_score(y_predicted, y_test), " of author ", author_name)
    print(classification_report(y_test, y_predicted))


if __name__ == '__main__':
    # attempt = pd.read_csv("authors/דוד פרישמן/בחירות.csv")
    # text = attempt["Text"][0]
    # print("text is ", text)
    parser = Parser(2)
    # features_vector = parser.create_document_feature_vector(text)
    samples_df = WorksRetrival.retrieve_samples_for_author("authors/דן אלמגור", "דן אלמגור")

    svm_model, X_train, X_test, y_train, y_test = create_svm_model_and_split_data(samples_df, "דן אלמגור")