import string
import regex as re
import numpy as np

punctuations = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
function_words = ['ה', 'ו', 'ש', 'כש', 'ל', 'את', 'מתי', 'מתישהו', 'למשל',
                  'גם', 'אשר', 'לא', 'אין', 'אחר', 'אחרי', 'של', 'אני', 'על',
                  'זה', 'פי', 'כן', 'כאשר', 'איה', 'הגיע', 'בא', 'לכן', 'כי',
                  'בגלל', 'היית', 'עד', 'כאן', 'בי', 'בתוכי', 'מפני', 'האם',
                  'כך', 'זהו', 'היו', 'היה', 'אבל', '', '']


def clean_text_and_reverse_order(text: str):
    punctuations_regex = "(!|\"|#|$|%|&|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\]|\^|_|`|{|}|~|\t|\n|&nbsp;)+"
    seperated_punctuations = re.sub(punctuations_regex, ' ', text)
    splitted = seperated_punctuations.split()
    print(splitted)
    splitted.reverse()
    cleaned_text = " ".join(splitted)
    print("CLeaned text is: ")
    print(cleaned_text)
    return cleaned_text


def count_punctuations(text: str):
    # Could've used python count, didnt want to because this is more efficient
    num_words = len(text)
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