import csv
from os import path
import re
import string
from nltk.corpus import stopwords
from pyarabic.araby import strip_tashkeel, normalize_hamza, strip_tatweel

PUNC_RE = u"(\r|\n|,|\_|\"|'|\(|\)|-|:|;|\.|!|\?|؟|،|؛|{|}|\[|\]|\\\)"
SUFFIXES_RE = u"(ٍ|َ|ُ|ِ|ّ|ْ|ً)"


def get_arabic_stopwords():
    arabic_stopwords = set(stopwords.words("arabic"))
    arabic_stopwords.update(
        [
            "من",
            "شو",
            "على",
            "انت",
            "و",
            "ع",
            "حتى",
            "انتي",
            "مو",
            "اللي",
            "يعني",
            "شي",
            "مش",
            "شنو",
            "عليه",
        ]
    )
    arabic_stopwords_file = (
        "../data/stopwords/arabic_stop_words.csv"
    )
    if path.exists(arabic_stopwords_file):
        with open(arabic_stopwords_file) as f:
            arabic_stopwords.update([line.rstrip() for line in f])

    return [normalize_text(word) for word in arabic_stopwords]


def normalize_text(text):
    # remove punctuation
    text = re.sub(PUNC_RE, " ", text)
    text = re.sub(SUFFIXES_RE, "", text, flags=re.U)
    text = re.sub(u"آ", u"ا", text)
    text = re.sub(u"(آ|أ|إ|آ)", u"ا", text)
    text = re.sub(u"ى", u"ي", text)
    text = re.sub(u"ؤ", u"و", text)
    text = re.sub(u"چ", u"ك", text)
    text = re.sub(u"ه\Z", u"ة", text)
    text = re.sub(u"ﻻ", u"لا", text)
    text = strip_tashkeel(text)
    text = normalize_hamza(text)
    text = strip_tatweel(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(" +", " ", text)

    return text


ARABIC_STOPWORDS = get_arabic_stopwords()
