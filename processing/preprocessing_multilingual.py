import re
import nltk
import processing.emoji as emoji
import processing.preprocessing_arabic as preprocessing_arabic

from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from nltk.tokenize import wordpunct_tokenize
nltk.download('punkt')
nltk.download('stopwords')

twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
hashtag_re = re.compile(r'\B(\#[a-zA-Z0-9]+\b)(?!;)')
url_re = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
number_re = re.compile(r'[0-9]+')
emoji_re = re.compile(emoji._EMOJI_REGEXP)
retweet_1_re = re.compile(r'^RT ')
retweet_2_re = re.compile(r'^RT ')


def preprocess_text(text,language='english'):
    if not isinstance(text,str):
        return ''
    text = clean_text(text)
    
    # language specific peprocessing
    if language == "arabic":
        text = preprocessing_arabic.normalize_text(text)
    
    # remove stop words
    text = remove_stop_words(text, language=language)
    return text

def clean_text(text):
    text = twitter_username_re.sub("",text)
    text = emoji_re.sub("",text)
    text = number_re.sub("",text)
    text = hashtag_re.sub("",text)
    text = url_re.sub("",text)
    text = retweet_1_re.sub("",text)
    text = retweet_2_re.sub("",text)
    text = text.replace('\n', ' ')
    text = text.lower()
    return text

def remove_stop_words(text,language='english'):
    if language=='arabic':
        stop_words = preprocessing_arabic.get_arabic_stopwords()
        text_tokens = wordpunct_tokenize(text)
    else:   
        stop_words = set(stopwords.words('english'))  
        # tokenize document
        text_tokens = word_tokenize(text)
    
    # remove stop words
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    # merge to sentence
    cleaned_text = " ".join(tokens_without_sw)
    return cleaned_text

