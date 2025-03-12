import re
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
class PreprocessData:
    def __init__(self, data):
        # here intializa data 
        self.data = data
        # here download wordnet
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        # here intialize wordnet_lemmatizer for lemmetization
        self.wordnet_lemmatizer = WordNetLemmatizer()
    
    # conversion into data frame
    def convert_to_df(self,text):
        # this help to convert list of text data into data frame
        data = pd.DataFrame(text, columns='Cleaned Data')
        return data
    
    # this function help to collect html presents in the text
    def collect_html_tags(self):
        # to match html tags
        tags = re.findall(r"</?([a-zA-Z0-9]+)[^>]*>", self.data)
        return tags

    # function for remove html tags
    def remove_html_tag(self):
        # use regularization for compiling pattern of html tag
        pattern = re.compile('<.*?>')
        # regularization substitute method substitute "" if pattern matches
        return pattern.sub(r'', self.data)
    
    # remove curly braces using  regular expression for matching
    def remove_text_inside_curly_braces(self,text):
        # \{ --> matches the opening
        # \} --> matches the closing
        # [^}] --> matches any character inside {}
        # * match first 
        new_text = re.sub(r"\{[^}]*\}", " ", text).strip()
        return new_text
    
    # remove css selector like class 
    def remove_css_selector(self,text):
        new_text = re.sub(r"[#.]\S+\s*", " ", text)
        return new_text
    
    # this function help to remove css selector which contain tag selector
    def remove_css_tag_selector(self,text):
        # Remove words starting with div, h followed by a number, table, and @media
        new_text = re.sub(r"\b(div\w*|h\d+|table\w*|p|hr|strong|img)\b\s*,?\s*", " ", text)  # Remove tags and elements
        return new_text
    
    # this function help to remove remaining css selector tag 
    def remove_css_selector_remaining(self,text, tag_list):
        # pattern to match the css element like bold 
        pattern = r"\b(" + "|".join(tag_list) + r")\b"
        
        # substitute black space if pattern matxh
        new_text = re.sub(pattern, " ", text)
        
        # use strip to remove space in left and right of text
        new_text = re.sub(r"\s+", " ", new_text).strip()
        
        return new_text
    
    # this function help to remove pattern like media screen 
    def remove_media(self,text):
        # pattern contain word like media screen ...
        pattern = r"(@media\s+screen|hrhrprint|imgabbr|}|>|\[\]|\|)"
        # substitute blanck space if pattern matches..
        new_text = re.sub(pattern, " ", text)
        # remove extra space
        new_text = re.sub(r"\s+", " ", new_text).strip()
        return new_text
    
    # this function help to remove punctuation
    def remove_punc(self,text):
        # regular exp pattern matches the letter A-Z, a-z , 0-9 if not matches this pattern substitute with blanck spaces 
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        return text
    
    # this function help to remove english stop word
    def remove_stopwords(self,text):
        # here intialize new text empty list for storing token
        new_text = []
        #this download can help to download stop word
        nltk.download('stopwords')
        # text.split() ---> help to split sentences in word level tokens
        for word in text.split():
            # here condition apply that english word contained that token then it append blanck in next 
            # otherwise  tokens append to new text
            if word in stopwords.words("english"):
                new_text.append('')
            else:
                new_text.append(word)

        # all token inside new_text assigned to new varaible x
        x = new_text[:]
        # now we clear the list new_text
        new_text.clear()
        # finally return string with join operation
        return " ".join(x)
    
    # this function help for lemmetizing token like base root with some meaning
    def lemmatization(self, text):
        # here intialize new text empty list for storing token
        new_text = []
        # text.split() ---> help to split sentences in word level tokens
        for word in text.split():
            # here lemma of word append to new text, lemmatized method help to convert into lemma
            # lemmatized method take two parameter like tokens and pos tag set as verb , it default post as noun
            new_text.append(self.wordnet_lemmatizer.lemmatize(word,pos='v'))
        # new_text all items assigned to x
        x = new_text[:]
        # here it help to clear new text list
        new_text.clear()
        return " ".join(x)
    