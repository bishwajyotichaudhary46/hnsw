import re
class SentenceTokens:
    def __init__(self, data):
        self.data = data
    
    # this function help to convert sentence tokens
    def convert_sentence_tokens(self):
        # regular expression help to idetentify pattern like . , ?, ! and then split
        sentences = re.split(r'(?<=\.|\?|!)\s+', self.data)
        return sentences
        