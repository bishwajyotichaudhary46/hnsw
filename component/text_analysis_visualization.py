import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
class TextAnalysisVisualization:
    def __init__(self, data):
        self.data = data

    def conversion_sen_tokens_list(self):
        # list comprehensive help to listing data frame data into list
        new_data = [sent for sent in self.data]
        return new_data
        

    def conversion_word_tokens(self,sents):
        # here tokens.split() method help to split sentence into word level tokens
        new_data = [token for tokens in sents for token in tokens.split()]
        return new_data
    # this function help to plot words cloud
    def plot_wordcloud(self, words, title):
        # the function word cloud help to plot token in graph
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
        # plt.figure estimate the size of figure 
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis ---> help to hide grid 
        plt.axis('off')
        # plt.title help to make title in top of the figure
        plt.title(title)
        # plt.savefig help to save figure
        plt.savefig('results/'+title+'.png')
    
    # this function help to calculate average word in sentence
    def cal_avg_word_sen(self):
        # len function gives the total number of sentences in data
        num_sen = len(self.data)
        # conversion sentences into word level tokens
        words = [word for sent in self.data for word in sent.split()]
        # it provide total no.of words in data
        total_word = len(words)
        #print(total_word)
        # return avg of word in sentences
        return total_word/num_sen
    
    def top_words(self,words, name):
        # Counter help to count the number of repeated word
        word_freq = Counter(words)
        # most commont method help to identify top 10 most common word
        frequent_words = word_freq.most_common(10)  
        # zip function help to ziping frequency of word
        words, counts = zip(*frequent_words) 
        plt.figure(figsize=(10, 5))
        # plt.bar help to plot bar graph , word in x-axis and frequency in y-axis
        plt.bar(words, counts, color='blue')
        # put x-axis label words
        plt.xlabel('Words')
        # put y-axis label frequency
        plt.ylabel('Frequency')
        # put tiltle on the top of graph
        plt.title(name)
        # align x-label text to 45degree rotation
        plt.xticks(rotation=45)
        # this save the garph
        plt.savefig('results/'+name+'.png')
        