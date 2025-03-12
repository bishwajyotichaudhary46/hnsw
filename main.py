from component.data_loading import DataLoading
from component.preprocess import PreprocessData
from component.sentence_tokenization import SentenceTokens
from component.data_spliting import DataSpliting
from component.text_analysis_visualization import TextAnalysisVisualization
from component.sentence_embedding import SentenceEmbedding
from component.hnsw import HNSW
from component import common
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

''''Data Loading'''
print("Data Loading Started !!")
dl = DataLoading("pg75477-images.html")
data = dl.load_file()
# write loaded data in data_loaded txt file
common.write_file('data_loaded', data)
print("Data Loading Done!!!!")
''''Preprocesing Data'''
print("*"*10, "Preprocesing Data", "*"*10)
psd = PreprocessData(data)
html_tags = psd.collect_html_tags()
# collected html tag written in html_tag file
common.write_file('html_tag', str(html_tags))

# removed html tag written from text
text_0 = psd.remove_html_tag()
# saved result in removed html tag text
common.write_file('removed_html', text_0)

# remove curly braces 
text_1 = psd.remove_text_inside_curly_braces(text_0)
# saved result in removed curly braces text file
common.write_file('removed_curly', text_1)

# remove css selector 
text_2 = psd.remove_css_selector(text_1)
# saved result removed css selector classes
common.write_file('css_selector_removed',text_2)

# removed html tag selector
text_3 = psd.remove_css_tag_selector(text_2)
# saved result after removed html tag selector
common.write_file('html_selector_removed',text_3)

# removed remaining css selector
text_4 = psd.remove_css_selector_remaining(text_3,html_tags)
# saved result after removed remaining css selector
common.write_file('remaining_selector_removed',text_4)

# removed media query css
text_5 = psd.remove_media(text_4)
# saved result after removed remaining 
common.write_file("removed_media", text_5)

# conversion of sentence tokenization
sentence_tokens = SentenceTokens(text_5)
tokens = sentence_tokens.convert_sentence_tokens()
# saved result after tokenization
common.write_file("Sentences_tokens", str(tokens))


# dataframe conversion 
df = pd.DataFrame(tokens, columns=['Cleaned Data'])
# saved result in dataframe
common.save_df("sentence_tokens", df)

# removed punctuation 
data_frame = df['Cleaned Data'].apply(psd.remove_punc)
# saved result after removed punctuation
common.save_df("removed_punctuation", data_frame)

# conversion lower case
data_frame = data_frame.str.lower()
# saved result after lower casing 
common.save_df("lower_case", data_frame)

# removing stop word
data_frame = data_frame.apply(psd.remove_stopwords)
# saved result after removing stopwords
common.save_df("removed_stop", data_frame)

# conversion lemmatization
data_frame = data_frame.apply(psd.lemmatization)
# saved result after conversion into lemmatization
common.save_df("lemmatization", data_frame)

# spliting data into query and documents
ds = DataSpliting(data_frame)
docs, queries = ds.spliting()
print("Text Preprocessing Done!!!!!!!!")

'''Analysis and Visualizations'''
print("*"*10, "Analysis and Visualizations", "*"*10)
# text visualization and analysis for documents
tvs = TextAnalysisVisualization(docs)
# conversion df to list
docs_lst = tvs.conversion_sen_tokens_list()
# conversion df to tokens words
docs_tokens = tvs.conversion_word_tokens(docs_lst)

# plotted word cloud saved in results in image
tvs.plot_wordcloud(docs_tokens, "Document_word_cloud")

# calculate average word in sentence in docs 
avg_docs = tvs.cal_avg_word_sen()
# average word in sentence in docs results are saved
common.write_file('avg_word_sen_docs', 'Avg Word in Documents in sentences: '+str(avg_docs))

# top most frequently occurs word in docs are saved in result as bar graph
tvs.top_words(docs_tokens, 'Most Frequent Word in Docs')

# text visualization and analysis for query
tvs = TextAnalysisVisualization(queries)
# conversion df to list
queries_lst = tvs.conversion_sen_tokens_list()
# conversion df to tokens words
queries_tokens = tvs.conversion_word_tokens(queries_lst)

# plotted word cloud saved in results in image
tvs.plot_wordcloud(queries_tokens, "Query_word_cloud")

# calculate average word in sentence in query 
avg_docs = tvs.cal_avg_word_sen()
# average word in sentence in query results are saved
common.write_file('avg_word_sen_query', 'Avg Word in Query in sentences: '+str(avg_docs))

# top most frequently occurs word in query are saved in result as bar graph
tvs.top_words(docs_tokens, 'Most Frequent Word in Docs')


'''Embeddings'''
print("*"*10, "Embedding Started", "*"*10)
# docs embedding
de = SentenceEmbedding(docs_lst )
print("Input Encoding Started")
docs_encoded = de.encoding_input()
print("Docs Embedding Mapping Started!!")
model_output = de.embedding(docs_encoded)
docs_embedding = de.mean_pooling(model_output, docs_encoded['attention_mask'])
print("Docs Embeding save in document embedding.txt file")
de.save_embedding(docs_embedding,"Document_embedding")

# query embedding
qe = SentenceEmbedding(queries_lst)
print("Input Encoding Started")
query_encode = qe.encoding_input()
print("Query Embedding Mapping Started!!")
model_output = qe.embedding(query_encode)
query_embedding = qe.mean_pooling(model_output, query_encode['attention_mask'])
print("Query Embeding save in query embedding.txt file")
qe.save_embedding(query_embedding, "Query_embedding")

print("*"*20,"Embedding Done!!!!", "*"*20)

'''Cosine Similarity within docs'''
# it gives the similarity matrix
cosine_matrix = cosine_similarity(docs_embedding)
# get top-5 similarity indexes
top_5_indexies = common.get_top_5_indexies(cosine_matrix)
# save the result after getting top -5 similar docs
cosine_df = common.mapping_cosine(docs_data=docs_lst, top_5_similar=top_5_indexies)
print(cosine_df.head())

print("*"*20)

print("HNSW")

'''HNSW'''
hsnw = HNSW(docs_embedding)
p = hsnw.hsnw()
print("Finding similarity")
labels, distance = hsnw.similarity( query_embedding, p)

print("Saving Results")
# saved result after finding top-5 similar documents
df = common.mapping(docs_data=docs_lst, query_data=queries_lst, labels=labels)

print(df.head())

print("Finally Done!!!")