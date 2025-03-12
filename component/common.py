import pandas as pd
import numpy as np
def write_file(title, text):
    with open('results/'+title+'.txt', 'w') as f:
        f.write(text)

def save_df(title, data):
    data.to_csv('results/'+title+'.csv')

def mapping(docs_data,query_data, labels):
    top_1 = []
    top_2 = []
    top_3 = []
    top_4 = []
    top_5 = []
    query = []
    for key,label in enumerate(labels):
        query.append(query_data[key])
        # here labels are in ascending order so, we need to provide reversing indexes
        top_1.append(docs_data[label[-1]])
        top_2.append(docs_data[label[-2]])
        top_3.append(docs_data[label[-3]])
        top_4.append(docs_data[label[-4]])
        top_5.append(docs_data[label[-5]])

    data = {
        "Query": query,
        "Top-1": top_1,
        "Top-2": top_2,
        "Top-3": top_3,
        "Top-4": top_4,
        'Top-5': top_5
        
    }
    df = pd.DataFrame(data)
    df.to_csv('results/top-5 docs.csv')
    return df

def get_top_5_indexies(matrix):
    top_5_similar = []
    for similar in matrix:
        # here append sorted indexes value of higher similar top-5
        # np.argsort gives indexes in sorted order , where -6 is used to indexes upto 6 with higher similarity
        # ::-1 is used to reversed the array, it gives indexes in descending order
        top_5_similar.append(np.argsort(similar)[-6:][::-1])
    return top_5_similar

def mapping_cosine(docs_data, top_5_similar):
    top_1 = []
    top_2 = []
    top_3 = []
    top_4 = []
    top_5 = []
    docs = []
    for label in top_5_similar:
        # here used type casting int because it can give in int64 format so we need type casting
        docs.append(docs_data[int(label[0])])
        top_1.append(docs_data[int(label[1])])
        top_2.append(docs_data[int(label[2])])
        top_3.append(docs_data[int(label[3])])
        top_4.append(docs_data[int(label[4])])
        top_5.append(docs_data[int(label[5])])

    data = {
        "Docs": docs,
        "Top-1": top_1,
        "Top-2": top_2,
        "Top-3": top_3,
        "Top-4": top_4,
        'Top-5': top_5
        
    }
    df = pd.DataFrame(data)
    df.to_csv('results/top-5 cosine similarity docs.csv')
    return df
