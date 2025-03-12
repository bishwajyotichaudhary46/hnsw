# Assignment-4

## Author Bishwjyoti Chaudhary 

## How to Run this
1. Create Conda Virtual Environment
 ```
 conda create -p hns_env python==3.10 -y
 ```
 
2. Installing dependecies using pip
```
pip install -r requirements.txt
```
3. For Run 
```
python main.py
```

## All output result are saved in results folders

## There is also create setup.py file for making local package
## All code are done in modular ways


## Documentation 


### .ipynb file are inside in experiment where all experiment done before converting in modular form
### All the file python are inside in component 

Step-1: Load Data

        where used open file to open .html file with encoding utf-8

Step-2: Preprocess Data

        2.a: Removing all html tag
        2.b: Removing all css selector
        2.c: Spliting text into sentence tokenization
        2.d: Converting into dataframe 
        2.e: Removing Punctuation
        2.f: Conversion into lowercase
        2.g: Removing English Stop Word
        2.h: Apply lemmatization for conversion into lemma words

Step-3: Data Spliting

        Data Spliting using sklern library into documents and query

Step-4: Analysis And Visualization

        4.a: First conversion into sentence tokens into word tokens.
        4.b: Plot Word Cloud of Seperatly Documents and Queries
        4.c: Plot Bar graph of top most frequents occuring word in Documents and Queries.
        4.d: Calculate of Average No. Words in Sentences

Step-5: Embedding Using Bert

        5.a: Using transformer library for Embedding.
        5.b: Downloading Pretrained tokenizer and model from huggingface model hub i.e deepset/sentence_bert, where tokenizer used to conversion of encoding sentences and model used for getting embedding of sentences.
        5.c: After getting embedding of sentence then pass to mean pooling  to get correct averaging of embedding.
        5.d: Save embeding into .txt file

Step-6: Cosine Similarity of Documents

        6.a: Using sklearn consine similarity method.
        6.b: Cosine Similarity pass to arg.sort method to get sorted order indexes.
        6.c: Save result in top-5 cosine similar doc.txt

Step-7: HSNW

        7.a: Using a hnswlib for indexing embedding vector to fast and efficient way to approximate nearest neighbout search based on the HNSW graph Algorithm.


        7.b: hnswlib.Index --> creates non-intialized index in space. Where parameters space and dim . Space take which type of similarity you want like l2, ip or cosine and dim is the shape of single embedding.

        7.c: init_index --> intialize index with no elements. where parameters are max_elements, ef_construction and M. max_elements refers to  maximum number of elements in structure to store, ef_construction defines construction time or accuracy trade-off and M refers to maximum numbers of outgoing connections in graph.

        7.d: add_item --> add data into structure. where parameters data and ids. ids be the unique values array whose length must be equal to the data length.

        7.e: ef_set --> mean ef_search (exploration factor for seaech), that used for accuracy or speed trade off. It takes value, higher value means better accuracy but slower search.

        7.f: knn_query --> to find k-nearest neighbours. where it takes paremeter like data, and k . K takes number of neighour which you want. It returns a distance and labels.





