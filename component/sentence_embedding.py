from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class SentenceEmbedding:
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
        self.model = AutoModel.from_pretrained('deepset/sentence_bert')
    
    def encoding_input(self):
        # tonkenizer help to encode the data 
        # parameters like padding and trucation which to fixed input in same length
        # maxlength help to make same dimension of all tensor encoded
        # return tensor 'pt' it means pytorch torch format.
        encoded_input = self.tokenizer(self.data, padding=True, truncation=True,max_length=512, return_tensors='pt')
        return encoded_input
    
    def embedding(self, encoded_input):
        # torch.no_grad() help to disable gradient 
        with torch.no_grad():
           # here encoded input are passed to model to get prediction of embedding
            model_output = self.model(**encoded_input)

        return model_output
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    

    def save_embedding(self, emed, title):
        np.savetxt("results/"+title+".txt", emed)

        


    
        