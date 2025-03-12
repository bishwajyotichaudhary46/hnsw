import hnswlib
import numpy as np
class HNSW:
    def __init__(self,data):
        self.data = data
        self.dim = self.data.shape[1]
        self.ids = np.arange(self.data.shape[0])


    def hsnw(self):
        # hsnwlib.Index creates a non-intialized index in space  
        p = hnswlib.Index(space='cosine', dim = self.dim)
        # init_index intialize index without elements
        # max_elements --> maximum number of elements that can be stored in the structure.
        # M --> maximum number of outgoing connections in graph
        #  ef_construction --> contruction time or accuracy trade-off
        p.init_index(max_elements=10000, M=16, ef_construction=200)
        # insert data into the structure
        p.add_items(self.data, self.ids)
        # ef_serch set because trade off accuracy. higher higher value means better accuray but slower search
        p.set_ef(90)
        return p
    
    def similarity(self, vector, p):
        #retrieve nearest neighbour
        labels, distances = p.knn_query(vector, k=5)
        return labels, distances 

    


    
