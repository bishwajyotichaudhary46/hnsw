from sklearn.model_selection import train_test_split
class DataSpliting:
    def __init__(self, data):
        self.data = data

    def spliting(self):
        # this help to split data into docs and query
        documents, query = train_test_split(self.data, test_size=0.15, random_state=53)
        # this to_csv can help to save data into csv file
        documents.to_csv("results/documents.csv")
        query.to_csv("results/query.csv")
        return documents, query