# Datal loading class to load data from file
class DataLoading:
    def __init__(self, path):
        self.path = path
    
    def load_file(self):
        # use with open to open file
        with open(self.path, encoding='utf-8') as f:
            # read file and store it in data
            data = f.read()
        # return data 
        return data