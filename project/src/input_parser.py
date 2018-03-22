import pandas as pd

class InputParser:
    def __init__(self, file_path, separator=','):
        self.data = pd.read_csv(file_path,sep=separator).as_matrix()