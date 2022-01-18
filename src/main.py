import pickle

import pandas as pd

if __name__ == '__main__':
    with open("../asset/result02.pickle", 'rb') as f:
        result_pairs = pickle.load(f)

    result_pd = pd.DataFrame(result_pairs, columns=["real","pred"])