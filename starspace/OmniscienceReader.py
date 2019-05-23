import ast
import pandas as pd
from tqdm import tqdm
from gensim.parsing import preprocessing
from gensim.utils import tokenize

def document_preprocess(text):
    first = text.encode('ascii', 'ignore').decode('utf-8').lower()
    second = preprocessing.remove_stopwords(first)
    third = preprocessing.strip_punctuation(second)
    fourth = preprocessing.strip_short(preprocessing.strip_numeric(third))
    return fourth

class OmniscienceReader(object):
    """
    docstring for OmniscienceReader]
    [x] preprocess raw text: gensim preprocesser : stop words + stemming + lemma + tokenize + -num
    [x] raw text -> word2vec using fasttext
    [x] avg word2vec across docs to create doc2vec
    [x] distirbution of classes per document
    """
    def __init__(self, file_path):
        super(OmniscienceReader, self).__init__()
        self.file_path = file_path
        self.preprocess()

    def preprocess(self):
        temp = pd.read_csv(self.file_path, sep='\t', encoding='utf-8')
        temp = temp.dropna()

        # temp["doc"] = temp["abstract"].apply(lambda x: document_preprocess(x))
        temp["doc"] = temp["abstract"].apply(lambda x: document_preprocess(x))
        temp["label_id"] = temp["label_id"].apply(lambda x: ast.literal_eval(x))
        temp["labels"] = temp["labels"].apply(lambda x: ast.literal_eval(x))
        temp["labels"] = temp["labels"].apply(lambda x: [i.lower().replace(" ", "-") for i in x])

        self.om_df = temp
        
        # self.om_df.to_csv(self.file_path, sep='\t', encoding='utf-8', index=False)
        
        return self.om_df

