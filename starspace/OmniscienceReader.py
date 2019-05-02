import ast
import pandas as pd
from tqdm import tqdm

class OmniscienceReader(object):
    """
    docstring for OmniscienceReader
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
        self.om_df = pd.read_csv(self.file_path, sep='\t', encoding='utf-8')
        self.om_df = self.om_df.dropna()

        self.om_df["omniscience_label_ids"] = self.om_df["omniscience_label_ids"].apply(lambda x: ast.literal_eval(x) )
        self.om_df["omniscience_labels"] = self.om_df["omniscience_labels"].apply(lambda x: ast.literal_eval(x) )
        
        # self.om_df["omniscience_label_ids"] = self.om_df["omniscience_label_ids"].apply(lambda x: list(set(x)))
        self.om_df["omniscience_labels"] = self.om_df["omniscience_labels"].apply(lambda x: list(set(x)))
        
        self.om_df["category"] = self.om_df["file_id"].apply(lambda x: x.split(":")[0])
        self.om_df["doc_id"] = 0

        for i in tqdm(self.om_df.index):
            self.om_df.at[i, "doc_id"] = i
            if self.om_df.at[i, "category"] == "EVISE.PII":
                self.om_df.at[i, "omniscience_label_ids"] = list(map(int, self.om_df.at[i, "omniscience_label_ids"][0]))

        return self.om_df