import re
import pandas as pd
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('./BERT_base_TCR/', do_lower_case=False)
model = BertModel.from_pretrained('./BERT_base_TCR/')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def BERT_embedding(x):
    seq = " ".join(x)
    seq = re.sub(r"[UZOB]", "X", seq)
    encoded_input = tokenizer(seq, return_tensors='pt').to(device)
    output = model(**encoded_input)
    print(output)
    return output


dat = pd.read_csv('./data/BAP/epi_split/train.csv')
dat.columns = ['epi', 'tcr']

dat['tcr_embeds'] = None
dat['epi_embeds'] = None

for i in tqdm(range(len(dat))):
    dat.epi_embeds[i] = BERT_embedding(dat.epi[i])[0].reshape(-1,768).mean(dim=0).tolist()
    dat.tcr_embeds[i] = BERT_embedding(dat.tcr[i])[0].reshape(-1,768).mean(dim=0).tolist()


dat.to_pickle("./epi_split_train.pkl")
