import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = 'I love Paris'

tokens = tokenizer.tokenize(sentence)
print(tokens)

tokens = ['[CLS]'] + tokens + ['[SEP]']
print(tokens)

tokens = tokens + ['[PAD]'] + ['[PAD]']
print(tokens)

attention_mask = [1 if i != '[PAD]' else 0 for i in tokens]
print(attention_mask)

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

token_ids = torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)

hidden_rep, cls_head = model(token_ids, attention_mask= attention_mask)
# print(hidden_rep.shape)     # [1, 7, 768] => [batch_size, sequence_length, hidden_size]
# print(cls_head.shape)       # [1, 768] => [batch_size, hidden_size]

