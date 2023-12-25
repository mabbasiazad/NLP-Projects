import torch
from torchtext.datasets import AG_NEWS
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
Prerequisities: 
You have to install the following package to get the data
!pip install -U portalocker>=2.0.0`
"""

train_iter, test_iter = AG_NEWS()
# print(train_iter)
# train_iter = AG_NEWS(split="train") you can use it if you want to have just train or test

#simple way to get some data
# print(list(train_iter)[0])

ITER = iter(train_iter) #initiating the iterator / get the iterator
sampledata_0 = next(ITER) # print the first element of the iterator
# print(sampledata)

'''
iter() method can be used to get an iterator from any collection (such as lists)
iterating is the technical term for looping
an iterator is an object that can be used to loop through collections
'''
# get the iterator object form the list collection
sampledata_1 = next(ITER) # print the nect element of the irerator
# print(sampledata)

# for label, line in train_iter:
#     print(f"Label: {label}")
#     print(f"Line: '{line}'")
#     break
#========= Making Vocab Dictionary ================
# tokenizer = get_tokenizer("basic_english")
# counter = Counter()
# seq_length = []
# for (label, line) in train_iter:
#     seq = tokenizer(line)
#     counter.update(seq)
#     seq_length.append(len(seq))

# # for (label, line) in test_iter:
# #     seq = tokenizer(line)
# #     counter.update(seq)
# #     seq_length.append(len(seq))

# vocab = vocab(counter, min_freq=1, specials=["<pad>"])
# default_index = 0  #if the token is not in the vocab list returns default_index 
# vocab.set_default_index(default_index)

# print(vocab(['here', 'is', 'an', 'example']))

#=========== Making Vocab Dictiornay (Simple) =========================
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])

print(vocab(['here', 'is', 'an', 'example']))

#=====================================================================

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

#===================================== Batching ==========================
def collate_batch(batch):
    label_list, text_list = [], []

    for (_label, _text) in batch:
        text_coded = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        # print("size of coded list: ", text_coded.size(0))
        text_list.append(text_coded)
        label_list.append(_label)
    
    # print(text_list)
    # print(label_list)
    label_list = torch.tensor(label_list)
    max_seqlength = max([item.size(0) for item in text_list])

    text_padded = [torch.cat(item, torch.tensor(vocab["<pad>"]).expand(max_seqlength- len(item)))
                   for item in text_list] 
    text_list = torch.cat([item[None] for item in text_padded])

    return label_list.to(device), text_list.to(device) 
    
batch = [sampledata_0, sampledata_1]
collate_batch(batch)