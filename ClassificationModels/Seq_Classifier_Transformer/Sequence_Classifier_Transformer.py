"""
Prerequisities: 
You have to install the following package to get the data AG_NEWS()
!pip install -U portalocker>=2.0.0`
"""

import torch
from torch import nn
from src.Transformer import TransformerBlock
from CTransformer import CTransformer
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
====================================================================================================
Data PreProcessing (for more detail go to PlayingWithData.py)
=====================================================================================================
"""
#1. loading the data 
train_iter, test_iter = AG_NEWS()

#2. tokenize the sequence and assign an integer to each token
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"]) # retrun default when the token is not in the vocab dictionary

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

"""
===============================================================================================
 HYPERPARAMETERS
===============================================================================================
"""
LEARNING_RATE = 0.0001
LR_WARMUP = 10_000
BATCH_SIZE = 4
EMSIZE = 128  # embedding size
NUM_HEADS = 8  # num of transformer head
DEPTH = 6  # no of transformer blocks
NUM_EPOCHS = 1
USE_VALIDATE_SET = True
NUM_CLASSES = len(set([label for (label, text) in train_iter]))
MAX_SEQ_LENGTH = 3 * max([len(text) for (lebel, text) in train_iter]) 
# multiplying by 3 for safety - longer text in test set
# this is important for position embedding
VOCAB_SIZE = len(vocab)
# this is important for token embeding


"""
=============================================================
Data Batching
=============================================================
"""
def collate_batch(batch):
    label_list, text_list = [], []

    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(
            text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)

    padded_value = vocab["<pad>"]
    max_size = max([item.size(0) for item in text_list])

    padded = [
        torch.cat(
            [item, torch.tensor([padded_value]).expand(max_size - len(item))])
        for item in text_list
    ]

    text_list = torch.cat([item[None] for item in padded])

    return label_list.to(device), text_list.to(device)


train_loader = DataLoader(
    list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

test_loader = DataLoader(
    list(test_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
#====================================================================================

if __name__ == "__main__": 
# This part is for dimensionality testing for ctransfomer class
#     b = 2
#     t = 7
#     k = 20
#     vocab_size = 100
#     max_seq_length = 10
#     input = torch.randint(vocab_size, (2, 7))
#     ctransformer = CTransformer(k, 8, 3, max_seq_length, vocab_size, 2)
#     print(ctransformer(input))
    model = CTransformer(k=EMSIZE, heads=NUM_HEADS, depth=DEPTH, 
                         max_seq_length=MAX_SEQ_LENGTH, vocab_size=VOCAB_SIZE, 
                         num_classes=NUM_CLASSES
                        )
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sch = torch.optim.lr_scheduler.LambdaLR(optim, lambda i: min(i / (LR_WARMUP / BATCH_SIZE), 1.0))
    
    NUM_EPOCHS = 1 # to test the training loop 
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (label, text) in enumerate(train_loader):
            x = text.to(device)
            y = label.to(device)

            prediction =  model(x)
            loss = loss_fn(prediction, y)
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sch.step() # change the learning rate 

            if batch_idx % 400 == 0: 
                print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], lr: {optim.param_groups[0]['lr']:.5f}, loss: {loss.item():.4f}")
        
    with torch.no_grad():
        correct = 0 
        for label, text in test_loader:
            label = label.to(device)
            text =  text.to(device)
            label = torch.tensor(label)
            output = model(text)
            prediction = torch.argmax(output, dim=1)
            correct += (prediction == label).sum().item()

    print("test accuracy %: ", (correct / len(test_loader)) * 100)

            

            

    