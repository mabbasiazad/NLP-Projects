'''
check the working directory before running the code
it should be the GenerativeModels folder
'''
import torch
from GenTransformer import GenTransformer
from helpers.generate_sequence import generate_sequence
from helpers.sample_batch import sample_batch
import numpy as np 
import gzip
import random
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 500
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
LR_WARMUP = 5000

'''
===========================================================================
prepare enwik8 data
===========================================================================
'''

# 1. loading the data
    
'''
I read the file in ASCII format so it assigns a number to each character.
the output of `np.frombuffer` is a list of integers each one represents a char.
Therefore, this model does not need a tokenizer. 
'''
def decode(outputs_ids):
    decoded_outputs = "".join([chr(x) for x in outputs_ids])
    return decoded_outputs

def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        # X shape : (100_000_000,)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

#note important: change the current directory to Generative Models folder in cmd line
path = './data/enwik8.gz'
data_train, data_val, data_test = enwik8(path)
data_train = torch.cat([data_train, data_val], dim = 0)

# TEST
# print(data_train)
# print(decode(data_train[0:1000]))

# 2. data batching
# I used sample_batch method in helpers directory

'''
============================================================================================
Machine learning modeling
=============================================================================================
'''
# !!! Training loop !!!
# -- We don't loop over the data, instead we sample a batch of random subsequences each time.

ISINSTANCE_SEEN = 0
NUM_ITER = 1_000_000 #very large value so you can keep running until the output looks good."
TEST_EVERY = 5

model = GenTransformer(
            k = 512, 
            heads = 8, 
            depth = 6, 
            max_seq_length = SEQ_LEN, 
            vocab_size = 256, 
            num_chars = 256
)

model = model.to(device)

loss_fn = torch.nn.NLLLoss()
optim = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
sch = torch.optim.lr_scheduler.LambdaLR(optim, lambda i: min(i / (LR_WARMUP / BATCH_SIZE), 1.0))


for i in tqdm.trange(NUM_ITER):
    inputs, targets = sample_batch(data=data_train, seq_length=SEQ_LEN, batch_size=BATCH_SIZE)
    ISINSTANCE_SEEN += inputs.size(0)
    # print("input size: ", inputs.size()) # (b, t=Seq_Len)
    # print("target size: ", targets.size()) # (b, t)    
    outputs = model(inputs)
    # print("outputs size: ", outputs.size()) #(b, t, chars=256)
    loss = loss_fn(outputs.transpose(1, 2), targets)
    # print(loss)
    optim.zero_grad()
    loss.backward()
    optim.step()
    sch.step()

    if i % TEST_EVERY == 0: 
        with torch.no_grad():
            seedfr = random.randint(0, data_test.size(0) - SEQ_LEN)
            seed = data_test[seedfr:seedfr + SEQ_LEN].to(torch.long)
            
            generate_sequence(model, seed=seed, seq_length=SEQ_LEN, verbose=True, generate_length=600)

    break
