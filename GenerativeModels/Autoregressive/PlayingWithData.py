import torch
import numpy as np 
import gzip
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN = 400
BATCH_SIZE = 10


'''
I read the file in ASCII format so it assigns a number to each character.
the output of `np.frombuffer` is a list of integers each one represents a char.
Therefore, this model does not need a tokenizer. 
'''
def decode(outputs_ids):
    decoded_outputs = "".join([chr(x) for x in outputs_ids])
    return decoded_outputs

def sample_batch(data, seq_length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.
    For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
    target for the model.
    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - seq_length - 1)

    # Slice out the input sequences
    seqs_inputs  = [data[start:start + seq_length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + seq_length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target


def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        # X shape : (100_000_000,)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

#note: change the current path to exprement folder
path = './data/enwik8.gz'
data_train, data_val, data_test = enwik8(path)
data_train = torch.cat([data_train, data_val], dim = 0)
print(data_train)

# print(data_train[0:15])
# print(decode(data_train[0:1000]))

# print(sample_batch(data=data_train, seq_length=SEQ_LEN, batch_size=BATCH_SIZE))

# read this file for data set creation
#  https://github.com/lucidrains/reformer-pytorch/blob/master/examples/enwik8_simple/train.py
def cycle(loader):
    while True:
        for data in loader:
            yield data

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
# generates len(data_train) / SEQ_LEN number of samples
# for this case 95000000 / 400 = 237500 number of sequence

print(len(data_train))
print(len(train_dataset))

train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
# print(len(train_loader))
print(next(train_loader))