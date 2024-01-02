import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import gzip

class GenTransformer(nn.Module): 
    def __init__(self, k=128, heads=8, depth=6, max_seq_length=100, vocab_size=256, num_chars=256):
        """
        k: embeding size
        heads: number of transformer heads
        depth : number of transformer blocks
        max_seq_lengtn: used for positon embeding
        vocab size: our vocabs are 256 ASCII codes 
        num_char : 256 ASCII codes
        """
        super().__init__()
        self.token_emb = nn.Embedding(
            vocab_size, k)  # look up table for word embedding: (vocab_size x k)
        self.position_em = nn.Embedding(
            max_seq_length, k)  # look up table for positions
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads, mask=True))  
        
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(k, num_chars)
    
    def forward(self, x): 
        """
        param 
            x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        return: 
            A (b, t, num_chars) tensor of log-probabilities over the
                 classes (where char is the nr. of character).
        """
        token_emb = self.token_emb(x)
        b, t, k = token_emb.size()

        positions = torch.arange(t)
        position_emb = self.position_em(positions)[None, :, :].expand(b, t, k)

        x = token_emb + position_emb

        x = self.tblocks(x)
        x = self.toprobs(x)  #(b, t, n_char)

        return F.log_softmax(x, dim=2)


# if __name__ == "__main__":
#     x = torch.randint(256, (2, 5))
#     model = GenTransformer()
#     prediction = model(x)
#     print(prediction)




        
