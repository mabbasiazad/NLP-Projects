import torch
from torch import nn
from src.Transformer import TransformerBlock

class CTransformer(nn.Module): 
    def __init__(self, k, heads, depth, max_seq_length, vocab_size, num_classes):
        # sourcery skip: for-append-to-extend, for-index-underscore
        """
        k: embeding size
        heads: number of transformer heads
        depth : number of transformer blocks
        max_seq_lengtn: used for positon embeding
        vocab size: used for word embeding 
        num_classes : number of output classes 
        """
        super().__init__()
        self.layers = []
        self.token_emb = nn.Embedding(vocab_size, k) 
        # look up table for word embedding: (vocab_size x k)
        self.pos_emb = nn.Embedding(max_seq_length, k)

        for i in range(depth):
            self.layers.append(TransformerBlock(k, heads=heads, mask=False))
        
        self.transformer = nn.Sequential(*self.layers)

        self.tologits = nn.Linear(k, num_classes)

    def forward(self, x):  # sourcery skip: inline-immediately-returned-variable
        """
        param:  A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        return: A (b, c) tensor of logits over the
                 classes (where c is the nr. of classes).
        """
        token_emb = self.token_emb(x) # x will be (b, t, k)
        b, t, k = token_emb.size()

        pos_vec = torch.arange(t)
        pos_emb = self.pos_emb(pos_vec)[None, :, :].expand(b, t, k)

        x = token_emb + pos_emb

        x =  self.transformer(x)
        x = torch.mean(x, dim=1)
        # or x = x.mean(dim=1)
        output = self.tologits(x)

        return output