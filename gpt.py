import torch
import torch.nn as nn
from torch.nn import functional as F 

class BiGramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # Batch_size x context_length x vocab_size
        if targets==None:
            loss = None
        else:
            batch_size, context_length, vocab_size = logits.shape
            logits = logits.view(batch_size*context_length, vocab_size)
            targets = targets.view(batch_size*context_length)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            # Get the last id - will be used to predict next token
            logits = logits[:,-1, :] # batch_size x vocab_size

            # Softmax
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        
        return idx

vocab_size = 65

# m = BiGramLM(vocab_size)
# out = m(xb, yb) 