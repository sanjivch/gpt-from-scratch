import torch
import torch.nn as nn


torch.manual_seed(1337)
batch_size, context_length, vocab_size = 4, 8, 2 
x = torch.randn(batch_size, context_length, vocab_size)
print(x.shape)

# Slow implementation
x_avg = torch.zeros((batch_size, context_length, vocab_size))
for batch in range(batch_size):
    for ctx_i in range(context_length):
        x_prev = x[batch, :ctx_i+1]
        x_avg[batch, ctx_i] = torch.mean(x_prev, 0)



# Fast implemenation
weights = torch.tril(torch.ones(context_length, context_length))
weights = weights / weights.sum(1, keepdim=True)
print(weights)
x_avg_fast = weights @ x

print(x[0], x_avg[0], x_avg_fast[0])
#check if x_avg and x_avg_fast are same
print(torch.allclose(x_avg, x_avg_fast))