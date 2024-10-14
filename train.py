import torch
from gpt import BiGramLM

with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print(f"{len(text)=}")

# Tokenizer - Character level tokenizer
# TODO GPT2 Tokenizer from HF/tiktoken
# unique tokens
tokens = sorted(list(set(text)))
vocab_size = len(tokens)
# print(f"{vocab_size=}")
vocab_dict = { token:i for i, token in enumerate(tokens)}
token_dict = { i:token for i, token in enumerate(tokens)}

# Encode / Decode Tokens
def tokenize(prompt):
    '''
    Returns a list of token ids given a prompt
    '''
    return [vocab_dict[token] for token in prompt]

def detokenize(token_ids):
    '''
    Returns a string given token ids
    '''
    return "".join([token_dict[id] for id in token_ids ])

# print(tokenize("hii there"))
# print(detokenize([46, 47, 47, 1, 58, 46, 43, 56, 43]))

# Load data
data = torch.tensor(tokenize(text), dtype=torch.long)
# print(data.shape)


# Split data 90-10
train_data = data[:int(0.9* len(data))]
val_data = data[int(0.9* len(data)):]

# get batches of data
torch.manual_seed(1337)
batch_size = 4
context_length = 8
print(train_data[:context_length+1])

def get_batch(split):
    data = train_data if split=='train' else val_data
    idx = torch.randint(len(data) - context_length, (batch_size,))
    # print(f"Rand ids: {idx}")
    x = torch.stack([data[id:id+context_length] for id in idx])
    y = torch.stack([data[id+1:id+context_length+1] for id in idx])
    # x_train = train_data[:context_length]
    # y_train = train_data[1:context_length+1]
    return x, y

xb, yb = get_batch('train')
# print(xb.shape, yb.shape)

for b in range(batch_size):
    for t in range(context_length):
        context = xb[b, :t+1]
        target = yb[b, t]

        # print(f"{context=} {target=}")

m = BiGramLM(vocab_size)
logits, loss = m(xb, yb) 

print(logits.shape, loss)

# generate new tokens - start with new line (one character) and batch_size is 1  
# idx = torch.zeros((1,1), dtype=torch.long)
# gen_token_ids = m.generate(idx, max_new_tokens=10)[0].tolist()
# print(detokenize(gen_token_ids))

# Training loop

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
num_epochs = 10000
for steps in range(num_epochs):

    # get batch
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"{steps=} {loss.item()}")

# generate new tokens - start with new line (one character) and batch_size is 1  
idx = torch.zeros((1,1), dtype=torch.long)
gen_token_ids = m.generate(idx, max_new_tokens=500)[0].tolist()
print(detokenize(gen_token_ids))