import torch
from bigram import BiGramLM


# Hyperparameters
# ===============
batch_size = 32
context_length = 8
num_epochs = 3000
eval_interval = 300
learning_rate = 1e-2
device ="cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)


# Read data
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

# Data Loader - get batches of data
print(train_data[:context_length+1])

def get_batch(split):
    data = train_data if split=='train' else val_data
    idx = torch.randint(len(data) - context_length, (batch_size,))
    # print(f"Rand ids: {idx}")
    x = torch.stack([data[id:id+context_length] for id in idx])
    y = torch.stack([data[id+1:id+context_length+1] for id in idx])
    # x_train = train_data[:context_length]
    # y_train = train_data[1:context_length+1]
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


xb, yb = get_batch('train')
# print(xb.shape, yb.shape)

for b in range(batch_size):
    for t in range(context_length):
        context = xb[b, :t+1]
        target = yb[b, t]

        # print(f"{context=} {target=}")

model = BiGramLM(vocab_size)
model= model.to(device)
# logits, loss = m(xb, yb) 

# print(logits.shape, loss)

# generate new tokens - start with new line (one character) and batch_size is 1  
# idx = torch.zeros((1,1), dtype=torch.long)
# gen_token_ids = m.generate(idx, max_new_tokens=10)[0].tolist()
# print(detokenize(gen_token_ids))

# Training loop

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"{epoch=} train loss {losses['train']}; val loss {losses['val']}")

    # get batch
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # print(f"{steps=} {loss.item()}")

# generate new tokens - start with new line (one character) and batch_size is 1  
context = torch.zeros((1,1), dtype=torch.long).to(device)
gen_token_ids = model.generate(context, max_new_tokens=500)[0].tolist()
print(detokenize(gen_token_ids))