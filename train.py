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

print(tokenize("hii there"))
print(detokenize([46, 47, 47, 1, 58, 46, 43, 56, 43]))



