# BPE-Tokenizer

## nitok
nitin + tokenizer :)

based on GPT-2 and GPT-4 tokenizer 

`nitok.py`
- `def train(self, text, vocab_size, verbose=False)`
- `def encode(self, text)`
- `def decode(self, ids)`

## its extremely memory heavy
22MB of tinystories validation set takes about 1.9GB of memory
about 2GB of training set completely fills and crashes upto 47GB ram machine
I believe this is only during pretokenization and training
still its not worth using in any project scenario, will be shifting to huggingface tokenizer
