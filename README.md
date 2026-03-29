# Nitok

nitin + tokenizer :)

This is based on the GPT-4 tiktokenizer 

I originally planed to make the the original tokenizer for Dumbo
but it was extreamly slow, memory intensive and single threaded so
I switched to using sentencepiece instead

`nitok.py`
- `train(text, vocab_size, verbose=False)`
- `encode(text)`
- `decode(ids)`
- `save(filename="merges.model")`
- `load(filename="merges.model")`

### Train text

`1984.txt`

1984 by George Orwell in a single line of text

taken from https://www.textfixer.com/tools/paragraph-to-lines.php
