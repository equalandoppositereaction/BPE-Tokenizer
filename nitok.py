

class Tokenizer: #not the brightest when it comes to OOPS I suppose
    
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.ids = []

    def mostfreq(tokens):
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(ids, pair, idx): 
        newids = [] 
        i = 0 
        while i < len(ids): 
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx) 
                i += 2 
            else:
                newids.append(ids[i])
                i+=1
        return newids
    
    def train(self, text, vocab_size, verbose=False):

        tokens = text.encode('utf-8')
        tokens = list(map(int, tokens))
        num_merges = vocab_size  - 256
        self.ids = list(tokens)

        for i in range(num_merges):
            stats = mostfreq(self.ids)
            pair = max(stats, key = stats.get)
            idx = 256 + i
            if verbose:
                print(f"merge {pair} into {idx}")
            self.ids = merge(self.ids, pair, idx)
            self.merges[pair] = idx
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        print('Finished training')

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = mostfreq(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens