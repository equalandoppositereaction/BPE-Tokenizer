import regex as re
class Tokenizer: #not the brightest when it comes to OOPS I suppose
    
  def __init__(self):
    self.vocab = {idx: bytes([idx]) for idx in range(256)}
    self.vocab.update({256 : b"<|endoftext|>", 257 : b"<|system|>", 258 : b"<|user|>"})
    self.merges = {}
    self.ids = []
    self.vocab_size = None

    self.special_tokens = ["<|endoftext|>", "<|system|>", "<|user|>"]
    special_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
    self.num_special = len(self.special_tokens)

    ptr = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    self.ptr = re.compile(f"{special_pattern}|{ptr}")


  def mostfreq(self, chunks):
    import collections
    counts = collections.Counter()
    for tokens in chunks:
      counts.update(zip(tokens, tokens[1:]))
    return dict(counts)


  def _pair_counts(self, tokens):
    import collections
    return dict(collections.Counter(zip(tokens, tokens[1:])))
  

  def merge(self, ids, pair, idx):
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
    
    self.vocab_size = vocab_size

    chunks = self.ptr.findall(text) #cs336 suggests using re.finditer but I dont want to read that shitty documentation again 
    chunks = [list(chunk.encode('utf-8')) for chunk in chunks if chunk not in self.special_tokens]

    stats = {}
    pair_to_chunks = {}
    chunk_pair_counts = []

    for i, chunk in enumerate(chunks):
      counts = self._pair_counts(chunk)
      chunk_pair_counts.append(counts)
      for pair, count in counts.items():
        stats[pair] = stats.get(pair, 0) + count
        pair_to_chunks.setdefault(pair, set()).add(i)
    
    num_merges = vocab_size - (256 + self.num_special)

    for i in range(num_merges):
      if not stats:
        break
      pair = max(stats, key = stats.get)
      idx = (256 + self.num_special) + i
      if verbose:
        print(f"merge {pair} into {idx}")

      affected = list(pair_to_chunks.get(pair, set()))
      if not affected:
        stats.pop(pair, None)
        continue

      for chunk_idx in affected:
        old_counts = chunk_pair_counts[chunk_idx]
        old_chunk = chunks[chunk_idx]
        new_chunk = self.merge(old_chunk, pair, idx)

        if new_chunk == old_chunk:
          continue

        for old_pair, old_count in old_counts.items():
          new_total = stats.get(old_pair, 0) - old_count
          if new_total <= 0:
            stats.pop(old_pair, None)
          else:
            stats[old_pair] = new_total

          if old_pair in pair_to_chunks:
            pair_to_chunks[old_pair].discard(chunk_idx)
            if not pair_to_chunks[old_pair]:
              pair_to_chunks.pop(old_pair)

        new_counts = self._pair_counts(new_chunk)
        for new_pair, new_count in new_counts.items():
          stats[new_pair] = stats.get(new_pair, 0) + new_count
          pair_to_chunks.setdefault(new_pair, set()).add(chunk_idx)

        chunks[chunk_idx] = new_chunk
        chunk_pair_counts[chunk_idx] = new_counts

      self.merges[pair] = idx
    
    self.vocab.update({idx: bytes([idx]) for idx in range(256)})
    for (p0, p1), idx in self.merges.items():
      self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    print('Finished training')


  def decode(self, ids):
    tokens = b"".join(self.vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
  

  def encode(self, text):
    chunks = self.ptr.findall(text)
    result = []
    for chunk in chunks:

      if chunk in self.special_tokens:
        idx = 256 + self.special_tokens.index(chunk)
        result.append(idx)
        continue

      tokens = list(chunk.encode('utf-8'))

      while len(tokens) >= 2:
        pairs = list(zip(tokens, tokens[1:]))
        pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))

        if pair not in self.merges:
          break

        idx = self.merges[pair]
        tokens = self.merge(tokens, pair, idx)
            
      result.extend(tokens)

    return result
  

  def save(self, filename="merges.model"):
    with open(filename, 'w', encoding="utf-8") as f:
      for (p0, p1), idx in self.merges.items():
        f.write(f"{p0} {p1} {idx}\n")
    print(f"saved to {filename}")
  
  def load(self, filename="merges.model"):
    with open(filename, 'r', encoding='utf-8') as f:
      for line in f:
        p0, p1, idx = map(int, line.split())
        self.merges[(p0, p1)] = idx

    for (p0, p1), idx in self.merges.items():
      self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
      
    self.vocab_size = len(self.vocab)
    print(f"Loaded vocab from {filename}")









