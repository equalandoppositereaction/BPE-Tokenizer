#idk what to do here
from nitok import Tokenizer
#import multiprocessing
filename = "TinyStoriesV2-GPT4-train.txt"
with open(filename, encoding="utf-8") as f:
    text = f.read()
tok = Tokenizer()
tok.train(text, 10000, True) #very shitty, want multiprocessing, dont know multi processing, me sad :( but still no AI caus fuck AI :)
tok.save("tiny_train.model")
#Validation

valtext = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typicallyapproached with supervised learning on taskspecific datasets."
valtest = tok.decode(tok.encode(valtext))
print(valtext == valtest)

encoded = tok.encode("One of them was a girl whom he often passed in the corridors")
for x in encoded:
  print(tok.decode([x]))