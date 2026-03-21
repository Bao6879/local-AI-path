import torch

#Setup
with open('input.txt', 'r', encoding='utf-8') as f:
    text=f.read() #Everything in the file.
chars=list(set(text)) #All unique characters in the files.
vocabSize=len(chars)

cToI={c:i for i, c in enumerate(chars)} #Character -> index and vice versa
iToC={i:c for i, c in enumerate(chars)}
encode=lambda s: [cToI[c] for c in s] #Takes in a string -> turns to array of ints and vice versa

decode=lambda a: ''.join([iToC[i] for i in a]) 
data=torch.tensor(encode(text), dtype=torch.int) #Encoded dataset, stored in a tensor
trainData=data[:0.9*len(data)] #Split of 9:1 train:test
testData=data[0.9*len(data):]

#Settings
contextLength=8
