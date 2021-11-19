import torch
from torch import functional as F
from torch import nn

checkpoint = torch.load('./model_temp')
EMBEDDING_DIM = checkpoint['EMBEDDING_DIM']
CONTEXT_SIZE = checkpoint['CONTEXT_SIZE']
BATCH_SIZE = checkpoint['BATCH_SIZE']
vocab = checkpoint['vocab']
context = checkpoint['context']
word_to_ix = checkpoint['word_to_ix']
context = checkpoint['context']

# print('context', context)
# print('word_to_ix',word_to_ix)



class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, batch_size):
        super(CBOW, self).__init__()
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim * 2, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # print('inputs', inputs)
        # print('embed row', self.embeddings(inputs))
        embeds = self.embeddings(inputs).view((self.batch_size, -1))
        # print('embeds', embeds)
        out = F.relu(self.linear1(embeds))
        # print('linear1', out)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        # print('log_probs', log_probs)
        return log_probs

model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE)
model.load_state_dict(checkpoint['model_state_dict'])
print(model.eval())
# print(model.embeddings.weight)

# print(model.embeddings.weight[word_to_ix['외쳐요']])

embedding = []
for context_line in context:
    embedding_temp = []
    for word in context_line:
        embedding_temp.append(model.embeddings.weight[word_to_ix[word]].tolist())
    embedding.append(embedding_temp)

print(embedding[0])
torch.tensor(embedding[0])
#