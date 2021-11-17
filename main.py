import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import MeCab

CONTEXT_SIZE = 2  # 2 words to the left, 2 to    the right
# Embedding Dimension
EMBEDDING_DIM = 50
BATCH_SIZE = 1000
nb_epochs = 1
cuda = torch.device('cuda:0')

# Dataset 상속
class CustomDataset(Dataset):
  #def __init__(self, dir, file_name):
  def __init__(self, CONTEXT_SIZE):
      self.data = pd.read_csv('./발라드.csv')

      wakati = MeCab.Tagger("-Owakati")

      self.data = self.data[['lyrics', '이별 유무']]

      self.data['lyrics'] = [wakati.parse(data) for data in self.data['lyrics']]

      self.data['lyrics'] = self.data['lyrics'].apply(str.split)

      self.x_data = self.data['lyrics']
      self.y_data = self.data['이별 유무']

      self.raw_text = [word
                       for sentence in self.data['lyrics'] for word in sentence]
      # print(self.raw_text)
      self.vocab = set(self.raw_text)
      self.vocab_size = len(self.vocab)

      self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
      self.word = [word_ for word_ in self.word_to_ix]

      li_temp = self.data['lyrics']

      self.data = []
      for temp in li_temp:
          for i in range(CONTEXT_SIZE, len(temp) - CONTEXT_SIZE):
              context = (
                      [temp[i - j - 1] for j in range(CONTEXT_SIZE)]
                      + [temp[i + j + 1] for j in range(CONTEXT_SIZE)]
              )
              target = temp[i]
              self.data.append([context, target])

      self.x_ = []
      self.y_ = []
      for temp in li_temp:
          for i in range(CONTEXT_SIZE, len(temp) - CONTEXT_SIZE):
              context = (
                      [self.word_to_ix[temp[i - j - 1]] for j in range(CONTEXT_SIZE)]
                      + [self.word_to_ix[temp[i + j + 1]] for j in range(CONTEXT_SIZE)]
              )
              target = self.word_to_ix[temp[i]]
              self.x_.append(context)
              self.y_.append(target)

      # print(self.data_[0])

  # 총 데이터의 개수를 리턴
  def __len__(self):
    return len(self.x_)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    # x = self.data[idx]
    # print('x',x)
    x = torch.tensor(self.x_[idx], dtype=torch.long, device=cuda)
    y = torch.tensor(self.y_[idx], dtype=torch.long, device=cuda)
    return x, y

dataset = CustomDataset(CONTEXT_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w[0]] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


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

losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(dataset.vocab), EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE)
model.cuda(device=cuda)
optimizer = optim.SGD(model.parameters(), lr=0.001)


for epoch in range(1, nb_epochs+1):
    # print('epoch',epoch, losses)
    total_loss = 0
    print('epoch', epoch)
    for idx, (x,y) in enumerate(dataloader):
        if len(dataloader) == idx+1:
            print('current total loss', total_loss)
            break
        if idx%100 == 0:
            print(idx, end=' ')
        # print(x)
        # print(y)
        # break
        model.zero_grad()
        log_probs = model(x)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        # print('log_probs', log_probs)
        # print('y', y)
        loss = loss_function(log_probs, y)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print('losses',losses)  # The loss decreased every iteration over the training data!

# To get the embedding of a particular word, e.g. "beauty"
# print(model.embeddings.weight[dataset.raw_text["안녕"]])

print('total embeddings',
      model.embeddings.weight)

torch.save(model.state_dict(), './model')
model = CBOW(len(dataset.vocab), EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE)
model.load_state_dict(torch.load('./model'))
print(model.eval())
print(model.embeddings.weight)
