from memorizing_transformers_pytorch import MemorizingTransformer

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# constants (to handle data, training, and validation)
# 100 000
NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
SEQ_LEN = 512
SEGMENTS = 5

LEARNING_RATE = 2e-4
MAX_GRAD_CLIP_NORM = 0.5

VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model
model = MemorizingTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    memorizing_layers = 4,
    max_knn_memories = 512 * 15,
    num_retrieved_memories = 32,
    xl_memory_layers = (7, 8),
    xl_max_memories = 512,
)
# !
# ).cuda()

# prepare enwik8 data into train and val sets
with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    # data and length -> random sequence from data of sequence length
    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq
        # !
        # return full_seq.cuda()

    # number of sequences of seq_len that fit in data
    def __len__(self):
        return self.data.size(0) // self.seq_len

# dataset and dataloader
train_dataset = TextSamplerDataset(data_train, SEQ_LEN * SEGMENTS)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
valid_dataset = TextSamplerDataset(data_val, SEQ_LEN * SEGMENTS)
valid_loader = cycle(DataLoader(valid_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer
optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    model.train()
    # data size: (16x2561), different data in each batch
    data = next(train_loader)

    train_loss = 0.
    # init the KNN-memory
    with model.knn_memories_context(batch_size = BATCH_SIZE) as knn_memories:
        xl_memories = None    
        # seq: size 16x2560, label: size 16x2560, different seq and labels in each batch
        seq, labels = data[:, :-1], data[:, 1:]

        # chunk the sequence and labels into segmments -> iterate        
        for seq_segment, labels_segment in zip(seq.chunk(SEGMENTS, dim = -1), labels.chunk(SEGMENTS, dim = -1)):  
            # seq_segment: size 16x512, labels_segment: size 16x512
            # run data through model -> loss
            loss, xl_memories = model(
                seq_segment,
                labels = labels_segment,
                knn_memories = knn_memories,
                xl_memories = xl_memories
            )
            # update loss for this segment
            train_loss += loss.item() / SEGMENTS
            # backward pass
            (loss / SEGMENTS).backward()    

    print(f'training loss: {train_loss}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_CLIP_NORM)
    optim.step()
    optim.zero_grad()

    # validate every 100 batches
    if not (i % VALIDATE_EVERY):
        model.eval()
        valid_data = next(valid_loader)
        valid_loss = 0.

        with torch.no_grad(), model.knn_memories_context(batch_size = BATCH_SIZE) as knn_memories:
            xl_memories = None    
            seq, labels = valid_data[:, :-1], valid_data[:, 1:]
            
            # chunk val sequence and val labels into segments -> iterate
            for seq_segment, labels_segment in zip(seq.chunk(SEGMENTS, dim = -1), labels.chunk(SEGMENTS, dim = -1)):
                loss, xl_memories = model(
                    seq_segment,
                    labels = labels_segment,
                    knn_memories = knn_memories,
                    xl_memories = xl_memories
                )
                # update validation loss for this segment
                valid_loss += loss.item() / SEGMENTS

        print(f'valid loss: {valid_loss}')
