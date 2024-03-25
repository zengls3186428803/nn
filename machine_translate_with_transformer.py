import time
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
import torch
from my_transformer.myTransformer import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer

torch.manual_seed(131)

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_BLOCK_LAYERS = 6
DEVICE = "cpu"
device = DEVICE
NUM_EPOCHS = 18

# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL[
    "train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL[
    "valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE, NUM_BLOCK_LAYERS, NHEAD)
model = model.to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)  # 两个beta是Adam里的移动平均的系数。
# 初始化参数
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               vocab_transform[ln],  # Numericalization
                                               tensor_transform)  # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    src_batch = src_batch.transpose(0, 1)
    tgt_batch = tgt_batch.transpose(0, 1)
    return src_batch, tgt_batch


def generate_square_subsequent_mask(sz):
    return torch.Tensor(torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1)


def create_mask(src, tgt, num_head):
    batch_size = src.shape[0]
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)

    mem_mask = torch.Tensor(src_padding_mask).contiguous().view(batch_size, 1, src_seq_len).expand(-1,
                                                                                                   tgt_seq_len,
                                                                                                   -1)

    src_mask = src_mask.contiguous().view(1, src_seq_len, src_seq_len).expand(batch_size, -1, -1)
    src_padding_mask = torch.Tensor(src_padding_mask).contiguous().view(batch_size, 1, src_seq_len).expand(-1,
                                                                                                           src_seq_len,
                                                                                                           -1)
    tgt_mask = tgt_mask.contiguous().view(1, tgt_seq_len, tgt_seq_len).expand(batch_size, -1, -1)
    tgt_padding_mask = torch.Tensor(tgt_padding_mask).contiguous().view(batch_size, 1, tgt_seq_len).expand(-1,
                                                                                                           tgt_seq_len,
                                                                                                           -1)
    final_src_mask = src_mask + src_padding_mask
    final_tgt_mask = tgt_mask + tgt_padding_mask
    final_mem_mask = mem_mask

    final_src_mask = final_src_mask.view(batch_size, 1, src_seq_len, src_seq_len).expand(-1, num_head, -1, -1)
    final_tgt_mask = final_tgt_mask.view(batch_size, 1, tgt_seq_len, tgt_seq_len).expand(-1, num_head, -1, -1)
    final_mem_mask = final_mem_mask.view(batch_size, 1, tgt_seq_len, src_seq_len).expand(-1, num_head, -1, -1)

    # print("final_src_mask.size=", final_src_mask.size())
    # print("final_tgt_mask.size=", final_tgt_mask.size())
    # print("final_mem_mask.size=", final_mem_mask.size())
    # print(final_mem_mask)

    return final_src_mask, final_mem_mask, final_tgt_mask


def train_epoch(model, optimizer):
    model: Transformer
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        src_mask, mem_mask, tgt_mask = create_mask(src, tgt_input, NHEAD)
        logits = model(src, tgt_input, src_mask, mem_mask, tgt_mask)
        # print("logits.size = ", logits.size())
        # print(logits)
        optimizer.zero_grad()
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        src_mask, mem_mask, tgt_mask = create_mask(src, tgt_input, NHEAD)
        logits = model(src, tgt_input, src_mask, mem_mask, tgt_mask)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(list(val_dataloader))


for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    val_loss = evaluate(model)
    print((
        f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask, _, _ = create_mask(src, src, NHEAD)
    src_mask = src_mask.to(DEVICE)
    tgt = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    print(tgt)
    mem = model.encode(src, src_mask)

    for i in range(max_len - 1):
        mem = mem.to(DEVICE)
        _, mem_mask, tgt_mask = create_mask(src, tgt, NHEAD)
        out = model.decode(tgt=tgt, mem=mem, mem_mask=mem_mask, tgt_mask=tgt_mask)
        # print("out.size()=", out.size())
        # out.size=(batch_size, seq_L, embedding_size) = (1, seq_L, embedding_size)

        # print("out[:, -1].size()=", out[:, -1].size())
        prob = model.out(out[:, -1])  # 查看最后一个词的下一个词。

        _, next_word = torch.max(prob, dim=1)
        # print("next_word.size()=", next_word.size())
        next_word = next_word.item()
        tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return tgt


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    tgt_tokens = greedy_decode(model, src, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                                                                         "").replace(
        "<eos>", "")


print(translate(model, "Eine Gruppe von Menschen steht vor einem Iglu ."))

