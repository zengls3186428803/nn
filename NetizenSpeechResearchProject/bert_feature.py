import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from tools.resouces_monitor import show_mem

tokenizer = BertTokenizer.from_pretrained('./chinese_wwm_ext', do_lower_case=True)
bert = BertModel.from_pretrained('./chinese_wwm_ext')
bert.eval()


def get_feature_vector(texts, max_len):
    result = None
    flag = False
    for text in texts:
        sentence = text
        tokens = tokenizer.tokenize(str(sentence))
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        differ = max_len - len(tokens)
        if max_len > len(tokens):
            tokens = tokens + ['[PAD]'] * differ
        else:
            tokens = tokens[:max_len - 1] + ['[PAD]']
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if t != '[PAD]' else 0 for t in tokens]

        token_ids = torch.tensor(token_ids)
        token_ids = torch.reshape(token_ids, [1, -1])

        attention_mask = torch.tensor(attention_mask)
        attention_mask = torch.reshape(attention_mask, [1, -1])

        output = bert(token_ids, attention_mask=attention_mask, output_all_encoded_layers=False)

        if not flag:
            flag = True
            result = output[0]

        else:
            result = torch.cat((result, output[0]))
    # show_mem()
    return result
