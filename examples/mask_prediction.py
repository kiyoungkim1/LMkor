# inspired by https://github.com/renatoviolin/next_word_prediction

import torch
import string
import transformers

transformers.logging.set_verbosity_error()

from transformers import BertTokenizerFast, BertForMaskedLM
bert_tokenizer = BertTokenizerFast.from_pretrained('kykim/bert-kor-base')
bert_model = BertForMaskedLM.from_pretrained('kykim/bert-kor-base').eval()

from transformers import AlbertForMaskedLM
albert_tokenizer = BertTokenizerFast.from_pretrained('kykim/albert-kor-base')
albert_model = AlbertForMaskedLM.from_pretrained('kykim/albert-kor-base').eval()

# from transformers import BartForConditionalGeneration
# roberta_tokenizer = BertTokenizerFast.from_pretrained('kykim/bart-kor-base')
# roberta_model = BartForConditionalGeneration.from_pretrained('kykim/bart-kor-basee').eval()

from transformers import BertTokenizerFast, BertForMaskedLM
bert_multilingual_tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
bert_multilingual_model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased').eval()

from transformers import XLMRobertaTokenizerFast, XLMRobertaForMaskedLM
xlmroberta_tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD][UNK]<pad><unk> '
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return ' / '.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True, mask_token='[MASK]', mask_token_id=4):
    # mask_token = tokenizer.mask_token
    # mask_token_id = tokenizer.mask_token_id

    text_sentence = text_sentence.replace('<mask>', mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def predict(text_sentence, top_k = 10, top_clean=3):
    if '<mask>' not in text_sentence:
        print('<mask> 를 입력해주세요. 예시: 이거 <mask> 재밌네? ')
        return

    # ========================= BERT =================================
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= ALBERT =================================
    input_ids, mask_idx = encode(albert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = albert_model(input_ids)[0]
    albert = decode(albert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= BART =================================
    # input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = bart_model(input_ids)[0]
    # bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= BERT MULTILINGUAL =================================
    input_ids, mask_idx = encode(bert_multilingual_tokenizer, text_sentence,
                                 mask_token = bert_multilingual_tokenizer.mask_token,
                                 mask_token_id = bert_multilingual_tokenizer.mask_token_id)
    with torch.no_grad():
        predict = bert_multilingual_model(input_ids)[0]
    bert_multilingual = decode(bert_multilingual_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= XLM ROBERTA BASE =================================
    input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence,
                                 mask_token=xlmroberta_tokenizer.mask_token,
                                 mask_token_id=xlmroberta_tokenizer.mask_token_id)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    results = {'kykim/bert-kor-base': bert,
            'kykim/albert-kor-base': albert,
            # 'bart': bart,
            'bert_multilingual': bert_multilingual,
            'xlm': xlm}

    for model, tokens in results.items():
        print('{0}: {1}'.format(model, tokens))
