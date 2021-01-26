import transformers
from transformers import BertTokenizerFast, TFGPT2LMHeadModel, GPT2LMHeadModel
transformers.logging.set_verbosity_error()

class Inference():
    def __init__(self, model_name, tf_pt='tf'):
        self.tf_pt = tf_pt

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        if self.tf_pt == 'tf':
            self.model = TFGPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)

    def __call__(self, text, howmany=3):
        input_ids = self.tokenizer.encode(text, return_tensors='tf' if self.tf_pt=='tf' else 'pt')
        input_ids = input_ids[:, 1:]  # remove cls token

        outputs = self.model.generate(
            input_ids,
            min_length=30,
            max_length=50,
            do_sample=True,
            top_k=10,
            top_p=0.95,
            no_repeat_ngram_size=2,
            num_return_sequences=howmany
        )

        for idx, generated in enumerate(
                [self.tokenizer.decode(sentence, skip_special_tokens=True) for sentence in outputs]):
            print('{0}: {1}'.format(idx, generated))
