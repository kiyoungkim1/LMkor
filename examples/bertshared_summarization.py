import transformers
from transformers import BertTokenizerFast, EncoderDecoderModel
transformers.logging.set_verbosity_error()

class Summarize():
    def __init__(self, model_name):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = EncoderDecoderModel.from_pretrained(model_name)

    def __call__(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors= 'pt')

        sentence_length = len(input_ids[0])
        min_length = max(10, int(0.1*sentence_length))
        max_length = min(128, int(0.3*sentence_length))


        outputs = self.model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length
        )

        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
