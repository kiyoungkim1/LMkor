# Pretrained Language Model For Korean

* 최고의 성능을 내는 언어개발들이 세계 각지에서 개발되고 있지만 대부분 영어만을 다루고 있습니다. 한국어 자연어 처리 연구를 시작하시는데 도움이 되고자 한국어로 학습된 최신 언어모델들을 공개합니다. 
* Transformers 라이브러리를 통해 사용가능하도록 만들었으며 encoder 기반(BERT 등), decoder 기반(GPT3), encoder-decoder(T5, BERTSHARED) 모델을 모두 제공하고 있습니다.
* 뉴스와 같이 잘 정제된 언어 뿐만 아니라, 실제 인터넷 상에서 쓰이는 신조어, 줄임말, 오자, 탈자를 잘 이해할 수 있는 모델을 개발하기 위해, 대분류 주제별 텍스트를 별도로 수집하였으며 대부분의 데이터는 블로그, 댓글, 리뷰입니다.
* 더 높은 정확도의 모델이나 도메인 특화 언어모델 및  모델의 상업적 사용에 대해서는 kky416@gmail.com로 문의 부탁드립니다.

## Recent update
* 2021-01-30: [Bertshared](https://arxiv.org/abs/1907.12461) (Bert를 기반으로 한 seq2seq모델) 모델 추가
* 2021-01-26: [GPT3](https://github.com/openai/gpt-3) 모델 초기 버전 추가
* 2021-01-22: [Funnel-transformer](https://github.com/laiguokun/Funnel-Transformer) 모델 추가

## Pretraining models

|                                     | Hidden size      | layers     |max length  | batch size | learning rate | training steps |                |
| --------------------------------    |----------------: | ---------: | ---------: | ---------: | ------------: | -------------: |--------------: |
| albert-kor-base                     |              768 |         12 |        256 |       1024 |          5e-4 |           0.9M |                |
| bert-kor-base                       |              768 |         12 |        512 |        256 |          1e-4 |           1.9M |                |
| funnel-kor-base                     |              768 |      6_6_6 |        512 |        128 |          8e-5 |           0.9M |                |
| electra-kor-base                    |              768 |         12 |        512 |        256 |          2e-4 |           1.9M |                |
| gpt3-kor-small_based_on_gpt2        |              768 |         12 |       2048 |       4096 |          1e-2 |           4.5K | will be update |
| bertshared-kor-base                 |          768/768 |      12/12 |     512/512 |        16 |          5e-5 |            20K |                |

* 원본 모델과 달리 tokenizer는 모든 모델에 대해 wordpiece로 통일하였습니다.
* ELECTRA 모델은 discriminator입니다.
* BERT 모델에는 whole-word-masking이 적용되었습니다.
* FUNNEL-TRANSFORMER 모델은 ELECTRA모델을 사용했고 generator와 discriminator가 모두 들어가 있습니다.
* GPT3의 경우 정확한 아키텍쳐를 공개하진 않았지만 GPT2와 거의 유사하며 few-shot 학습을 위해 input길이를 늘리고 계산 효율화를 위한 몇가지 처리를 한 것으로 보입니다. 따라서 GPT2를 기반으로 이를 반영하여 학습하였습니다.
* Bertshared는 transformer seq2seq모델로 encoder와 decoder를 bert-kor-base로 초기값을 준 다음 training을 한 것입니다. encoder와 decoder가 파라미터를 공유하게 함으로써 하나의 bert 모델 용량으로 seq2seq를 구현할 수 있게 되었습니다 ([reference](https://arxiv.org/abs/1907.12461)). 공개한 모델은 summarization 태스크에 대해 학습한 것입니다.

## Notebooks
|    |  설명  | Colab  |
| ---| ------| ----- |
| GPT3 generation            |   GPT3 모델을 통해 한글 텍스트를 입력하면 문장의 뒷부분을 생성합니다.      |         [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kiyoungkim1/LMkor/blob/main/notebooks/gpt3_text_generation.ipynb) |
| Bertshared summarization   |   Bertshared모델을 통해 문서를 요약합니다.                          |         [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kiyoungkim1/LMkor/blob/main/notebooks/summarization_with_bertshared.ipynb) |

* 간단한 테스트 결과와 사용법을 보여드리기 위한 것으로, 자체 데이터로 원하시는 성능을 얻기 위해서는 tuning이 필요합니다.

## Usage
* [Transformers](https://github.com/huggingface/transformers) 라이브러리를 통해 pytorch와 tensorflow 모두에서 편하게 사용하실 수 있습니다.

```python
# electra-base-kor
from transformers import ElectraTokenizerFast, ElectraModel, TFElectraModel
tokenizer_electra = ElectraTokenizerFast.from_pretrained("kykim/electra-kor-base")

model_electra_pt = ElectraModel.from_pretrained("kykim/electra-kor-base")    # pytorch
model_electra_tf = TFElectraModel.from_pretrained("kykim/electra-kor-base")  # tensorflow

# bert-base-kor
from transformers import BertTokenizerFast, BertModel
tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
model_bert = BertModel.from_pretrained("kykim/bert-kor-base")

# albert-base-kor
from transformers import BertTokenizerFast, AlbertModel
tokenizer_albert = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")
model_albert = AlbertModel.from_pretrained("kykim/albert-kor-base")

# funnel-base-kor
from transformers import FunnelTokenizerFast, FunnelModel
tokenizer_funnel = FunnelTokenizerFast.from_pretrained("kykim/funnel-kor-base")
model_funnel = FunnelModel.from_pretrained("kykim/funnel-kor-base")

# gpt3-kor-small_based_on_gpt2
from transformers import BertTokenizerFast, GPT2LMHeadModel
tokenizer_gpt3 = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
model_gpt3 = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")

# bertshared-kor-base (only for pytorch in transformers)
from transformers import BertTokenizerFast, EncoderDecoderModel
tokenizer_bertshared = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")
model_bertshared = EncoderDecoderModel.from_pretrained("kykim/bertshared-kor-base")
```

## Dataset

* 학습에 사용한 데이터는 다음과 같습니다.
  
```
  - 국내 주요 커머스 리뷰 1억개 + 블로그 형 웹사이트 2000만개 (75GB)
  - 모두의 말뭉치 (18GB)
  - 위키피디아와 나무위키 (6GB)
```

* 불필요하거나 너무 짤은 문장, 중복되는 문장들을 제외하여 100GB의 데이터 중 최종적으로 70GB (약 127억개의 token)의 텍스트 데이터를 학습에 사용하였습니다.   
* 데이터는 화장품(8GB), 식품(6GB), 전자제품(13GB), 반려동물(2GB) 등등의 카테고리로 분류되어 있으며 도메인 특화 언어모델 학습에 사용하였습니다.

## Vocab
| Vocab Len | lower_case    | strip_accent  |
| --------: | ------------: | ------------: |
|     42000 |         True  |         False |


* 한글, 영어, 숫자와 일부 특수문자를 제외한 문자는 학습에 방해가된다고 판단하여 삭제하였습니다(예시: 한자, 이모지 등)
* [Huggingface tokenizers](https://github.com/huggingface/tokenizers) 의 wordpiece모델을  사용해 40000개의 subword를 생성하였습니다.   
* 여기에 2000개의 unused token과 넣어 학습하였으며, unused token는 도메인 별 특화 용어를 담기 위해 사용됩니다.

## Fine-tuning
* Fine-tuning 코드와 KoBert, HanBERT, KoELECTRA-Base-v3 결과는 [KoELECTRA](https://github.com/monologg/KoELECTRA) 를 참고하였습니다. 이 외에는 직접 fine-tuning을 수행하였으며 batch size=32, learning rate=3e-5, epoch=5~15를 사용하였습니다.

|                       | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) |  **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :-------------------- | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :-----------------------------------:  |
| KoBERT                |       89.59        |         87.92          |       81.25        |        79.62         |           81.59           |            94.85            |                  66.21                 |
| HanBERT               |       90.06        |         87.70          |       82.95        |        80.32         |           82.73           |            94.72            |                  68.32                 |
| kcbert-base           |       89.87        |         85.00          |       67.40        |        75.57         |           75.94           |            93.93            |                **68.78**               |
| KoELECTRA-Base-v3     |       90.63        |       **88.11**        |       84.45        |        82.24         |         **85.53**         |            95.25            |                  67.61                 |
|**OURS**|
| **albert-kor-base**   |       89.45        |         82.66          |       81.20        |        79.42         |           81.76           |            94.59            |                  65.44                 |
| **bert-kor-base**     |       90.87        |         87.27          |       82.80        |        82.32         |           84.31           |            95.25            |                  68.45                 |
| **electra-kor-base**  |       91.29        |         87.20          |     **85.50**      |      **83.11**       |           85.46           |          **95.78**          |                  66.03                 |
| **funnel-kor-base**   |     **91.36**      |         88.02          |       83.90        |                      |           84.52           |            95.51            |                  68.18                 |

## Comment
* **데이터셋의 특성을 잘 이해**하여야 합니다. nsmc에는 training set에는 '너무재밓었다그래서보는것을추천한다'가 부정으로 라벨링되어 있습니다. 만약 해당 리뷰가 test set에 있었다면 긍정으로 판별될 확률이 높고 accuracy는 떨어지게 됩니다.  좋은 모델이 점수가 잘 나오는 경향성은 분명히 있지만 accuracy가 100인 모델이 최고의 모델이 아닐수도 있습니다. 또한 **메트릭 스코어는 데이터셋의 일관성(consistency)에 큰 영향**을 받으며습니다. 따라서 앞으로는 데이터셋 자체를 평가할 수 있는 메트릭도 중요해 지리라 생각합니다.

* 비슷한 말이지만 데이터셋에는 '**가치판단**'이 들어가 있습니다. '배우 얼굴만 보여요'가 누군가에게는 긍정일수도 있고 부정일 수도 있습니다. **너무 쉬운 데이터셋은 인공지능 활용가치가 떨어지며, 너무 어려운 데이터셋은 애매모호함이 많습니다**. 따라서 데이터셋을 공개하거나 사용할 때는 이를 분명히 숙지해야 합니다.

* [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)에 따르면 **cross-entropy loss 퍼포먼스는 모델의 크기, 데이터의 양, 컴퓨팅 시간의 로그함수이며 모델 아키텍쳐의 영향는 적다는 것**입니다. 모델 크기가 가장 중요하며, 그 다음 배치 사이즈, 컴퓨팅 시간이라고 합니다. 

## Citation

```
@misc{kim2020lmkor,
  author = {Kiyoung Kim},
  title = {Pretrained Language Models For Korean},
  year = {2020},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/kiyoungkim1/LMkor}}
}
```

## Reference
* [BERT](https://github.com/google-research/bert)
* [ALBERT](https://github.com/google-research/albert)
* [ELECTRA](https://github.com/google-research/electra)
* [FUNNEL-TRANSFORMER](https://github.com/laiguokun/Funnel-Transformer)
* [GPT3](https://github.com/openai/gpt-3) and [GPT2](https://github.com/openai/gpt-2)
* [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
* [Huggingface Transformers](https://github.com/huggingface/transformers)
* [Huggingface Tokenizers](https://github.com/huggingface/tokenizers)
* [모두의 말뭉치](https://corpus.korean.go.kr/)
* [KoELECTRA](https://github.com/monologg/KoELECTRA)
* [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

## Acknowledgments

* Cloud TPUs are provided by [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc/) program.
  
* Also, [모두의 말뭉치](https://corpus.korean.go.kr/) is used for pretraining data.

## License

* The pretrained models is distributed under the terms of the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
