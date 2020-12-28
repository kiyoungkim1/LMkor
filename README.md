# Pretrained Language Model For Korean
* Pre-release상태며 2021년 초 더 많은 모델을 공개 할 예정입니다
<br/><br/><br/>
* 뉴스와 같이 잘 정제된 언어 뿐만 아니라, 실제 인터넷 상에서 쓰이는 신조어, 줄임말, 오자, 탈자를 잘 이해할 수 있는 모델을 개발하였습니다. 이를 위해 대분류 주제별 텍스트를 별도로 수집하였으며 대부분의 데이터는 블로그, 댓글, 리뷰입니다.
* Vocab에 여유분을 많이 둬서 커머스 대분류별로 도메인 특화 언어모델을 개발 할 수 있습니다.
* 도메인 특화 언어모델 등의 모델에 관한 문의나, 상업적 사용에 대해서는 kky416@gmail.com로 문의 부탁드립니다.
<br/><br/><br/>
* 본 모델을 활용한 인공지능 서비스를 만들다 모델을 공개하기로 결정 하였습니다. 기술에 관심있는 분께는 다양한 모델로 연구 기회를 늘려드리고, 사업화에 관심있는 분께는 기술보다는 비즈니스모델에 집중하게 해드리고 싶어서입니다.
* 인공지능 사업의 경쟁력은 기술이 아니라 비즈니스모델에 있다고 생각합니다. 따라서 텍스트, 이미지, 음성 등의 인공지능을 활용해 다양한 비즈니스모델을 단기간 테스트하고 평가하며 좋은 서비스를 찾아가는 조직을 구성하고 있는 중입니다. 
* 지금도 두 개의 프로젝트를 진행중이며, 함께 무언가를 가길 원하시는 기획자, 개발자, 디자이너는 kky416@gmail.com로 연락 주십시오. 인공지능 전문가가 아니어도 괜찮습니다. 최고의 기술이 아닌 최고의 서비스를 만들고자 하기 때문입니다.


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
| Vocab Len | lower_case | strip_accent |
| --------: | ------------: | ------------: |
|     42000 |         True |         False |


* 한글, 영어, 숫자와 일부 특수문자를 제외한 문자는 학습에 방해가된다고 판단하여 삭제하였습니다(예시: 한자, 이모지 등)
* [Huggingface tokenizers](https://github.com/huggingface/tokenizers) 의 wordpiece모델을  사용해 40000개의 subword를 생성하였습니다.   
* 여기에 2000개의 unused token과 넣어 학습하였으며, unused token는 도메인 별 특화 용어를 담기 위해 사용됩니다.

## Pretraining models

|                   | Hidden size      | layers     |max length  | batch size | learning rate | training steps |
| ----------------- |----------------: | ---------: | ---------: | ---------: | ------------: | -------------: |
| albert-kor-base   |              768 |         12 |        256 |       1024 |          5e-4 |          0.25M |
| bert-kor-base     |              768 |         12 |        512 |        256 |          5e-5 |             1M |
| electra-kor-base  |              768 |         12 |        512 |        256 |          2e-4 |             1M |

* Electra 모델은 discriminator입니다.
* 모델 구조나 hyper-parameter는 논문과 차이가 날 수 있으며, 학습은 TPU(v2, v3), GPU등 다양한 환경에서 수행되었습니다.

## Fine-tuning
* Fine-tuning 코드와 KoBert, XLM-Roberta-Base, HanBERT, KoELECTRA-Base-v3 결과는 [KoELECTRA](https://github.com/monologg/KoELECTRA) 를 참고하였습니다. 이 외에는 직접 fine-tuning을 수행하였으며 batch size=32, learning rate=3e-5, epoch=5~15를 사용하였습니다.
* 각 모델에는 동일한 코드가 적용되었지만, 다양한 환경에서 fine-tuning이 수행되어 환경에 따라 결과값이 약간씩 차이가 날 수 있습니다.

|                       | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) |  **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :-------------------- | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :-----------------------------------:  |
| KoBERT                |       89.59        |         87.92          |       81.25        |        79.62         |           81.59           |            94.85            |                  66.21                 |
| XLM-Roberta-Base      |       89.03        |         86.65          |       82.80        |        80.23         |           78.45           |            93.80            |                  64.06                 |
| HanBERT               |       90.06        |         87.70          |       82.95        |        80.32         |           82.73           |            94.72            |                  68.32                 |
| kcbert-base           |       89.87        |         84.46          |       66.70        |        75.57         |           75.86           |            94.46            |                  68.13                 |
| KoELECTRA-Base-v3     |       90.63        |       **88.11**        |       84.45        |      **82.24**       |         **85.53**         |            95.25            |                  67.61                 |
|**OURS**|
| **albert-kor-base**   |       88.85        |         81.66          |       78.60        |        78.02         |           80.13           |            93.67            |                  65.48                 |
| **bert-kor-base**     |       90.83        |         86.54          |       82.95        |        81.53         |           82.27           |            95.12            |                **68.87**               |
| **electra-kor-base**  |     **91.29**      |         87.14          |     **84.90**      |        82.03         |           84.89           |          **95.38**          |                  66.84                 |


## Comment
* **데이터셋의 특성을 잘 이해**하여야 합니다. nsmc에는 training set에는 '너무재밓었다그래서보는것을추천한다'가 부정으로 라벨링되어 있습니다. 만약 해당 리뷰가 test set에 있었다면 긍정으로 판별될 확률이 높고 accuracy는 떨어지게 됩니다.  좋은 모델이 점수가 잘 나오는 경향성은 분명히 있지만 accuracy가 100인 모델이 최고의 모델이 아닐수도 있습니다. 또한 **메트릭 스코어는 데이터셋의 일관성(consistency)에 큰 영향**을 받으며습니다. 따라서 앞으로는 데이터셋 자체를 평가할 수 있는 메트릭도 생겨야 할것입니다.

* 비슷한 말이지만 데이터셋에는 '**가치판단**'이 들어가 있습니다. '배우 얼굴만 보여요'가 누군가에게는 긍정일수도 있고 부정일 수도 있습니다. **너무 쉬운 데이터셋은 인공지능 활용가치가 떨어지며, 너무 어려운 데이터셋은 애매모호함이 많습니다**. 따라서 데이터셋을 공개하거나 사용할 때는 이를 분명히 숙지해야 합니다.

* [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)에 따르면 **cross-entropy loss 퍼포먼스는 모델의 크기, 데이터의 양, 컴퓨팅 시간의 로그함수이며 모델 아키텍쳐의 영향는 적다는 것**입니다. 모델 크기가 가장 중요하며, 그 다음 배치 사이즈, 컴퓨팅 시간이라고 합니다. 

## Citation

```
@misc{kim2020lmkor,
  author = {Kiyoung Kim},
  title = {Pretrained Language Models For Korean},
  year = {2020},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/kiyoungkim1/LM-kor}}
}
```

## Reference
* [BERT](https://github.com/google-research/bert)
* [ALBERT](https://github.com/google-research/albert)
* [ELECTRA](https://github.com/google-research/electra)
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
