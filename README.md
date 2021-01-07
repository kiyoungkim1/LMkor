### 공지 (2021.01.07)
* Google과 피드벡을 주고 받던 중, 대규모 한국어 자연어처리 모델을 개발한다는 취지에 장기간(21년 3월 중순쯤까지) **v3-8, v3-32 TPU를 제공받게** 되었습니다. 특히 선점형 v3-32 TPU 30개는 혼자 쓰기에는 너무 많고, 놀려두기에는 아까운 리소스라 이를 공유하고자 합니다. 언어, 이미지, 음성을 가리지 않고 **대규모 TPU연산이 필요하거나 궁금한게 있으신 분은 연락**주십시오.
* **한국어 텍스트 데이터를 요청드립니다.** 모델의 성능은 모델의 크기, 데이터의 양, 학습시간의 함수입니다. 저는 가장 큰 모델로 가장 오래 학습시키겠습니다. 여러분께서 자체 한국어 텍스트 데이터를 보유하고 계시다면 원기옥처럼 모아주십시오. 여러분들의 이름과 함께 최고의 언어모델을 학습시켜 보겠습니다.
* **대규모 언어 모델 학습과 TPU연산에 관심 있는 분들의 연락**을 기다립니다. 이제는 개인적으로 하는 프로젝트의 수준을 넘어갈 수도 있겠다고 생각되기 때문입니다. 경험이 있으신 분들이라면 부분부분적으로 도와주셔도 감사하겠습니다! 지금의 언어모델들은 '책으로 연애를 배운 사람'과 같다고 생각합니다. 책으로 배운 연애기술과 실제 연애가 다른건 분명하겠지요. 언어모델들도 단순히 텍스트를 넘어 실제경험인 이미지나 음성과 결합될것이라 생각합니다. 따라서 언어 외에도 이미지와 음성 연구자들과 협업도 언제나 환영입니다.


* 관련한 사항은 **kky416@gmail.com**로 연락주십시오.


# Pretrained Language Model For Korean

* 뉴스와 같이 잘 정제된 언어 뿐만 아니라, 실제 인터넷 상에서 쓰이는 신조어, 줄임말, 오자, 탈자를 잘 이해할 수 있는 모델을 개발하였습니다. 이를 위해 대분류 주제별 텍스트를 별도로 수집하였으며 대부분의 데이터는 블로그, 댓글, 리뷰입니다.
* Vocab에 여유분을 많이 둬서 커머스 대분류별로 도메인 특화 언어모델을 개발 할 수 있습니다.
* 도메인 특화 언어모델 등의 모델에 관한 문의나, 상업적 사용에 대해서는 kky416@gmail.com로 문의 부탁드립니다.


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
| albert-kor-base   |              768 |         12 |        256 |       1024 |          5e-4 |           0.9M |
| bert-kor-base     |              768 |         12 |        512 |        256 |          1e-4 |           1.9M |
| electra-kor-base  |              768 |         12 |        512 |        256 |          2e-4 |           1.9M |

* Electra 모델은 discriminator입니다.
* Bert에는 whole-word-masking이 적용되었습니다.

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
| **electra-kor-base**  |     **91.29**      |         87.20          |     **85.50**      |      **83.11**       |           85.46           |          **95.78**          |                  66.03                 |


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
