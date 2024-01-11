# 신궁(Mistral base)

이 저장소에는 미스트랄 베이스 모델의 Lora 학습 코드를 담고 있습니다.

SFT Model: [https://huggingface.co/StatPan/singung-sft-v0.1](허깅페이스 모델 허브 SFT Lora 학습 모델 레포)
학습 이후 merge_and_unload 모델.

DPO Model: [https://huggingface.co/StatPan/singung-dpo-v0.1-2200](허깅페이스 모델 허브 SFT-DPO Lora 학습 모델 레포)
학습 이후 merge_and_unload 모델.

## requirements 설치

```
pip install -r requirements.txt
```

## Model History
```
Base Model: [https://huggingface.co/mistralai/Mistral-7B-v0.1](미스트랄 Pretrained 모델)

SFT_singung.py 스크립트를 활용한 학습 모델

SFT Model: [https://huggingface.co/StatPan/singung-sft-v0.1](허깅페이스 모델 허브 SFT Lora 학습 모델 레포)

DPO_singung.py 스크립트를 활용한 학습 모델

학습 중...

```


## References

[1] [Generating Long Sequences with Sparse Transformers, Child et al. 2019](https://arxiv.org/pdf/1904.10509.pdf)

[2] [Longformer: The Long-Document Transformer, Beltagy et al. 2020](https://arxiv.org/pdf/2004.05150v2.pdf)