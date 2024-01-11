from datasets import Dataset, load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoModel,
                          AutoTokenizer,
                          AutoConfig)
import torch


def get_hf_dataset(dataset_repo:str,
                           split:str="train") -> Dataset:
    """load_hf_sample_dataset 허깅페이스 데이터셋 허브에서 데이터 로드.

    Args:
        dataset_repo (str): 허깅페이스 데이터셋 허브에서 데이터셋 레포명 또는 로컬 데이터셋 폴더 경로.
        split (str, optional): 데이터셋의 부분, [train, validation, test], 또는 train[:50] 형태로 조정가능. Defaults to "train".

    Returns:
        Dataset: datasets 라이브러리의 Dataset 객체 반환
    """
    ds = load_dataset(dataset_repo, split=split)
    return ds

def get_hf_model_tokenizer(model_repo:str,
                           base_model:bool=False,
                           use_flash:bool=True,
                           load_in_4bit:bool=False,) -> (AutoModelForCausalLM, AutoTokenizer):
    """get_hf_model_tokenizer 
    허깅페이스 포맷의 모델과 토크나이저를 불러옴.  

    Args:
        model_repo (str): 허깅페이스 모델 허브에 존재하는 레포명 또는 로컬 모델 폴더 경로.
        use_flash (bool): flash attention의 사용 여부를 표기.

    Returns:
        AutoModelForCausalLM, AutoTokenizer : LLM모델과 LLM 토크나이저 반환.
    """
    config = AutoConfig.from_pretrained(model_repo)
    if config.torch_dtype in [torch.float16,torch.float32]:
        dtype = torch.float16
    
    elif config.torch_dtype == torch.bfloat16:
        dtype = torch.bfloat16
    
    
    if base_model == True:
        model = AutoModel.from_pretrained(model_repo,
                                          torch_dtype="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_repo,
        device_map='auto',
        use_flash_attention_2=use_flash,
        torch_dtype=dtype,
        load_in_4bit=load_in_4bit)    
        
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    
    return model, tokenizer