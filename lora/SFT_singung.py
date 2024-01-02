from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer,
                          BitsAndBytesConfig, 
                          TrainingArguments)
from trl import (DataCollatorForCompletionOnlyLM, 
                 SFTTrainer)
from transformers import (MistralForCausalLM, 
                          LlamaTokenizerFast)
from datasets import (Dataset, 
                      load_dataset)

from peft import LoraConfig
import torch

import yaml
import fire

SYSTEM_PROMPT = "### System:"
USER_PROMPT = " ### User: "
ASSISTANT_PROMPT = " ### Assistant: "

def get_lora_model(mistral_pretrained_id:str,use_flash:bool,) -> MistralForCausalLM:
    """
    로라 학습 전 베이스 모델을 불러오는 함수.

    Args:
        mistral_pretrained_id (str, optional): 허깅페이스 허브에 존재하는 repo 또는 로컬 폴더 경로. Defaults to "mistralai/Mistral-7B-v0.1".
        use_flash (bool, optional): Flash attention2 사용 여부. https://arxiv.org/abs/2307.08691 참조. Defaults to False.

    Returns:
        MistralForCausalLM
    """
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    )
    model = AutoModelForCausalLM.from_pretrained(mistral_pretrained_id,
                                                quantization_config=bnb_config,
                                                use_flash_attention_2=use_flash,
                                                device_map="cuda:1",
                                                )
    assert type(model).__name__ == 'MistralForCausalLM', "MistralForCausalLM 베이스 모델 학습용 입니다."
    
    return model
    
def get_tokenizer(mistral_pretrained_id:str,
                  add_bos_token:bool,
                  add_eos_token:bool,) -> LlamaTokenizerFast:
    """
    학습 전 토크나이저 로드 및 세팅

    Args:
        mistral_pretrained_path (str, optional): 허깅페이스 허브에 존재하는 repo 또는 로컬 폴더 경로.. Defaults to "mistralai/Mistral-7B-v0.1".
        add_bos_token (bool, optional): 토크나이저의 학습 전 세팅. Defaults to True.
        add_eos_token (bool, optional): 토크나이저의 학습 전 세팅. Defaults to True.

    Returns:
        LlamaTokenizerFast: 미스트랄 모델의 토크나이저 
    """
    
    tokenizer = AutoTokenizer.from_pretrained(mistral_pretrained_id)   
    tokenizer.add_bos_token = add_bos_token
    tokenizer.add_eos_token = add_eos_token
    tokenizer.pad_token = tokenizer.unk_token # pad token과 eos token이 같을 때, 특정 Datacollator와 문제 생김을 방지하기 위해 unk token 사용
    tokenizer.padding_side = "left" 
    
    assert type(tokenizer).__name__ == 'LlamaTokenizerFast', ""
    
    return tokenizer


def get_dataset(dataset_path:str="jhflow/orca_ko_en_pair",
                dataset_split:str|None="train",
                system_prompt_column:str|None="system_prompt_ko",
                user_prompt_column:str="question_ko",
                assistant_prompt_column:str="output_ko",):
    """
    학습을 위해 데이터셋을 불러오는 함수.

    Args:
        dataset_path (str, optional): 허깅페이스 허브에 있는 dataset repo 주소, 또는 폴더 경로. 
                                      
        dataset_split (str | None, optional): 데이터셋으로 로드하는 경우, train,validation,test로 
                                              나눠지는 경우가 존재함. 
                                              train으로 지정할 경우 train set만 불러옴. 
                                              
        system_prompt_column (str, optional): 시스템 프롬프트 컬럼명. 사용하지 않을 경우 None 입력. 
                                              
        user_prompt_column (str, optional): 유저의 지시, 질의 컬럼명. 
        assistant_prompt_column (str, optional): 인공지능의 답변, 정답 컬럼명. 
                                                 

    Returns:
        Dataset: Huggingface datasets의 Dataset 반환
    """
    
    ds = load_dataset(dataset_path, split=dataset_split)
    df = ds.to_pandas()
    
    if system_prompt_column != None:
        df["template"] = (SYSTEM_PROMPT + "\n" + df[system_prompt_column] + "\n" + 
                          USER_PROMPT + "\n" + df[user_prompt_column] + "\n" +
                          ASSISTANT_PROMPT + "\n" + df[assistant_prompt_column])
        ds = Dataset.from_pandas(df[["template"]])
        
    elif system_prompt_column == None:
        df["template"] = (USER_PROMPT + "\n" + df[user_prompt_column] + "\n" +
                          ASSISTANT_PROMPT + "\n" + df[assistant_prompt_column])
        ds = Dataset.from_pandas(df[["template"]])

    return ds


def get_data_collator(tokenizer:LlamaTokenizerFast) -> DataCollatorForCompletionOnlyLM:
    """
    SFTTrainer와 동작하는 data collator 세팅

    Args:
        tokenizer (LlamaTokenizerFast): data collator와 함께 동작할 토크나이저 객체

    Returns:
        DataCollatorForCompletionOnlyLM: SFTTrainer 객체와 동작하는 data collator 반환
    """
    instruction_template = USER_PROMPT.strip()
    response_template = ASSISTANT_PROMPT.strip()
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, 
                                           response_template=response_template,
                                           tokenizer=tokenizer)
    return collator


def main(mistral_pretrained_id:str="/root/work/pretrained_models/Mistral-7B-v0.1",
         use_flash:bool=False,
         add_bos_token:bool=True,
         add_eos_token:bool=True,
         sft_dataset_path:str="jhflow/orca_ko_en_pair",
         dataset_split:str|None="train",
         system_prompt_column:str|None="system_prompt_ko",
         user_prompt_column:str="question_ko",
         assistant_prompt_column:str="output_ko",
         max_seq_length:int=4096,
         dataset_num_proc=1,) -> None:
    
    model = get_lora_model(mistral_pretrained_id=mistral_pretrained_id,
                           use_flash=use_flash)
    
    tokenizer = get_tokenizer(mistral_pretrained_id=mistral_pretrained_id,
                              add_bos_token=add_bos_token,
                              add_eos_token=add_eos_token)
    
    data_collator = get_data_collator(tokenizer=tokenizer)
    
    with open("./lora_config.yaml", 'r') as f:
        lora_config_dict = yaml.safe_load(f)
        peft_config = LoraConfig(**lora_config_dict)
    
    with open("./args.yaml", 'r') as f:
        args_dict = yaml.safe_load(f)
        args = TrainingArguments(**args_dict)
    
    ds = get_dataset(dataset_path=sft_dataset_path,
                     dataset_split=dataset_split,
                     system_prompt_column=system_prompt_column,
                     user_prompt_column=user_prompt_column,
                     assistant_prompt_column=assistant_prompt_column,)
    
    trainer = SFTTrainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=ds,
    dataset_text_field="template",
    max_seq_length=max_seq_length,
    dataset_num_proc=dataset_num_proc,
    peft_config=peft_config,
)
    trainer.train()
    trainer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    fire.Fire(main)