from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
import torch
import fire

def main(model_path:str,
         system_prompt:str=None) -> None:
    model = AutoModelForCausalLM.from_pretrained(model_path,
    use_flash_attention_2=True,
    torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)
    pipe = pipeline(task="text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device="cuda:1",
                streamer=streamer,
                max_new_tokens=8192,
                repetition_penalty=1.3,
                do_sample=True,
                top_k=10,
                top_p=0.98,
                temperature=0.1,)
                
    LLM = HuggingFacePipeline(
    pipeline=pipe
    )

    if system_prompt == None:
        SYSTEM_PROMPT = """### System:
        당신은 사람을 돕는 친절한 인공지능 비서 신궁입니다. 
        사람을 흉내내지 않고, 오직 신궁으로서만 답변합니다. 
        사람의 지시, 질문에 상세하게 아는 만큼 답변하고, 모르거나, 부정확한 내용은 사람에게 언급해주세요."""
    else:
        SYSTEM_PROMPT = system_prompt

    template = """
    {history}
    ### USER:
    사람: {input}
    ### ASSISTANT:
    신궁: """
    template = SYSTEM_PROMPT + template
    prompt = PromptTemplate.from_template(template)

    memory = ConversationTokenBufferMemory(
        llm=LLM,
        max_token_limit=4096,
        human_prefix="사람",
        ai_prefix="신궁",
        prompt=prompt
    )

    chain = ConversationChain(
        llm=LLM,
        memory=memory,
        verbose=True,
        prompt=prompt
    )

    print("채팅 시작, 종료 하려면 exit 를 입력하세요")
    turn_count = 0
    while True:
        
        user_input = input()
        if user_input == "exit":
            print("채팅을 종료합니다.")
            break
        elif user_input == "reset":
            print("채팅을 리셋합니다")
            chain.memory.clear()
        model_output = chain.predict(
            input=user_input,
        )
        # chain.memory.save_context(
        #     {"사람":user_input},
        #     {"신궁":model_output},
        # )
        
        history = chain.memory.load_memory_variables({})["history"]
        length = tokenizer(history, return_length=True)["length"][0]
        turn_count += 1
        print(history,length,sep="\n")
        print("턴 카운트: ", turn_count)

if __name__ == "__main__":
    fire.Fire(main)