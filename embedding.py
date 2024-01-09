from util import get_hf_model_tokenizer
from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
import torch
from tqdm import tqdm


WEIGHTED_DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PROMPTED_DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

class HuggingFaceLLMWeightedEmbeddings(BaseModel, Embeddings):
    client: Any  #: :meta private:
    model_name: str = WEIGHTED_DEFAULT_MODEL_NAME
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    tokenizer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    use_gpu: bool = True
    """Run embedding on GPUs."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

        except ImportError as exc:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            ) from exc
        
        if self.model_name == WEIGHTED_DEFAULT_MODEL_NAME:
            print(f"you are using default model {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **self.tokenizer_kwargs,
        )
        if self.use_gpu == True and "cuda" not in self.model.device.__str__():
            self.model = self.model.cuda()
        self.use_gpu = True if "cuda" in self.model.device.__str__() else False
        self.model.eval()

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        
    def embed(self, text:str) -> List[float]:
        if self.use_gpu:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            last_hidden_state = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
        # to ignore padding or some special tokens
        weights_for_non_padding = inputs.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)
        # weighted multiplied embedding
        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        # weighted sentence embeding
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
        return sentence_embeddings[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        return [self.embed(text) for text in tqdm(texts)]
    

class HuggingFaceLLMPromptedEmbeddings(BaseModel, Embeddings):
    client: Any  #: :meta private:
    model_name: str = PROMPTED_DEFAULT_MODEL_NAME
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    tokenizer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    use_gpu: bool = True
    """Run embedding on GPUs."""
    prompt_template: str = "This sentence: {text} means in one word:"
    "prompted template parameter for embedding"
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

        except ImportError as exc:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            ) from exc
        if self.model_name == PROMPTED_DEFAULT_MODEL_NAME:
            print(f"you are using default model {self.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **self.tokenizer_kwargs,
        )
        if self.use_gpu == True and "cuda" not in self.model.device.__str__():
            self.model = self.model.cuda()
        self.model.eval()

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        
    def embed(self, text:str) -> List[float]:
        """
        Embedding using LLM Prompt Base, last token`s hidden state

        Args:
            text (str): text for embedding

        Returns:
            List[float]: list values with float values
        """
        
        if self.use_gpu:
            inputs = self.tokenizer(self.prompt_template.format(text=text), return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(self.prompt_template.format(text=text), return_tensors="pt")
        with torch.no_grad():
            last_hidden_state = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
        idx_of_the_last_non_padding_token = inputs.attention_mask.bool().sum(1)-1
        sentence_embeddings = last_hidden_state[0][idx_of_the_last_non_padding_token]
        
        return sentence_embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        return [self.embed(text) for text in tqdm(texts)]