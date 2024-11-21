from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pydantic import Field
from typing import Any, List, Optional
import os
import json


class CustomHuggingFaceEndpointEmbeddings(HuggingFaceEndpointEmbeddings):
    """Custom HuggingFaceEndpointEmbeddings with `model_name` support."""

    model_name: Optional[str] = None

    def __init__(self, *args, model_name: Optional[str] = None, **kwargs):
        """
        Initialize the custom class with the optional `model_name`.

        Args:
            model_name: Explicit name of the model (e.g., "BAAI/bge-m3").
            *args: Positional arguments for the parent class.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        self.model_name = model_name or self.model

    def embed_documents(self, texts: List[Any]) -> List[List[float]]:
        """
        Call out to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # Convert all elements to strings and replace newlines
        texts = [str(text).replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        responses = self.client.post(
            json={"inputs": texts, **_model_kwargs}, task=self.task
        )
        return json.loads(responses.decode())

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async call to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        responses = await self.async_client.post(
            json={"inputs": texts, "parameters": _model_kwargs}, task=self.task
        )
        return json.loads(responses.decode())

    def embed_query(self, text: str) -> List[float]:
        """
        Call out to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = self.embed_documents([text])[0]
        return response

    async def aembed_query(self, text: str) -> List[float]:
        """
        Async call to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = (await self.aembed_documents([text]))[0]
        return response



bge_m3_embed = CustomHuggingFaceEndpointEmbeddings(
    model_name='BAAI_bge_m3',
    model='http://100.67.185.22:8080',
)  # docker name musing_blackburn

qwen2_embed = CustomHuggingFaceEndpointEmbeddings(
    model_name='Alibaba-NLP_gte-Qwen2-7B',
    model='http://100.67.185.22:8083',
)  # docker name nostalgic_khayyam

nomic_embed = CustomHuggingFaceEndpointEmbeddings(
    model_name='nomic-ai_nomic-embed-text-v1_5',
    model='http://100.67.185.22:8082',
)  # docker name thirsty_heisenberg
