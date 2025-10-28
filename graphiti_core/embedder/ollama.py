"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from collections.abc import Iterable
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from ollama import AsyncClient
# else:
#     try:
#         from ollama import AsyncClient
#     except ImportError:
#         raise ImportError(
#             'ollama is required for OllamaEmbedder. '
#             'Install it with: pip install graphiti-core[ollama]'
#         ) from None

from pydantic import Field

from .client import EmbedderClient, EmbedderConfig
from ..helpers import semaphore_gather

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'Qwen3-Embedding-8B'
DEFAULT_BATCH_SIZE = 100

import httpx

class OllamaEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str
    base_url: str

class OllamaEmbedder(EmbedderClient):
    """
    Ollama Embedder Client
    """
    def __init__(
        self,
        config: OllamaEmbedderConfig,
        batch_size: int | None = None,
    ):
        self.config = config
        
        if batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size
        
        self.limits = httpx.Limits(max_connections=self.batch_size, max_keepalive_connections=20)


    # async def create(self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> list[float]:
    async def create(self, input_data: str) -> list[float]: 

        async with httpx.AsyncClient(limits=self.limits) as client:
            response = await client.post(self.config.base_url, 
                                         json={"input": input_data,
                                               "model": self.config.embedding_model},
                                         headers={
                                             "Authorization": self.config.api_key
                                         })
            data = response.json()
            # print(data)
            return data.get("data")[0].get("embedding")


    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        
        return await semaphore_gather(
            *[
                self.create(
                    input_data,
                )
                for input_data in input_data_list
            ]
        )


async def test():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    embedder = OllamaEmbedder(
        config=OllamaEmbedderConfig(
            api_key=os.environ["EMBEDDING_APIKEY"],
            embedding_model=os.environ["EMBEDDING_MODEL"],
            embedding_dim=4096,
            base_url=os.environ["EMBEDDING_URL"],
        )
    )

    r  = await embedder.create_batch(["java is a language", "python is not good."])
    print(r)

import asyncio
asyncio.run(test())
