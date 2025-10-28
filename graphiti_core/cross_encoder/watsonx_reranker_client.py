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
from typing import Any,TYPE_CHECKING

import numpy as np
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI

from ..helpers import semaphore_gather
from ..llm_client import LLMConfig, OpenAIClient, RateLimitError
from ..prompts import Message
from .client import CrossEncoderClient

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from langchain_ibm import ChatWatsonx
    from ibm_watsonx_ai.utils.utils import HttpClientConfig
    from ibm_watsonx_ai import APIClient,Credentials
else:
    try:
        from langchain_ibm import ChatWatsonx
        from ibm_watsonx_ai.utils.utils import HttpClientConfig
        from ibm_watsonx_ai import APIClient,Credentials
    except ImportError:
        raise ImportError(
            'langchain_ibm is required for ChatWatsonx.'
        ) from None

import httpx
from pydantic import BaseModel, SecretStr

class WatsonxRerankClient(CrossEncoderClient):
    def __init__(
        self,
        config: LLMConfig
    ):

        self.config = config
        try:
            http_client_config = HttpClientConfig(
                timeout=httpx.Timeout(timeout=2 * 60, connect=10),
                limits=httpx.Limits(
                    max_connections=40,
                    max_keepalive_connections=15,
                    keepalive_expiry=10, # maximum time (in seconds) an idle connection will stay in the pool.
                )
            )

            self.watsonx_api_client = APIClient(
                Credentials(api_key = config.api_key, url = config.base_url),
                httpx_client=http_client_config,
                async_httpx_client=http_client_config)
        except TypeError as e:
            logger.warning(f"Error creating Watsonx Api Client: {e}")
            raise e
            self.client = client


    async def rerank_inference_one(self, messages: list[Message]):

        client = ChatWatsonx(
            model_id=self.config.model,
            url= SecretStr(self.config.base_url if self.config.base_url else ""),
            project_id=self.config.project_id,
            params={
                "top_logprobs": 2,
                "temperature": 0,
                "max_completion_tokens": 1,
                "logprobs": True,
                "logit_bias": {'4139': 30, '2575': 30} # True & False tokenId, upgrade the probs for these two token
            },           
            watsonx_client=self.watsonx_api_client
        )

        msgs: list[dict[str, str]] = []
        for m in messages:
            if m.role == 'user':
                msgs.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                msgs.append({'role': 'system', 'content': m.content})

        return await client.ainvoke(msgs)
    
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        openai_messages_list: Any = [
            [
                Message(
                    role='system',
                    content='You are an expert tasked with determining whether the passage is relevant to the query',
                ),
                Message(
                    role='user',
                    content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                ),
            ]
            for passage in passages
        ]

        try:
            from langchain_core.messages import AIMessage
            responses:list[AIMessage] = await semaphore_gather(
                *[
                    self.rerank_inference_one(
                        messages=openai_messages,
                    )
                    for openai_messages in openai_messages_list
                ]
            )

            responses_top_logprobs = [
                (response.response_metadata.get("logprobs", {}).get("content",[])
                if len(response.response_metadata.get("logprobs", {}).get("content",[]))>0
                else [])
                for response in responses
            ]
            scores: list[float] = []
            for content in responses_top_logprobs:
                if len(content) == 0:
                    continue

                norm_logprobs = content[0].get("top_logprobs",[])
                if len(norm_logprobs) == 0:
                    continue
                    
                if norm_logprobs[0].get("token").strip().split(' ')[0].lower() == 'true':
                    scores.append(np.exp(norm_logprobs[0].get("logprob")))
                else:
                    scores.append(1 - np.exp(norm_logprobs[0].get("logprob")))

            results = [(passage, round(float(score), 4)) for passage, score in zip(passages, scores, strict=True)]
            results.sort(reverse=True, key=lambda x: x[1])

            return results
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

async def test():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from graphiti_core.llm_client.openai_client import OpenAIClient
    # ---- Ollama (OpenAI-compatible) ----
    llm_config = LLMConfig(
        api_key=os.environ["WATSONX_APIKEY"],
        model=os.environ["WATSONX_MODEL"], 
        small_model=os.environ["WATSONX_MODEL"], # Use the same model as the model.
        base_url=os.environ["WATSONX_URL"],
        max_tokens=8192,
        project_id=os.environ["WATSONX_PROJECT_ID"]
    )

    cross_encoder = WatsonxRerankClient(config=llm_config)
    r  = await cross_encoder.rank("what is java", ["java is a language", "python is not good."])
    print(r)

import asyncio
asyncio.run(test())