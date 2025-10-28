import json
import logging
import typing
from typing import TYPE_CHECKING

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
from pydantic import BaseModel, SecretStr
import httpx
from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize
from .errors import RateLimitError


logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'qwen3:4b'
DEFAULT_MAX_TOKENS = 8192

class WatsonxClient(LLMClient):
    """Watsonx async client wrapper for Graphiti.
    """

    def __init__(self, config: LLMConfig, cache: bool = False):
        super().__init__(config, cache)

        try:
            # DEFAULT
            http_client_config = HttpClientConfig(
                timeout=httpx.Timeout(timeout=2 * 60, connect=10),
                limits=httpx.Limits(
                    max_connections=40,
                    max_keepalive_connections=15,
                    keepalive_expiry=10, # maximum time (in seconds) an idle connection will stay in the pool.
                )
            )

            self.watsonx_api_client = APIClient(
                Credentials(api_key = config.api_key,url = config.base_url),
                httpx_client=http_client_config,
                async_httpx_client=http_client_config)
        except TypeError as e:
            logger.warning(f"Error creating Watsonx Api Client: {e}")
            raise e

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        msgs: list[dict[str, str]] = []
        for m in messages:
            if m.role == 'user':
                msgs.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                msgs.append({'role': 'system', 'content': m.content})

        try:
            # Prepare options
            options: dict[str, typing.Any] = {}
            if max_tokens is not None:
                options['max_tokens'] = max_tokens
            if self.temperature is not None:
                options['temperature'] = self.temperature

            client = ChatWatsonx(
                model_id=self.config.model,
                url= SecretStr(self.config.base_url if self.config.base_url else ""),
                project_id=self.config.project_id,
                params={
                    "decoding_method": "sample",
                    "top_p": 1.0,
                    "top_k": 10,
                    # enhance these parameter in LLMConfig if needed.

                    "min_new_tokens": 1,
                    "repetition_penalty": 1,
                    "max_new_tokens": max_tokens,
                    "temperature": 0.2,
                    **options
                },
                
                watsonx_client=self.watsonx_api_client
            )
            
            if response_model is not None:
                r1 = await client.with_structured_output(response_model).ainvoke(msgs)
                if isinstance(r1, BaseModel):
                    return r1.model_dump()
                else:
                    return r1
            else:
                from langchain_core.messages import AIMessage
                response:AIMessage = await client.ainvoke(msgs)

                try:
                    if isinstance(response.content, str):
                        return json.loads(response.content) # Could enhance to extract the data we want???
                    else:
                        return {'text': response.content}
                except Exception:
                    return {'text': response.content}
        except Exception as e:
            # map obvious ollama rate limit / response errors to RateLimitError when possible
            err_name = e.__class__.__name__
            status_code = getattr(e, 'status_code', None) or getattr(e, 'status', None)
            if err_name in ('RequestError', 'ResponseError') and status_code == 429:
                raise RateLimitError from e
            logger.error(f'Error in generating LLM response (ollama): {e}')
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

    client = WatsonxClient(config=llm_config)

    r  = await client._generate_response(messages=[
       Message(role = "user", content = "what is java")
    ])
    print(r)

import asyncio
asyncio.run(test())