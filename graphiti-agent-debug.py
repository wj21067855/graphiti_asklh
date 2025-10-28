import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from graphiti_core.llm_client.config import LLMConfig
# from graphiti_core.llm_client.openai_client import OpenAIClient
# from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
# from graphiti_core.llm_client.ollama_client import OllamaClient
from graphiti_core.llm_client.watsonx_client import WatsonxClient
# from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.embedder.ollama import OllamaEmbedder, OllamaEmbedderConfig
from graphiti_core.cross_encoder.watsonx_reranker_client import WatsonxRerankClient
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_EPISODE_MENTIONS

from opentelemetry import trace
from graphiti_core.tracer import OpenTelemetryTracer

# set up log
# https://github.com/getzep/graphiti/blob/main/examples/podcast/podcast_runner.py#L76

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


async def main():
    load_dotenv()
    setup_logging()

    #LLM Client
    llm_config = LLMConfig(
        api_key=os.environ["WATSONX_APIKEY"],
        model=os.environ["WATSONX_MODEL"], 
        small_model=os.environ["WATSONX_MODEL"], # Use the same model as the model.
        base_url=os.environ["WATSONX_URL"],
        max_tokens=8192,
        project_id=os.environ["WATSONX_PROJECT_ID"]
    )

    llm_client = WatsonxClient(config=llm_config)

    #Embedder
    embedder = OllamaEmbedder(
        config=OllamaEmbedderConfig(
            api_key=os.environ["EMBEDDING_APIKEY"],
            embedding_model=os.environ["EMBEDDING_MODEL"],
            embedding_dim=4096,
            base_url=os.environ["EMBEDDING_URL"],
        )
    )

    #Rerank
    llm_config_rerank = LLMConfig(
        api_key=os.environ["WATSONX_APIKEY"],
        model=os.environ["WATSONX_RERANK_MODEL"], 
        # small_model=os.environ["WATSONX_RERANK_MODEL"], # Use the same model as the model.
        base_url=os.environ["WATSONX_URL"],
        max_tokens=8192,
        project_id=os.environ["WATSONX_PROJECT_ID"]
    )

    cross_encoder = WatsonxRerankClient(config=llm_config_rerank)


    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # 设置导出方式（这里是打印到控制台）
    exporter = ConsoleSpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    client = Graphiti(
        os.environ.get('NEO4J_URI'),
        os.environ.get('NEO4J_USER'),
        os.environ.get('NEO4J_PASSWORD'),
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        tracer=OpenTelemetryTracer(tracer=trace.get_tracer("track me: "))
    )

    # Note: This will clear the database
    await clear_data(client.driver)
    await client.build_indices_and_constraints()

    await client.add_episode(
        name='IBM CyberDefend | Overview',
        episode_body=(f"""
        "john likes dog"
"""),
        source=EpisodeType.text,
        reference_time=datetime.now(timezone.utc),
        source_description='SalesBot2',
    )



    # user_name = 'Alice'
    # await client.add_episode(
    #     name='User Creation',
    #     episode_body=(f'{user_name} is interested in buying a pair of shoes'),
    #     source=EpisodeType.text,
    #     reference_time=datetime.now(timezone.utc),
    #     source_description='SalesBot',
    # )

    # await client.add_episode(
    #     name='Example 2',
    #     episode_body=(f'Jon met Alice at DEF CON 32 in Las Vegas in Aug 2024.'),
    #     source=EpisodeType.text,
    #     reference_time=datetime.now(timezone.utc),
    #     source_description='SalesBot',
    # )

    # await client.add_episode(
    #     name='User Creation 2',
    #     episode_body=(f'Jon likes Alice'),
    #     source=EpisodeType.text,
    #     reference_time=datetime.now(timezone.utc),
    #     source_description='SalesBot2',
    # )

    # let's get Jess's node uuid
    # nl = await client._search("what is Alice intereste in?", NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
    # print(nl)
    # print(type(nl))

    # for result in nl:
    #     print(result)

    # and the ManyBirds node uuid
    # nl = await client._search('ManyBirds', NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
    # print(nl)


if __name__ == "__main__":
    asyncio.run(main())