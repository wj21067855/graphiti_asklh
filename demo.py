# refer : https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types
import asyncio
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.watsonx_client import WatsonxClient
from graphiti_core.embedder.ollama import OllamaEmbedder, OllamaEmbedderConfig
from graphiti_core.cross_encoder.watsonx_reranker_client import WatsonxRerankClient
from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_EPISODE_MENTIONS

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

# https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# Custom Entity Types
class Person(BaseModel):
    """A person entity with biographical information."""
    age: Optional[int] = Field(None, description="Age of the person")
    occupation: Optional[str] = Field(None, description="Current occupation")
    location: Optional[str] = Field(None, description="Current location")
    birth_date: Optional[datetime] = Field(None, description="Date of birth")

class Company(BaseModel):
    """A business organization."""
    industry: Optional[str] = Field(None, description="Primary industry")
    founded_year: Optional[int] = Field(None, description="Year company was founded")
    headquarters: Optional[str] = Field(None, description="Location of headquarters")
    employee_count: Optional[int] = Field(None, description="Number of employees")

class Product(BaseModel):
    """A product or service."""
    category: Optional[str] = Field(None, description="Product category")
    price: Optional[float] = Field(None, description="Price in USD")
    release_date: Optional[datetime] = Field(None, description="Product release date")

# Custom Edge Types
class Employment(BaseModel):
    """Employment relationship between a person and company."""
    position: Optional[str] = Field(None, description="Job title or position")
    start_date: Optional[datetime] = Field(None, description="Employment start date")
    end_date: Optional[datetime] = Field(None, description="Employment end date")
    salary: Optional[float] = Field(None, description="Annual salary in USD")
    is_current: Optional[bool] = Field(None, description="Whether employment is current")

class Investment(BaseModel):
    """Investment relationship between entities."""
    amount: Optional[float] = Field(None, description="Investment amount in USD")
    investment_type: Optional[str] = Field(None, description="Type of investment (equity, debt, etc.)")
    stake_percentage: Optional[float] = Field(None, description="Percentage ownership")
    investment_date: Optional[datetime] = Field(None, description="Date of investment")

class Partnership(BaseModel):
    """Partnership relationship between companies."""
    partnership_type: Optional[str] = Field(None, description="Type of partnership")
    duration: Optional[str] = Field(None, description="Expected duration")
    deal_value: Optional[float] = Field(None, description="Financial value of partnership")

entity_types = {
    "Person": Person,
    "Company": Company,
    "Product": Product
}
edge_types = {
    "Employment": Employment,
    "Investment": Investment,
    "Partnership": Partnership
}
edge_type_map = {
    ("Person", "Company"): ["Employment"],
    ("Company", "Company"): ["Partnership", "Investment"],
    ("Person", "Person"): ["Partnership"],
    ("Entity", "Entity"): ["Investment"],  # Apply to any entity type
}


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
        name="Business Update",
        episode_body="Sarah joined TechCorp as CTO in January 2023 with a $200K salary. TechCorp partnered with DataCorp in a $5M deal.",
        source_description="Business news",
        reference_time=datetime.now(),
        entity_types=entity_types,
        edge_types=edge_types,
        edge_type_map=edge_type_map
    )

if __name__ == "__main__":
    asyncio.run(main())