import os
import constants
import together
from typing import Any, Dict, Optional
from pydantic import Extra, Field, root_validator
from langchain.llms.base import LLM
from langchain.schema import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from prettytable import PrettyTable
from termcolor import colored

os.environ["TOGETHER_API_KEY"] = constants.TOGETHER_API_KEY


class TogetherLLM(LLM):
    """Large language models from Together."""

    model: str = "mistralai/Mistral-7B-v0.1"
    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    temperature: float = 0.0
    max_tokens: int = 512

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = os.environ["TOGETHER_API_KEY"]
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        together.api_key = self.together_api_key
        output = together.Complete.create(
            prompt=prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = output["output"]["choices"][0]["text"]
        return text


llm = TogetherLLM(
    model="mistralai/Mistral-7B-Instruct-v0.1", temperature=0.1, max_tokens=512
)

# embeddings = HuggingFaceInstructEmbeddings(
#     model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}
# )
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
)

docs = [
    Document(
        page_content="Complex, layered, rich red with dark fruit flavors",
        metadata={
            "name": "Opus One",
            "year": 2018,
            "rating": 96,
            "grape": "Cabernet Sauvignon",
            "color": "red",
            "country": "USA",
        },
    ),
    Document(
        page_content="Luxurious, sweet wine with flavors of honey, apricot, and peach",
        metadata={
            "name": "Château d'Yquem",
            "year": 2015,
            "rating": 98,
            "grape": "Sémillon",
            "color": "white",
            "country": "France",
        },
    ),
]

vectorstore = Chroma.from_documents(docs, embeddings)

metadata_field_info = [
    AttributeInfo(
        name="grape",
        description="The grape used to make the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="name",
        description="The name of the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="color",
        description="The color of the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the wine was released",
        type="integer",
    ),
    AttributeInfo(
        name="country",
        description="The name of the country the wine comes from",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="The Robert Parker rating for the wine 0-100",
        type="integer",
    ),
]

document_content_description = "Brief description of the wine"


def print_documents(docs):
    table = PrettyTable()
    table.field_names = [
        "Page Content",
        "Color",
        "Country",
        "Grape",
        "Name",
        "Rating",
        "Year",
    ]

    for doc in docs:
        table.add_row(
            [
                doc.page_content,
                colored(doc.metadata["color"], "red"),
                colored(doc.metadata["country"], "yellow"),
                colored(doc.metadata["grape"], "blue"),
                colored(doc.metadata["name"], "green"),
                colored(doc.metadata["rating"], "magenta"),
                colored(doc.metadata["year"], "cyan"),
            ]
        )
    print(table)


retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
)

print("Q: Who is Gary Oldman? ")
print(llm("Who is Gary Oldman? "))

print("Q: What are some red wines")
print_documents(retriever.get_relevant_documents("What are some red wines"))
