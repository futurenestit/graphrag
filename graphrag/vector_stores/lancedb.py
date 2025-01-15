# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The LanceDB vector storage implementation package."""

import json  # noqa: I001
from typing import Any

import pyarrow as pa

from django.conf import settings
from graphrag.model.types import TextEmbedder

from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)
import lancedb


class LanceDBVectorStore(BaseVectorStore):
    """LanceDB vector storage implementation."""

    def __init__(self, root_dir: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.storage_options = {
            "aws_access_key_id": settings.AWS_S3_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_S3_SECRET_ACCESS_KEY,
            "endpoint_url": settings.AWS_S3_ENDPOINT_URL,
            "region_name": settings.AWS_S3_REGION_NAME,
            "allow_http": "true",
        }

    def connect(self, **kwargs: Any) -> Any:
        """Connect to the vector storage."""

        db_uri = f"{self.root_dir}/{kwargs['db_uri']}".replace("\\", "/")
        self.db_connection = lancedb.connect(
            f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{db_uri}",
            storage_options=self.storage_options,
        )

        if (
            self.collection_name
            and self.collection_name in self.db_connection.table_names()
        ):
            self.document_collection = self.db_connection.open_table(
                self.collection_name
            )

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into vector storage."""
        data = [
            {
                "id": document.id,
                "text": document.text,
                "vector": document.vector,
                "attributes": json.dumps(document.attributes),
            }
            for document in documents
            if document.vector is not None
        ]

        if len(data) == 0:
            data = None

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float64())),
            pa.field("attributes", pa.string()),
        ])
        # NOTE: If modifying the next section of code, ensure that the schema remains the same.
        #       The pyarrow format of the 'vector' field may change if the order of operations is changed
        #       and will break vector search.
        if overwrite:
            if data:
                self.document_collection = self.db_connection.create_table(
                    self.collection_name, data=data, mode="overwrite"
                )
            else:
                self.document_collection = self.db_connection.create_table(
                    self.collection_name, schema=schema, mode="overwrite"
                )
        else:
            # add data to existing table
            self.document_collection = self.db_connection.open_table(
                self.collection_name
            )
            if data:
                self.document_collection.add(data)

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""
        if len(include_ids) == 0:
            self.query_filter = None
        else:
            if isinstance(include_ids[0], str):
                id_filter = ", ".join([f"'{id}'" for id in include_ids])
                self.query_filter = f"id in ({id_filter})"
            else:
                self.query_filter = (
                    f"id in ({', '.join([str(id) for id in include_ids])})"
                )
        return self.query_filter

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        if self.query_filter:
            docs = (
                self.document_collection.search(
                    query=query_embedding, vector_column_name="vector"
                )
                .where(self.query_filter, prefilter=True)
                .limit(k)
                .to_list()
            )
        else:
            docs = (
                self.document_collection.search(
                    query=query_embedding, vector_column_name="vector"
                )
                .limit(k)
                .to_list()
            )
        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=doc["id"],
                    text=doc["text"],
                    vector=doc["vector"],
                    attributes=json.loads(doc["attributes"]),
                ),
                score=1 - abs(float(doc["_distance"])),
            )
            for doc in docs
        ]

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        query_embedding = text_embedder(text)
        if query_embedding:
            return self.similarity_search_by_vector(query_embedding, k)
        return []

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Search for a document by id."""
        doc = (
            self.document_collection.search()
            .where(f"id == '{id}'", prefilter=True)
            .to_list()
        )
        if doc:
            return VectorStoreDocument(
                id=doc[0]["id"],
                text=doc[0]["text"],
                vector=doc[0]["vector"],
                attributes=json.loads(doc[0]["attributes"]),
            )
        return VectorStoreDocument(id=id, text=None, vector=None)
