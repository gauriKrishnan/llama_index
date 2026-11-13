# Copyright Hewlett Packard Enterprise Development LP.

from pydi_client import DIClient
from typing import Any, Dict, Optional, Union

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode


class AlletraX10000Retriever(BaseRetriever):
    """Retriever over an AlletraMP X10000 Data Intelligence collection.

    Pass ``s3_access_key``/``s3_secret_key`` for a bucket collection, or
    ``file_username``/``file_password`` for a filesystem collection.
    """

    def __init__(
        self,
        uri: str,
        collection_name: str,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        search_config: Union[Any, Dict[str, Any]] = None,
        top_k: int = 5,
        file_username: Optional[str] = None,
        file_password: Optional[str] = None,
    ):
        self.uri = uri
        self.top_k = top_k
        self.collection_name = collection_name
        self.access_key = s3_access_key
        self.secret_key = s3_secret_key
        self.file_username = file_username
        self.file_password = file_password
        self.search_config = search_config

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str

        client = DIClient(uri=self.uri)
        data = client.similarity_search(
            collection_name=self.collection_name,
            query=query,
            top_k=self.top_k,
            access_key=self.access_key,
            secret_key=self.secret_key,
            file_username=self.file_username,
            file_password=self.file_password,
            search_parameters=self.search_config,
        )

        nodes = []
        for item in data:
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=item["dataChunk"],
                        metadata=item.get("chunkMetadata") or {},
                    ),
                    score=item["score"],
                )
            )
        return nodes
