from google.cloud import aiplatform_v1beta1
from google.protobuf import struct_pb2
from google.cloud.aiplatform_v1beta1.types import IndexDatapoint
from typing import Optional, Any

from google_vertex_vector_utilities.vector_utils import ShardSize

class AsyncVectorHandler:

    def __init__(self, google_cloud_project: str, location: str) -> None:
        self.google_cloud_project = google_cloud_project
        self.location = location

        self.parent = f"projects/{self.google_cloud_project}/locations/{self.location}"

        self.index_client = aiplatform_v1beta1.IndexServiceAsyncClient(
            client_options = { # type:ignore
                "api_endpoint": f"{self.location}-aiplatform.googleapis.com"
            }
        )
        self.endpoint_client = aiplatform_v1beta1.IndexEndpointServiceAsyncClient(
            client_options = { # type:ignore
                "api_endpoint": f"{self.location}-aiplatform.googleapis.com"
            }
        )

    async def load(self,
                   index_id: Optional[str] = None,
                   endpoint_id: Optional[str] = None,
                   deployed_index_id: Optional[str] = None) -> Any:
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.deployment_id = deployed_index_id

        if self.endpoint_id:
            self.match_client = aiplatform_v1beta1.MatchServiceAsyncClient(
                client_options = { # type:ignore
                    "api_endpoint": await self._get_endpoint_url()
                }
            )

        if not self.index_id:
            try:
                self.index_id = await self._get_index_id()
            except ValueError as e:
                raise e

        return self

    async def create_index(self,
                           index_display_name: str,
                           index_dimension_count: int = 1536,
                           approximate_neighbors_count: int = 150,
                           distance_measure_type: str = "DOT_PRODUCT_DISTANCE",
                           shard_size: str = "SMALL",
                           leaf_node_embedding_count: int = 1000,
                           leaf_nodes_to_search_pct: int = 10) -> str:
        shard_size_enum = ShardSize.from_string(shard_size)

        self.index_config = {
            "display_name": index_display_name,
            "metadata": {
                "config": {
                    "dimensions": index_dimension_count,
                    "approximateNeighborsCount": approximate_neighbors_count,
                    "distanceMeasureType": distance_measure_type,
                    "shardSize": shard_size_enum.value,
                    "algorithmConfig": {
                        "treeAhConfig": {
                            "leafNodeEmbeddingCount": leaf_node_embedding_count,
                            "leafNodesToSearchPercent": leaf_nodes_to_search_pct
                        }
                    }
                }
            },
            "index_update_method": "STREAM_UPDATE"
        }

        operation = await self.index_client.create_index(
            parent = self.parent,
            index = self.index_config # type:ignore
        )
        result = await operation.result()
        self.index_id = result.name 

        return self.index_id

    async def create_endpoint(self, endpoint_display_name: str) -> str:
        operation = await self.endpoint_client.create_index_endpoint(
            parent = self.parent,
            index_endpoint = { # type:ignore
                "display_name": endpoint_display_name
            }
        )
        result = await operation.result() # type:ignore
        self.endpoint_id = result.name

        self.match_client = aiplatform_v1beta1.MatchServiceAsyncClient(
            client_options = { # type:ignore
                "api_endpoint": await self._get_endpoint_url()
            }
        )

        return self.endpoint_id

    async def deploy_index(self,
                           display_name: str,
                           endpoint_id: Optional[str] = None,
                           index_id: Optional[str] = None,
                           machine_type: Optional[str] = "e2-standard-2",
                           min_replica_count: Optional[int] = 1,
                           max_replica_count: Optional[int] = 1) -> str:
        display_name = display_name.replace("-", "_")

        endpoint_id = endpoint_id if endpoint_id else self.endpoint_id
        index_id = index_id if index_id else self.index_id

        await self.endpoint_client.deploy_index(
            index_endpoint = endpoint_id,
            deployed_index = { # type:ignore
                "id": display_name,
                "index": index_id,
                "dedicated_resources": {
                    "machine_spec": {
                        "machine_type": machine_type
                    },
                    "min_replica_count": min_replica_count,
                    "max_replica_count": max_replica_count
                }
            }
        )

        self.deployment_id = display_name

        return self.deployment_id

    async def insert_vector(self,
                            vector_id: str,
                            vector_embedding: list[float],
                            vector_filters: list[dict[str, list[str]]] | None = None,
                            vector_metadata: dict[str, str] | None = None) -> None:
        restrictions = []
        if vector_filters:
            for vfil in vector_filters:
                if vfil.get("allow_list") and len(vfil.get("allow_list")) > 0:
                    allow_list = [str(item) for item in vfil.get("allow_list") if item]
                    
                    if allow_list:
                        restriction = IndexDatapoint.Restriction(
                            namespace = str(vfil.get("namespace")),
                            allow_list = allow_list
                        )
                        restrictions.append(restriction)

        datapoint_kwargs = {
            "datapoint_id": vector_id,
            "feature_vector": vector_embedding
        }

        if restrictions:
            datapoint_kwargs["restricts"] = restrictions

        if vector_metadata:
            metadata_struct = struct_pb2.Struct()
            metadata_struct.update(vector_metadata)
            datapoint_kwargs["embedding_metadata"] = metadata_struct

        datapoint = IndexDatapoint(**datapoint_kwargs)

        request = aiplatform_v1beta1.UpsertDatapointsRequest(
            index = self.index_id,
            datapoints = [datapoint]
        )

        await self.index_client.upsert_datapoints(request=request)

    async def delete_vector(self,
                            vector_ids: list[str] | str) -> None:
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        request = aiplatform_v1beta1.RemoveDatapointsRequest(
            index = self.index_id,
            datapoint_ids = vector_ids
        )

        await self.index_client.remove_datapoints(request=request)

    async def query_index(self,
                          embedding: list[float],
                          query_filters: list[dict[str, str | list[str]]] | None = None,
                          num_results: int = 10,
                          include_metadata: bool = True) -> list[dict]:
        if not self.match_client:
            try:
                self.match_client = aiplatform_v1beta1.MatchServiceAsyncClient(
                    client_options = { # type:ignore
                        "api_endpoint": await self._get_endpoint_url()
                    }
                )
            except Exception as e:
                raise e

        restricts = []
        if query_filters:
            for qfil in query_filters:
                namespace = qfil.get("namespace")
                allow_list = qfil.get("allow_list")

                if namespace and allow_list and len(allow_list) > 0:
                    cleaned_list = [str(item) for item in allow_list if item]

                    if cleaned_list:
                        restricts.append(
                            IndexDatapoint.Restriction(
                                namespace = str(namespace),
                                allow_list = cleaned_list
                            )
                        )

        datapoint_kwargs = {
            "feature_vector": embedding
        }

        if restricts:
            datapoint_kwargs["restricts"] = restricts

        datapoint = IndexDatapoint(**datapoint_kwargs)

        query = aiplatform_v1beta1.FindNeighborsRequest.Query(
            datapoint = datapoint,
            neighbor_count = num_results
        )

        request = aiplatform_v1beta1.FindNeighborsRequest(
            index_endpoint = self.endpoint_id,
            deployed_index_id = self.deployment_id,
            queries = [query],
            return_full_datapoint = include_metadata
        )

        response = await self.match_client.find_neighbors(request=request)

        json_responses = []
        for query_result in response.nearest_neighbors:
            for neighbor in query_result.neighbors:
                result = {
                    "datapoint_id": neighbor.datapoint.datapoint_id
                }
                if include_metadata and neighbor.datapoint.embedding_metadata:
                    result["metadata"] = dict(neighbor.datapoint.embedding_metadata)
                json_responses.append(result)

        return json_responses

    async def search_index(self,
                           vector_ids: list[str],
                           include_metadata: bool = True,
                           include_embeddings: bool = False) -> list[dict]:
        if not self.match_client:
            try:
                self.match_client = aiplatform_v1beta1.MatchServiceAsyncClient(
                    client_options = { #type:ignore
                        "api_endpoint": await self._get_endpoint_url()
                    }
                )
            except Exception as e:
                raise e

        request = aiplatform_v1beta1.ReadIndexDatapointsRequest(
            index_endpoint = self.endpoint_id,
            deployed_index_id = self.deployment_id,
            ids = vector_ids
        )

        response = await self.match_client.read_index_datapoints(request=request)

        results = []
        for datapoint in response.datapoints:
            result = {
                "datapoint_id": datapoint.datapoint_id
            }

            if include_metadata and datapoint.embedding_metadata:
                result["metadata"] = dict(datapoint.embedding_metadata) # type:ignore

            if include_embeddings and datapoint.feature_vector:
                result["embedding"] = list(datapoint.feature_vector) # type:ignore

            results.append(result)

        return results

    async def get_all_ids(self) -> dict[str, Any]:
        return {
            "API_ENDPOINT": await self._get_endpoint_url(),
            "INDEX_ENDPOINT": self.endpoint_id,
            "DEPLOYED_INDEX_ID": self.deployment_id
        }

    async def _get_endpoint_url(self) -> str:
        endpoint = await self.endpoint_client.get_index_endpoint(name=self.endpoint_id)
        return endpoint.public_endpoint_domain_name

    async def _get_index_id(self) -> str:
        endpoint = await self.endpoint_client.get_index_endpoint(name=self.endpoint_id)
        if endpoint.deployed_indexes:
            index_resource_name = endpoint.deployed_indexes[0].index
            return index_resource_name
        else:
            raise ValueError("No deployed indexes found on this endpoint")
