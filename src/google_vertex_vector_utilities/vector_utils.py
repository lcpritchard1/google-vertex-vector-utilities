"""Vector search utilities for Google Cloud Vertex AI.

This module provides a handler class for managing vector search operations
using Google Cloud's Vertex AI Vector Search service. It supports creating,
deploying, and querying vector indexes.

Typical usage example:
    handler = VertexVectorHandler('my-project', 'us-central1')
    handler.load(index_id='index-123', endpoint_id='endpoint-456')
    results = handler.query_index(embedding=[0.1, 0.2, ...])
"""

from google.cloud import aiplatform_v1beta1
from google.cloud.aiplatform_v1beta1.types import IndexDatapoint
from typing import Optional, Any
from enum import Enum

class ShardSize(Enum):
    """Shard size options for vector index creation."""
    SMALL = "SHARD_SIZE_SMALL"
    MEDIUM = "SHARD_SIZE_MEDIUM"
    LARGE = "SHARD_SIZE_LARGE"

    @classmethod
    def from_string(cls, size: str) -> 'ShardSize':
        size_upper = size.upper()
        if size_upper == "SMALL":
            return cls.SMALL
        elif size_upper == "MEDIUM":
            return cls.MEDIUM
        elif size_upper == "LARGE":
            return cls.LARGE
        else:
            raise ValueError(f"Invalid shard size: {size}. Must be 'SMALL', 'MEDIUM', or 'LARGE'")

# TODO - Add logging to class operations

class VertexVectorHandler:
    """Manages vector search operations using Vertex AI.
    
    This class provides methods to create, deploy, and query vector indexes
    in Google Cloud's Vertex AI Vector Search service. It handles the full
    lifecycle of vector search operations including index creation, endpoint
    deployment, vector insertion, and similarity search.
    
    Attributes:
        google_cloud_project: The Google Cloud project ID.
        location: The Google Cloud region for Vertex AI resources.
        parent: The parent resource path for Vertex AI operations.
        index_client: Client for index-related operations.
        endpoint_client: Client for endpoint-related operations.
        match_client: Client for performing similarity searches.
        index_id: The ID of the loaded or created index.
        endpoint_id: The ID of the loaded or created endpoint.
        deployment_id: The ID of the deployed index.
    """
    
    def __init__(self,
                 google_cloud_project: str,
                 location: str) -> None:
        """Initializes the VertexVectorHandler.
        
        Args:
            google_cloud_project: The Google Cloud project ID.
            location: The Google Cloud region (e.g., 'us-central1').
        """
        self.google_cloud_project = google_cloud_project
        self.location = location
        
        self.parent = f"projects/{self.google_cloud_project}/locations/{self.location}"
        
        # TODO - Can this API endpoint be a variable?
        self.index_client = aiplatform_v1beta1.IndexServiceClient(
            client_options={"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
        )
        
        self.endpoint_client = aiplatform_v1beta1.IndexEndpointServiceClient(
            client_options={"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
        )
        
    def load(self,
             index_id: Optional[str] = None,
             endpoint_id: Optional[str] = None,
             deployed_index_id: Optional[str] = None) -> Any:
        """Loads existing Vertex AI resources.
        
        Loads existing index and endpoint resources for use in vector operations.
        If endpoint_id is provided but index_id is not, attempts to retrieve
        the index_id from the endpoint.
        
        Args:
            index_id: The ID of an existing vector index.
            endpoint_id: The ID of an existing index endpoint.
            deployed_index_id: The ID of the deployed index on the endpoint.
            
        Raises:
            ValueError: If no deployed indexes are found on the endpoint.
        """
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.deployment_id = deployed_index_id
        
        if self.endpoint_id:
            self.match_client = aiplatform_v1beta1.MatchServiceClient(
                client_options={"api_endpoint": self._get_endpoint_url()}
            )
            
        if not self.index_id:
            try:
                self.index_id = self._get_index_id()
                
            # TODO - This will need to be improved at some point
            except ValueError as e:
                raise e
            
        return self

    def create_index(self,
                     index_display_name: str,
                     index_dimension_count: int = 1536,
                     approximate_neighbors_count: int = 150,
                     distance_measure_type: str = "DOT_PRODUCT_DISTANCE",
                     shard_size: str = "SMALL",
                     leaf_node_embedding_count: int = 1000,
                     leaf_nodes_to_search_pct: int = 10
                     ) -> str:
        """Creates a new vector index.

        Creates a streamable vector index with the specified configuration
        for storing and searching vector embeddings.

        Args:
            index_display_name: Human-readable name for the index.
            index_dimension_count: Number of dimensions in the vector embeddings.
            approximate_neighbors_count: Number of neighbors to consider during search.
            distance_measure_type: Distance metric for similarity calculation.
                Options: 'DOT_PRODUCT_DISTANCE', 'SQUARED_L2_DISTANCE', 'COSINE_DISTANCE'.
            shard_size: Size of index shards. Options: 'SMALL', 'MEDIUM', 'LARGE'.
            leaf_node_embedding_count: Number of embeddings per leaf node.
            leaf_nodes_to_search_pct: Percentage of leaf nodes to search.

        Returns:
            The ID of the created index.
        """

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
            "index_update_method": "STREAM_UPDATE" # For now this is will only create streamable indexes
        }
        
        operation = self.index_client.create_index(parent=self.parent, index=self.index_config) # type:ignore
        self.index_id = operation.result().name # type:ignore
    
        return self.index_id
    
    def create_endpoint(self,
                        endpoint_display_name: str) -> str:
        """Creates a new index endpoint.
        
        Creates an endpoint for deploying and serving vector indexes.
        
        Args:
            endpoint_display_name: Human-readable name for the endpoint.
            
        Returns:
            The ID of the created endpoint.
        """
        endpoint = self.endpoint_client.create_index_endpoint(
            parent=self.parent,
            index_endpoint={"display_name": endpoint_display_name} # type:ignore
        )
        self.endpoint_id = endpoint.result().name # type:ignore
        
        self.match_client = aiplatform_v1beta1.MatchServiceClient(
            client_options={"api_endpoint": self._get_endpoint_url()}
        )
        
        return self.endpoint_id
    
    # TODO - Should add enum for valid machine types based on shard size; e2-standard-2 only valid for small shards
    def deploy_index(self,
                     display_name: str,
                     endpoint_id: Optional[str] = None,
                     index_id: Optional[str] = None,
                     machine_type: Optional[str] = "e2-standard-2",
                     min_replica_count: Optional[int] = 1,
                     max_replica_count: Optional[int] = 1) -> str:
        """Deploys an index to an endpoint.
        
        Deploys a vector index to an endpoint with specified compute resources
        for serving similarity search queries.
        
        Args:
            display_name: Name for the deployed index (hyphens will be replaced
                with underscores).
            endpoint_id: The endpoint to deploy to. Uses self.endpoint_id if None.
            index_id: The index to deploy. Uses self.index_id if None.
            machine_type: GCP machine type for serving the index.
            min_replica_count: Minimum number of replicas for autoscaling.
            max_replica_count: Maximum number of replicas for autoscaling.
            
        Returns:
            The ID of the deployed index.
        """
        display_name = display_name.replace("-", "_")
        
        endpoint_id = endpoint_id if endpoint_id else self.endpoint_id
        index_id = index_id if index_id else self.index_id
        
        self.endpoint_client.deploy_index(
            index_endpoint=endpoint_id,
            deployed_index={ # type:ignore
                "id": display_name,
                "index": index_id,
                "dedicated_resources": {
                    "machine_spec": {"machine_type": machine_type},
                    "min_replica_count": min_replica_count,
                    "max_replica_count": max_replica_count,
                }
            }
        )
        
        self.deployment_id = display_name
        
        return self.deployment_id
    
    def insert_vector(self,
                      vector_id: str,
                      vector_embedding: list[float],
                      vector_metadata: dict[str, str]) -> None:
        """Inserts a vector into the index.
        
        Upserts a single vector embedding with associated metadata into the
        vector index.
        
        Args:
            vector_id: Unique identifier for the vector.
            vector_embedding: The vector embedding values.
            vector_metadata: Key-value pairs of metadata associated with the vector.
        """
        datapoint = [
            IndexDatapoint(
                datapoint_id=vector_id,
                feature_vector=vector_embedding,
                embedding_metadata=vector_metadata
            )
        ]
        
        request = aiplatform_v1beta1.UpsertDatapointsRequest(
            index=self.index_id,
            datapoints=datapoint
        )
        
        self.index_client.upsert_datapoints(request=request)
        
    def query_index(self,
                    embedding: list[float],
                    num_results: int = 10,
                    include_metadata: bool = True) -> list[dict]:
        """Queries the index for similar vectors.
        
        Performs a similarity search to find the nearest neighbors to the
        provided embedding vector.
        
        Args:
            embedding: The query vector embedding.
            num_results: Number of nearest neighbors to return.
            include_metadata: Whether to include metadata in results.
            
        Returns:
            A list of dictionaries containing datapoint IDs and optionally metadata
            for the nearest neighbors. Each dict contains:
                - 'datapoint_id': The ID of the matching vector.
                - 'metadata': (optional) The metadata associated with the vector.
                
        Raises:
            Exception: If the match client cannot be initialized.
        """
        if not self.match_client: # This might error if self.match_client isn't already instantiated
            try:
                self.match_client = aiplatform_v1beta1.MatchServiceClient(
                    client_options={"api_endpoint": self._get_endpoint_url()}
                )
            except Exception as e: # TODO Fix whatever this is
                raise e
            
        datapoint = IndexDatapoint(
            feature_vector=embedding
        )
        
        query = aiplatform_v1beta1.FindNeighborsRequest.Query(
            datapoint=datapoint,
            neighbor_count=num_results
        )
        
        request = aiplatform_v1beta1.FindNeighborsRequest(
            index_endpoint=self.endpoint_id,
            deployed_index_id=self.deployment_id,
            queries=[query],
            return_full_datapoint=include_metadata
        )
        
        response = self.match_client.find_neighbors(request=request)
        
        json_responses = []
        for query_result in response.nearest_neighbors:  # This is per query
            for neighbor in query_result.neighbors:  # This has multiple neighbors
                result = {
                    "datapoint_id": neighbor.datapoint.datapoint_id
                }
                if include_metadata and neighbor.datapoint.embedding_metadata:
                    result["metadata"] = dict(neighbor.datapoint.embedding_metadata) # type:ignore
                json_responses.append(result)
                
        return json_responses

    def search_index(self,
                     vector_ids: list[str],
                     include_metadata: bool = True,
                     include_embeddings: bool = False) -> list[dict]:
        """Searches for vectors by their exact IDs.

          Retrieves specific vectors from the index by their datapoint IDs rather
          than by similarity search. Useful when you know the exact ID (e.g., a URL)
          and want to retrieve that vector's data directly.

          Args:
              vector_ids: List of vector IDs to retrieve (e.g., URLs, document IDs).
              include_metadata: Whether to include metadata in results.
              include_embeddings: Whether to include the full embedding vectors in results.

          Returns:
              A list of dictionaries containing information for each found vector.
              Each dict contains:
                  - 'datapoint_id': The ID of the vector.
                  - 'metadata': (optional) The metadata associated with the vector.
                  - 'embedding': (optional) The full embedding vector if include_embeddings=True.

          Raises:
              Exception: If the match client cannot be initialized.
        """
        if not self.match_client:
            try:
                self.match_client = aiplatform_v1beta1.MatchServiceClient(
                    client_options={"api_endpoint": self._get_endpoint_url()}
                )

            except Exception as e:
                raise e

        request = aiplatform_v1beta1.ReadIndexDatapointsRequest(
            index_endpoint=self.endpoint_id,
            deployed_index_id=self.deployment_id,
            ids=vector_ids
        )

        response = self.match_client.read_index_datapoints(request=request)

        print(response)

        results = []
        for datapoint in response.datapoints:
            result = {
                "datapoint_id": datapoint.datapoint_id
            }

            if include_metadata and datapoint.embedding_metadata:
                result["metadata"] = dict(datapoint.embedding_metadata)

            if include_embeddings and datapoint.feature_vector:
                result["embedding"] = list(datapoint.feature_vector)

            results.append(result)

        return results
    
    def get_all_ids(self):
        """Returns all resource identifiers.
        
        Returns:
            A dictionary containing:
                - 'API_ENDPOINT': The endpoint URL for matching operations.
                - 'INDEX_ENDPOINT': The index endpoint resource ID.
                - 'DEPLOYED_INDEX_ID': The deployed index ID.
        """
        return {
            "API_ENDPOINT": self._get_endpoint_url(),
            "INDEX_ENDPOINT": self.endpoint_id,
            "DEPLOYED_INDEX_ID": self.deployment_id
        }
    
    def _get_endpoint_url(self):
        """Retrieves the public endpoint URL.
        
        Returns:
            The public endpoint domain name for the index endpoint.
        """
        return self.endpoint_client.get_index_endpoint(name=self.endpoint_id).public_endpoint_domain_name
    
    def _get_index_id(self):
        """Retrieves the index ID from the endpoint.
        
        Returns:
            The resource name of the first deployed index on the endpoint.
            
        Raises:
            ValueError: If no deployed indexes are found on the endpoint.
        """
        endpoint = self.endpoint_client.get_index_endpoint(name=self.endpoint_id)
        if endpoint.deployed_indexes:
            index_resource_name = endpoint.deployed_indexes[0].index
            return index_resource_name
        else:
            raise ValueError("No deployed indexes found on this endpoint")