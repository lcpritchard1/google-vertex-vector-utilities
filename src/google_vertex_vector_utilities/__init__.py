"""Google Vertex Vector Utilities - Vector search utilities for Google Cloud Vertex AI."""

from google_vertex_vector_utilities.vector_utils import (
    VertexVectorHandler,
    ShardSize,
)

from google_vertex_vector_utilities.async_vector_utils import AsyncVectorHandler

__version__ = "0.2.0"

__all__ = ["VertexVectorHandler", "AsyncVectorHandler", "ShardSize"]
