# google-vertex-vector-utilities

Vector search utilities for Google Cloud Vertex AI Vector Search service.

## Installation

### From GitHub (with uv)

```bash
uv add git+https://github.com/lcpritchard1/google-vertex-vector-utilities.git
```

### Local Development

```bash
git clone https://github.com/lcpritchard1/google-vertex-vector-utilities.git
cd google-vertex-vector-utilities
uv sync
```

## Usage

```python
from google_vertex_vector_utilities import VertexVectorHandler, ShardSize

# Initialize the handler
handler = VertexVectorHandler(
    google_cloud_project='your-project-id',
    location='us-central1'
)

# Load an existing index and endpoint
handler.load(
    index_id='projects/123/locations/us-central1/indexes/456',
    endpoint_id='projects/123/locations/us-central1/indexEndpoints/789',
    deployed_index_id='deployed_index_name'
)

# Query the index
results = handler.query_index(
    embedding=[0.1, 0.2, 0.3, ...],  # Your embedding vector
    num_results=10,
    include_metadata=True
)

# Insert a vector
handler.insert_vector(
    vector_id='unique-id',
    vector_embedding=[0.1, 0.2, 0.3, ...],
    vector_metadata={'key': 'value'}
)
```

### Creating New Resources

```python
# Create a new index
index_id = handler.create_index(
    index_display_name='my-vector-index',
    index_dimension_count=1536,
    shard_size='SMALL'
)

# Create a new endpoint
endpoint_id = handler.create_endpoint(
    endpoint_display_name='my-endpoint'
)

# Deploy the index to the endpoint
deployment_id = handler.deploy_index(
    display_name='my-deployment',
    machine_type='e2-standard-2',
    min_replica_count=1,
    max_replica_count=1
)
```

## Requirements

- Python ≥ 3.12
- Google Cloud project with Vertex AI API enabled
- Appropriate GCP credentials configured
