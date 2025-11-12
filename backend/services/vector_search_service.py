"""
Vector search service using Qdrant for semantic canvas search.

This service manages storage and retrieval of canvas embeddings in Qdrant,
enabling semantic similarity search across canvases.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# Global Qdrant client (lazy initialization)
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            _ensure_collection_exists()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    return _qdrant_client


def _ensure_collection_exists():
    """Create collection if it doesn't exist."""
    client = _qdrant_client
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if QDRANT_COLLECTION_NAME not in collection_names:
            logger.info(f"Creating Qdrant collection: {QDRANT_COLLECTION_NAME}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE  # Cosine similarity for normalized embeddings
                )
            )
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} created successfully")
        else:
            logger.debug(f"Collection {QDRANT_COLLECTION_NAME} already exists")
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}")
        raise


def store_canvas_embedding(
    room_id: str,
    embedding: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Store or update a canvas embedding in Qdrant.
    
    Args:
        room_id: Unique identifier for the canvas/room
        embedding: 512-dimensional vector from CLIP model
        metadata: Additional metadata (name, description, type, owner, etc.)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_qdrant_client()
        
        # Ensure embedding is the right shape and type
        if embedding.ndim == 2:
            embedding = embedding.flatten()
        
        if len(embedding) != EMBEDDING_DIMENSION:
            logger.error(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(embedding)}")
            return False
        
        # Prepare payload with metadata
        payload = metadata or {}
        payload['room_id'] = room_id
        
        # Use room_id as the point ID (convert to hash for Qdrant)
        point_id = hash(room_id) & 0x7FFFFFFFFFFFFFFF  # Ensure positive int
        
        # Upsert the point (will update if exists, insert if new)
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            ]
        )
        
        logger.info(f"Stored embedding for room_id={room_id}")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to store embedding for room_id={room_id}: {e}")
        return False


def search_by_embedding(
    query_embedding: np.ndarray,
    top_k: int = 50,
    filters: Optional[Dict[str, Any]] = None,
    score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search for similar canvases using vector similarity.
    
    Args:
        query_embedding: 512-dimensional query vector from CLIP
        top_k: Number of results to return
        filters: Optional filters (e.g., {"type": "public", "owner": "user123"})
        score_threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        List of dicts with keys: room_id, score, and metadata fields
    """
    try:
        client = get_qdrant_client()
        
        # Ensure embedding is the right shape
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.flatten()
        
        if len(query_embedding) != EMBEDDING_DIMENSION:
            logger.error(f"Query embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(query_embedding)}")
            return []
        
        # Build Qdrant filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Perform vector search
        search_result = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for hit in search_result:
            result = {
                'room_id': hit.payload.get('room_id'),
                'score': float(hit.score),
                **hit.payload  # Include all metadata
            }
            results.append(result)
        
        logger.info(f"Vector search returned {len(results)} results (top_k={top_k})")
        return results
        
    except Exception as e:
        logger.exception(f"Vector search failed: {e}")
        return []


def update_canvas_embedding(
    room_id: str,
    new_embedding: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update an existing canvas embedding.
    
    This is just an alias for store_canvas_embedding since upsert handles both.
    """
    return store_canvas_embedding(room_id, new_embedding, metadata)


def delete_canvas_embedding(room_id: str) -> bool:
    """
    Delete a canvas embedding from Qdrant.
    
    Args:
        room_id: Unique identifier for the canvas/room to delete
    
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_qdrant_client()
        point_id = hash(room_id) & 0x7FFFFFFFFFFFFFFF
        
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=[point_id]
        )
        
        logger.info(f"Deleted embedding for room_id={room_id}")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to delete embedding for room_id={room_id}: {e}")
        return False


def batch_store_embeddings(embeddings: List[Dict[str, Any]]) -> int:
    """
    Store multiple embeddings in batch (more efficient).
    
    Args:
        embeddings: List of dicts with keys: room_id, embedding, metadata
    
    Returns:
        Number of successfully stored embeddings
    """
    try:
        client = get_qdrant_client()
        points = []
        
        for item in embeddings:
            room_id = item['room_id']
            embedding = item['embedding']
            metadata = item.get('metadata', {})
            
            # Prepare embedding
            if embedding.ndim == 2:
                embedding = embedding.flatten()
            
            if len(embedding) != EMBEDDING_DIMENSION:
                logger.warning(f"Skipping room_id={room_id} due to dimension mismatch")
                continue
            
            # Prepare payload
            payload = metadata.copy()
            payload['room_id'] = room_id
            
            point_id = hash(room_id) & 0x7FFFFFFFFFFFFFFF
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
        
        if points:
            client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points
            )
            logger.info(f"Batch stored {len(points)} embeddings")
            return len(points)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Batch store failed: {e}")
        return 0


def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the vector collection."""
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        
        return {
            'collection_name': QDRANT_COLLECTION_NAME,
            'vectors_count': collection_info.vectors_count,
            'points_count': collection_info.points_count,
            'status': collection_info.status,
            'config': {
                'dimension': EMBEDDING_DIMENSION,
                'distance': 'COSINE'
            }
        }
    except Exception as e:
        logger.exception(f"Failed to get collection stats: {e}")
        return {'error': str(e)}
