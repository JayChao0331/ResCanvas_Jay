"""Search algorithm implementations for ResCanvas.

Provides text and image-based semantic search using CLIP embeddings and Qdrant vector search.
"""
from typing import List, Dict, Any
import logging
import base64
import io
from PIL import Image
import tempfile
import os

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 50

# Lazy imports to avoid startup failures if dependencies aren't installed
_embedding_service = None
_vector_search_service = None


def _get_services():
    """Lazy load embedding and vector search services."""
    global _embedding_service, _vector_search_service
    
    if _embedding_service is None or _vector_search_service is None:
        try:
            from services import embedding_service, vector_search_service
            _embedding_service = embedding_service
            _vector_search_service = vector_search_service
            logger.info("Loaded embedding and vector search services")
        except Exception as e:
            logger.error(f"Failed to load AI services: {e}")
            raise
    
    return _embedding_service, _vector_search_service


def text_search(query: str, rooms: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N, seed: int | None = None) -> List[Dict[str, Any]]:
    """
    Semantic text search using CLIP embeddings.
    
    Args:
        query: Natural language search query (e.g., "rooms with trees")
        rooms: List of candidate room dicts (pre-filtered by visibility)
        top_n: Maximum number of results to return
        seed: Unused (kept for backward compatibility)
    
    Returns:
        List of rooms ranked by semantic similarity with score field added
    """
    try:
        embed_svc, vector_svc = _get_services()
        
        # Generate embedding for the query text
        query_embedding = embed_svc.embed_text([query])  # Returns (1, 512)
        
        if query_embedding is None or query_embedding.size == 0:
            logger.warning("Failed to generate query embedding, falling back to random")
            return _fallback_random_search(rooms, top_n, seed)
        
        # Search vector database for similar canvases
        vector_results = vector_svc.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_n * 2,  # Get extra results to filter by visibility
            score_threshold=0.0
        )
        
        # Create a map of room_id -> score from vector results
        score_map = {str(r['room_id']): r['score'] for r in vector_results}
        
        # Match vector results with provided rooms (visibility-filtered)
        # and add scores
        scored_rooms = []
        for room in rooms:
            room_id = room.get('id') or str(room.get('_id', ''))
            if room_id in score_map:
                room_copy = {**room, 'score': score_map[room_id]}
                scored_rooms.append(room_copy)
            else:
                # Room not in vector DB yet, give low score
                room_copy = {**room, 'score': 0.1}
                scored_rooms.append(room_copy)
        
        # Sort by score descending
        scored_rooms.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Text search for '{query}' returned {len(scored_rooms[:top_n])} results")
        return scored_rooms[:top_n]
        
    except Exception as e:
        logger.exception(f"Text search failed: {e}")
        return _fallback_random_search(rooms, top_n, seed)


def image_search(image_b64: str, rooms: List[Dict[str, Any]], q: str | None = None, top_n: int = DEFAULT_TOP_N, seed: int | None = None) -> List[Dict[str, Any]]:
    """
    Semantic image search using CLIP embeddings.
    
    Args:
        image_b64: Base64-encoded image (without data URI prefix)
        rooms: List of candidate room dicts (pre-filtered by visibility)
        q: Optional text query to combine with image (future enhancement)
        top_n: Maximum number of results to return
        seed: Unused (kept for backward compatibility)
    
    Returns:
        List of rooms ranked by visual similarity with score field added
    """
    try:
        embed_svc, vector_svc = _get_services()
        
        # Decode base64 image and save to temporary file
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Save to temp file (embedding service expects file path)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                image.save(tmp_path, 'PNG')
            
            # Generate embedding for the image
            query_embedding = embed_svc.embed_image(tmp_path)  # Returns (1, 512)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return _fallback_random_search(rooms, top_n, seed)
        
        if query_embedding is None or query_embedding.size == 0:
            logger.warning("Failed to generate image embedding, falling back to random")
            return _fallback_random_search(rooms, top_n, seed)
        
        # Search vector database for similar canvases
        vector_results = vector_svc.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_n * 2,  # Get extra results to filter by visibility
            score_threshold=0.0
        )
        
        # Create a map of room_id -> score from vector results
        score_map = {str(r['room_id']): r['score'] for r in vector_results}
        
        # Match vector results with provided rooms (visibility-filtered)
        scored_rooms = []
        for room in rooms:
            room_id = room.get('id') or str(room.get('_id', ''))
            if room_id in score_map:
                room_copy = {**room, 'score': score_map[room_id]}
                scored_rooms.append(room_copy)
            else:
                # Room not in vector DB yet, give low score
                room_copy = {**room, 'score': 0.1}
                scored_rooms.append(room_copy)
        
        # Sort by score descending
        scored_rooms.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Image search returned {len(scored_rooms[:top_n])} results")
        return scored_rooms[:top_n]
        
    except Exception as e:
        logger.exception(f"Image search failed: {e}")
        return _fallback_random_search(rooms, top_n, seed)


def _fallback_random_search(rooms: List[Dict[str, Any]], top_n: int, seed: int | None = None) -> List[Dict[str, Any]]:
    """Fallback to random ranking if embedding search fails."""
    import random
    logger.warning("Using fallback random search")
    rng = random.Random(seed) if seed is not None else random
    scored = [{**r, "score": rng.random()} for r in rooms]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


# Keep old stub implementations commented for reference
# def text_search(query: str, rooms: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N, seed: int | None = None) -> List[Dict[str, Any]]:
#     """Prototype text search: random scores + ranking."""
#     rng = random.Random(seed) if seed is not None else random
#     scored = [{**r, "score": rng.random()} for r in rooms]
#     scored.sort(key=lambda x: x["score"], reverse=True)
#     return scored[:top_n]

# def image_search(image_b64: str, rooms: List[Dict[str, Any]], q: str | None = None, top_n: int = DEFAULT_TOP_N, seed: int | None = None) -> List[Dict[str, Any]]:
#     """Prototype image search: random scores + ranking."""
#     rng = random.Random(seed) if seed is not None else random
#     scored = [{**r, "score": rng.random()} for r in rooms]
#     scored.sort(key=lambda x: x["score"], reverse=True)
#     return scored[:top_n]

