# Next Steps: Complete AI Search Implementation

## What's Done ✅

1. **Core Infrastructure** ✅
   - Qdrant vector database setup
   - `vector_search_service.py` - Complete CRUD operations
   - `search_algorithms.py` - Real semantic search (no more stubs)
   - Docker Compose configuration
   - All dependencies added

2. **Integration** ✅
   - Jay's API endpoint (`search_ai.py`) → Your search logic
   - Jay's frontend UI → Backend API
   - Your embedding service → Vector search service

---

## What's Missing ❌

### Priority 1: Canvas Snapshot Generation (Required for Image Embeddings)

**Problem:** To generate embeddings for canvas content, you need to convert stroke data to images.

**Options:**

#### Option A: Text-Based Embeddings (Simplest - Start Here)
Use canvas name + description for embeddings instead of visual content.

```python
# backend/scripts/populate_embeddings.py
from services.db import rooms_coll
from services.embedding_service import embed_text
from services.vector_search_service import store_canvas_embedding

def populate_text_embeddings():
    """Generate embeddings from room metadata (name + description)."""
    rooms = rooms_coll.find({"archived": {"$ne": True}})
    
    for room in rooms:
        room_id = str(room['_id'])
        name = room.get('name', '')
        desc = room.get('description', '')
        
        # Combine name and description for richer embedding
        text = f"{name}. {desc}" if desc else name
        
        if text.strip():
            embedding = embed_text([text])
            
            store_canvas_embedding(
                room_id=room_id,
                embedding=embedding,
                metadata={
                    'name': name,
                    'description': desc,
                    'type': room.get('type'),
                    'ownerName': room.get('ownerName')
                }
            )
            print(f"✓ Stored embedding for {room_id}: {name}")

if __name__ == "__main__":
    populate_text_embeddings()
```

**Pros:** Works immediately, no canvas rendering needed  
**Cons:** Doesn't capture visual content  
**Use case:** "Find rooms about trees" works, "Find rooms similar to this sketch" won't

#### Option B: Server-Side Canvas Rendering (Better, More Complex)

Render strokes to PNG using Pillow:

```python
# backend/services/canvas_renderer.py
from PIL import Image, ImageDraw
from services.db import strokes_coll

def render_canvas_to_image(room_id: str, width=800, height=600) -> str:
    """Render canvas strokes to PNG file, return path."""
    # Fetch strokes
    strokes = list(strokes_coll.find({"roomId": room_id}).sort("ts", 1))
    
    # Create image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    for stroke in strokes:
        points = stroke.get('points', [])
        color = stroke.get('color', '#000000')
        width = stroke.get('width', 2)
        
        # Draw lines between points
        for i in range(len(points) - 1):
            x1, y1 = points[i]['x'], points[i]['y']
            x2, y2 = points[i+1]['x'], points[i+1]['y']
            draw.line([(x1, y1), (x2, y2)], fill=color, width=int(width))
    
    # Save to temp file
    path = f"/tmp/canvas_{room_id}.png"
    img.save(path)
    return path
```

**Pros:** Captures visual content  
**Cons:** Need to understand stroke data format, coordinate systems  
**Recommendation:** Start with Option A, add this later

#### Option C: Frontend Thumbnail Export (Hybrid Approach)

Let the frontend generate thumbnails and upload them:

1. Add endpoint: `POST /api/v1/rooms/{room_id}/snapshot`
2. Frontend captures canvas as base64 PNG
3. Backend generates embedding and stores it

**Pros:** Frontend already knows how to render  
**Cons:** Requires frontend changes, manual trigger

---

### Priority 2: Background Embedding Worker

**Current State:** Embeddings are NOT auto-generated on canvas create/update

**Solution:** Create a periodic batch processor (simplest approach)

```python
# backend/workers/embedding_worker.py
import time
import logging
from services.db import rooms_coll
from services.embedding_service import embed_text
from services.vector_search_service import store_canvas_embedding, get_collection_stats

logger = logging.getLogger(__name__)

def sync_embeddings_batch():
    """
    Sync embeddings for all canvases that don't have them yet.
    Run this periodically (e.g., every 5 minutes).
    """
    # Get all room IDs in Qdrant
    stats = get_collection_stats()
    existing_count = stats.get('points_count', 0)
    
    # Get all rooms from MongoDB
    rooms = list(rooms_coll.find({"archived": {"$ne": True}}))
    total_rooms = len(rooms)
    
    logger.info(f"Found {total_rooms} rooms, {existing_count} embeddings exist")
    
    new_embeddings = 0
    for room in rooms:
        room_id = str(room['_id'])
        
        # Simple approach: Always regenerate (or add logic to check if exists)
        name = room.get('name', '')
        desc = room.get('description', '')
        text = f"{name}. {desc}" if desc else name
        
        if text.strip():
            embedding = embed_text([text])
            success = store_canvas_embedding(
                room_id=room_id,
                embedding=embedding,
                metadata={
                    'name': name,
                    'description': desc,
                    'type': room.get('type'),
                    'ownerName': room.get('ownerName')
                }
            )
            if success:
                new_embeddings += 1
    
    logger.info(f"Synced {new_embeddings} new embeddings")
    return new_embeddings

def run_worker(interval_seconds=300):
    """Run worker in loop."""
    logger.info(f"Starting embedding worker (interval={interval_seconds}s)")
    while True:
        try:
            sync_embeddings_batch()
        except Exception as e:
            logger.exception(f"Worker error: {e}")
        
        time.sleep(interval_seconds)

if __name__ == "__main__":
    run_worker()
```

**How to run:**
```bash
# In separate terminal
python backend/workers/embedding_worker.py

# Or add to supervisor/systemd/docker-compose
```

**Alternative (Production):** Use Celery for more robust job scheduling

---

### Priority 3: Hook into Canvas Updates

Trigger embedding regeneration when canvases change:

```python
# In backend/routes/rooms.py (after canvas update)

from services.embedding_service import embed_text
from services.vector_search_service import store_canvas_embedding

@rooms_bp.route('/api/v1/rooms/<room_id>', methods=['PATCH'])
@require_auth
def update_room(room_id):
    # ... existing update logic ...
    
    # After successful update, regenerate embedding
    try:
        name = updated_room.get('name', '')
        desc = updated_room.get('description', '')
        text = f"{name}. {desc}" if desc else name
        
        if text.strip():
            embedding = embed_text([text])
            store_canvas_embedding(
                room_id=room_id,
                embedding=embedding,
                metadata={
                    'name': name,
                    'description': desc,
                    'type': updated_room.get('type'),
                    'ownerName': updated_room.get('ownerName')
                }
            )
    except Exception as e:
        logger.warning(f"Failed to update embedding for {room_id}: {e}")
    
    return jsonify(updated_room)
```

---

### Priority 4: Database Indexes (Performance)

Add indexes for faster queries:

```python
# In backend/services/db.py (add to existing indexes)

# For search filtering
rooms_coll.create_index([("type", 1), ("archived", 1)])
rooms_coll.create_index([("ownerId", 1), ("archived", 1)])
```

---

## Recommended Implementation Order

### Week 1: Get It Working
1. ✅ Setup Qdrant (Done!)
2. ✅ Implement vector_search_service.py (Done!)
3. ✅ Update search_algorithms.py (Done!)
4. ⏳ **Create `populate_embeddings.py` script** (Option A - Text-based)
5. ⏳ **Test search from UI**

### Week 2: Automate
6. ⏳ Create `embedding_worker.py` (periodic batch sync)
7. ⏳ Add hooks to `rooms.py` for real-time updates
8. ⏳ Add canvas deletion → embedding cleanup

### Week 3: Visual Search
9. ⏳ Implement canvas rendering (Option B or C)
10. ⏳ Update embeddings to use visual content
11. ⏳ Test image-based search

### Week 4: Polish
12. ⏳ Add monitoring/logging
13. ⏳ Performance tuning
14. ⏳ Error handling improvements

---

## Quick Test Script

Save as `backend/scripts/test_vector_search.py`:

```python
#!/usr/bin/env python3
"""Quick test script for vector search functionality."""

from services.embedding_service import embed_text, embed_image
from services.vector_search_service import (
    store_canvas_embedding, 
    search_by_embedding,
    get_collection_stats
)
import numpy as np

def test_basic_flow():
    print("🧪 Testing Vector Search...")
    
    # 1. Store test embeddings
    test_data = [
        ("room1", "A beautiful landscape with mountains and trees"),
        ("room2", "Abstract geometric shapes in bright colors"),
        ("room3", "Portrait of a person with blue eyes"),
        ("room4", "Forest scene with tall pine trees"),
    ]
    
    print("\n📝 Storing test embeddings...")
    for room_id, description in test_data:
        emb = embed_text([description])
        store_canvas_embedding(room_id, emb, {"description": description})
        print(f"  ✓ {room_id}: {description[:50]}...")
    
    # 2. Check stats
    print("\n📊 Collection stats:")
    stats = get_collection_stats()
    print(f"  Points: {stats.get('points_count')}")
    print(f"  Dimension: {stats.get('config', {}).get('dimension')}")
    
    # 3. Search
    print("\n🔍 Searching for 'trees'...")
    query_emb = embed_text(["trees"])
    results = search_by_embedding(query_emb, top_k=5)
    
    print(f"\n  Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r['room_id']} (score: {r['score']:.3f})")
        print(f"       {r.get('description', '')[:60]}...")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_basic_flow()
```

Run with:
```bash
cd backend
python scripts/test_vector_search.py
```

---

## Summary

**You have:** Complete Qdrant integration, working vector search, connected UI  
**You need:** Populate embeddings (start with text-based), then add automation

**Fastest path to demo:**
1. Run the test script above
2. Create `populate_embeddings.py` for real rooms
3. Test search in the UI
4. Show Jay it works! 🎉
