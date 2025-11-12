# Vector Search Setup Guide

## ✅ Completed Integration

The vector search functionality has been integrated with Jay's search UI. Here's what was implemented:

### Files Modified/Created:
1. ✅ `backend/requirements.txt` - Added AI/ML dependencies
2. ✅ `backend/config.py` - Added Qdrant configuration
3. ✅ `backend/services/vector_search_service.py` - **Implemented complete Qdrant integration**
4. ✅ `backend/services/search_algorithms.py` - **Replaced stubs with real semantic search**
5. ✅ `backend/docker-compose.yml` - Added Qdrant service

### Your Files (Already Complete):
- ✅ `backend/services/embedding_service.py` - CLIP embeddings (text + image)

### Jay's Files (Already Complete):
- ✅ `backend/routes/search_ai.py` - API endpoint
- ✅ `frontend/src/components/Search/AISearchPanel.jsx` - Search UI
- ✅ `frontend/src/components/Search/VisualSearchUpload.jsx` - Image upload UI
- ✅ `frontend/src/pages/Dashboard.jsx` - Integrated search panel

---

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
cd backend

# Start Qdrant and Redis
docker-compose up -d qdrant redis

# Verify Qdrant is running
curl http://localhost:6333/healthz
# Should return: {"title":"healthz","version":"1.x.x"}

# Install Python dependencies (if not already done)
pip install -r requirements.txt

# Run backend
python app.py
```

### Option 2: Local Qdrant Installation

```bash
# macOS with Homebrew
brew install qdrant

# Or using Docker standalone
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Run backend
python app.py
```

---

## 🔧 Configuration

The following environment variables can be set in `.env`:

```bash
# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_COLLECTION_NAME=rescanvas_embeddings

# These are already in your config:
# REDIS_HOST=localhost
# REDIS_PORT=6379
# MONGO_ATLAS_URI=...
```

---

## 📊 How It Works

### 1. **Search Flow (Already Wired Up):**
```
User enters query → AISearchPanel.jsx
    ↓
POST /api/v1/search/ai → search_ai.py
    ↓
Filters rooms by visibility → search_algorithms.py
    ↓
text_search() or image_search()
    ↓
embedding_service.embed_text() or .embed_image()
    ↓
vector_search_service.search_by_embedding()
    ↓
Qdrant returns similar canvases
    ↓
Results ranked by similarity score
    ↓
UI displays ranked results
```

### 2. **Vector Search Service Functions:**

```python
# Store canvas embedding
vector_search_service.store_canvas_embedding(
    room_id="123abc",
    embedding=np.array([...512 dims...]),
    metadata={"name": "My Canvas", "type": "public", "owner": "user123"}
)

# Search by text
query_emb = embedding_service.embed_text(["rooms with trees"])
results = vector_search_service.search_by_embedding(query_emb, top_k=50)

# Search by image
img_emb = embedding_service.embed_image("path/to/image.png")
results = vector_search_service.search_by_embedding(img_emb, top_k=50)

# Delete embedding
vector_search_service.delete_canvas_embedding(room_id="123abc")
```

---

## 🧪 Testing the Integration

### 1. Start Services:
```bash
# Terminal 1: Start Qdrant
cd backend
docker-compose up qdrant

# Terminal 2: Start Backend
python app.py
```

### 2. Test Qdrant Connection:
```bash
# Check collection stats
curl http://localhost:6333/collections

# Should show empty collection or newly created one
```

### 3. Test Search API (currently will work but return empty results):
```bash
# Text search
curl -X POST http://localhost:10010/api/v1/search/ai \
  -H "Content-Type: application/json" \
  -d '{"q": "rooms with trees"}'

# Should return: {"status": "ok", "results": []}
# (Empty because no embeddings stored yet)
```

### 4. Test from Frontend:
1. Open Dashboard (http://localhost:3000)
2. Find the "AI Search" panel
3. Type a query: "rooms with trees"
4. Click "Search"
5. Should see results (or empty if no embeddings yet)

---

## ⚠️ What's Still Missing

### 1. **Canvas Snapshot Generation** (Not Yet Implemented)
To search by canvas content, you need to:
- Convert canvas strokes to images
- Generate embeddings for those images
- Store them in Qdrant

**Simple approach for testing:**
```python
# Create a manual embedding for a test canvas
from services.embedding_service import embed_text
from services.vector_search_service import store_canvas_embedding

# Text-based embedding (easiest for testing)
room_id = "YOUR_ROOM_ID_HERE"
description = "A beautiful landscape with trees and mountains"
embedding = embed_text([description])

store_canvas_embedding(
    room_id=room_id,
    embedding=embedding,
    metadata={
        "name": "Test Canvas",
        "description": description,
        "type": "public",
        "ownerName": "testuser"
    }
)
```

### 2. **Background Worker** (`embedding_worker.py`) - Not Yet Implemented
For automatic embedding generation, you'll need:
- Worker to listen for canvas create/update events
- Canvas-to-image rendering logic
- Batch processing for existing canvases

**Manual/Periodic approach (simplest):**
- Create a script to batch-process canvases
- Run it manually or via cron job
- See `NEXT_STEPS.md` for implementation details

---

## 🔍 Debugging

### Check if Qdrant is running:
```bash
curl http://localhost:6333/healthz
```

### Check collection stats:
```bash
curl http://localhost:6333/collections/rescanvas_embeddings
```

### View logs:
```bash
# Backend logs will show:
# "Connected to Qdrant at localhost:6333"
# "Collection rescanvas_embeddings created successfully"
# "Vector search returned X results (top_k=50)"
```

### Common issues:
1. **"Connection refused"** - Qdrant not running
   - Solution: `docker-compose up qdrant`

2. **"Import could not be resolved"** - Dependencies not installed
   - Solution: `pip install -r requirements.txt`

3. **"No module named 'torch'"** - PyTorch not installed
   - Solution: `pip install torch` (may take time, ~2GB download)

4. **Empty search results** - No embeddings stored yet
   - Solution: Manually store test embeddings (see above)

---

## 📈 Next Steps (Future Enhancements)

See `NEXT_STEPS.md` for:
- Implementing `embedding_worker.py`
- Canvas-to-image rendering
- Batch processing existing canvases
- Performance optimization
- Monitoring and observability

---

## 🎉 What's Working Now

✅ Qdrant integration complete  
✅ Vector search service implemented  
✅ Search algorithms use real embeddings  
✅ API endpoint ready  
✅ Frontend UI connected  
✅ Docker setup included  

**Ready to test!** Just need to:
1. Start Qdrant
2. Install dependencies
3. Populate some test embeddings
4. Try searching from the UI
