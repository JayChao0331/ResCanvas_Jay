"""Simple search algorithm stubs for ResCanvas.

These are intentionally naive implementations used by the website prototype.
They accept a list of room dicts and return them with a uniform score so the
frontend can exercise the UI without any model or vector DB dependencies.
"""
from typing import List, Dict, Any
import random

DEFAULT_TOP_N = 50



def text_search(query: str, rooms: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N, seed: int | None = None) -> List[Dict[str, Any]]:
    """Prototype text search: random scores + ranking."""
    rng = random.Random(seed) if seed is not None else random
    scored = [{**r, "score": rng.random()} for r in rooms]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

def image_search(image_b64: str, rooms: List[Dict[str, Any]], q: str | None = None, top_n: int = DEFAULT_TOP_N, seed: int | None = None) -> List[Dict[str, Any]]:
    """Prototype image search: random scores + ranking."""
    rng = random.Random(seed) if seed is not None else random
    scored = [{**r, "score": rng.random()} for r in rooms]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]



# def text_search(query: str, rooms: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N) -> List[Dict[str, Any]]:
#     """Naive text search stub.

#     Returns the input rooms unchanged except for attaching a default score and
#     a snippet field (from description) so the frontend can render results.
#     """
#     out = []
#     for r in rooms[:top_n]:
#         out.append({
#             **r,
#             'score': 1.0,
#             'snippet': (r.get('description') or '')[:300]
#         })
#     return out


# def image_search(image_b64: str, rooms: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N) -> List[Dict[str, Any]]:
#     """Naive image search stub.

#     Currently ignores the image and returns input rooms with default score.
#     """
#     out = []
#     for r in rooms[:top_n]:
#         out.append({
#             **r,
#             'score': 1.0,
#             'snippet': (r.get('description') or '')[:300]
#         })
#     return out
