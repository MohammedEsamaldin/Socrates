# API Reference

All routes are defined in `app.py` and served by Flask on `0.0.0.0:5000` by default.

---

## POST /start_session

Start a new verification session. A session isolates conversation history, verified claims, and the knowledge graph.

**Request**: no body required.

**Response**:
```json
{
  "status": "success",
  "session_id": "session_20250425_120000",
  "message": "New session started successfully"
}
```

**Error** (500):
```json
{
  "status": "error",
  "message": "Failed to start session: <detail>"
}
```

---

## POST /verify_claim

Main endpoint. Accepts text input and an optional image, runs the full verification pipeline, and returns structured results.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_input` | string | yes | Text to verify |
| `session_id` | string | no | Existing session ID. A new session is started automatically if omitted. |
| `image` | file | no | Image file (png/jpg/jpeg/gif/bmp, max 16 MB) for cross-modal verification |

**Response** (200):
```json
{
  "status": "success",
  "result": {
    "session_id": "session_20250425_120000",
    "timestamp": "2025-04-25T12:00:00.000000",
    "original_input": "The Eiffel Tower is in Berlin.",
    "verification_summary": {
      "total_claims": 1,
      "verified_claims": 0,
      "failed_claims": 1,
      "overall_status": "FAIL"
    },
    "socratic_dialogue": [
      {
        "type": "socratic_question",
        "content": "What evidence supports the location of the Eiffel Tower?",
        "reasoning": "Auto-generated for EXTERNAL_KNOWLEDGE_REQUIRED.",
        "confidence": 0.85
      },
      {
        "type": "contradiction_found",
        "content": "Upon investigation, the claim 'The Eiffel Tower is in Berlin.' appears to contradict available evidence.",
        "contradictions": ["Wikipedia: The Eiffel Tower is located in Paris, France"]
      }
    ],
    "detailed_results": [
      {
        "claim": "The Eiffel Tower is in Berlin.",
        "status": "FAIL",
        "confidence": 0.62,
        "evidence": [],
        "contradictions": ["Wikipedia: The Eiffel Tower is located in Paris, France"],
        "clarification_needed": null
      }
    ],
    "next_steps": [
      "Review and clarify contradicted claims"
    ]
  }
}
```

**Error** (400): `user_input` is empty.
**Error** (500): Internal pipeline error.

---

## POST /api/agla_verify

Direct AGLA cross-modal verification endpoint. Bypasses the full claim pipeline and calls the configured remote AGLA service directly.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | yes | Image file |
| `claim` | string | yes | Textual claim to verify against the image |
| `socratic_question` | string | no | Contextual Socratic question for AGLA |
| `use_agla` | bool string | no | Pass `"true"` or `"false"` to override AGLA usage |
| `alpha` | float string | no | AGLA alpha parameter |
| `beta` | float string | no | AGLA beta parameter |
| `return_debug` | bool string | no | Include debug info in response |

**Response** (200):
```json
{
  "status": "success",
  "verdict": true,
  "evidence": [
    "Socratic question: Is the tower visible in the image?",
    "AGLA verdict: True"
  ],
  "latency_ms": 342
}
```

With `return_debug=true`:
```json
{
  "status": "success",
  "verdict": false,
  "evidence": ["AGLA correction: The image shows the Eiffel Tower in Paris.", "AGLA verdict: False"],
  "latency_ms": 512,
  "debug": { ... }
}
```

**Error** (400): Missing `claim` or `image`.
**Error** (503): AGLA API not configured (`AGLA_API_URL` not set).
**Error** (500): AGLA call failed.

---

## GET /session_summary/\<session_id\>

Retrieve a lightweight summary of a session.

**Response** (200):
```json
{
  "status": "success",
  "summary": {
    "session_id": "session_20250425_120000",
    "total_inputs": 3,
    "verified_claims": 5,
    "knowledge_graph_size": 12
  }
}
```

---

## GET /knowledge_graph/\<session_id\>

Export the session's knowledge graph as a serializable dict (entities, relations, claims).

**Response** (200):
```json
{
  "status": "success",
  "knowledge_graph": {
    "entities": [...],
    "relations": [...],
    "claims": [...]
  }
}
```

---

## GET /api/health

Health check.

**Response** (200):
```json
{
  "status": "healthy",
  "timestamp": "2025-04-25T12:00:00.000000",
  "system": "Socrates Agent System"
}
```

---

## GET /

Renders `templates/index.html` — the web UI for interactive verification.

---

## Error Handlers

| HTTP Code | Handler | Description |
|-----------|---------|-------------|
| 413 | `too_large` | Uploaded file exceeds 16 MB |
| 404 | `not_found` | Renders `404.html` |
| 500 | `internal_error` | Renders `500.html` and logs the error |

---

## Authentication

No authentication is implemented. All endpoints are publicly accessible. [inferred: suitable for local/research deployment only; production use would require adding auth middleware.]
