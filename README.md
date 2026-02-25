# NeuroHeart Chat API

FastAPI microservice for multi-user chat sessions with HRV context, Qdrant RAG, and OpenAI.

Runs on port **8003**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your credentials
```

## Database migration

Run once against your PostgreSQL instance:

```bash
psql -U neuroheart_user -d neuroheart -h localhost -f migrations/001_chat_tables.sql
```

## Run locally

```bash
uvicorn app.main:app --reload --port 8003
```

## Quick test

```bash
# Health
curl http://127.0.0.1:8003/health

# Create conversation
curl -X POST http://127.0.0.1:8003/v1/chat/conversations \
  -H "x-app-token: nch-secure-token-2024" \
  -H "Content-Type: application/json" \
  -d '{"user_uid":"001992.2a11e991f8d249a886acf50cec6920eb.0239","title":"Test"}'

# Send message (replace CONV_UUID with value from above)
curl -X POST http://127.0.0.1:8003/v1/chat \
  -H "x-app-token: nch-secure-token-2024" \
  -H "Content-Type: application/json" \
  -d '{"user_uid":"001992.2a11e991f8d249a886acf50cec6920eb.0239","conversation_id":"CONV_UUID","message":"Why was my HRV low yesterday?","hrv_range":"7d"}'

# List conversations
curl "http://127.0.0.1:8003/v1/chat/conversations?user_uid=001992.2a11e991f8d249a886acf50cec6920eb.0239" \
  -H "x-app-token: nch-secure-token-2024"

# Fetch history
curl "http://127.0.0.1:8003/v1/chat/history?user_uid=001992.2a11e991f8d249a886acf50cec6920eb.0239&conversation_id=CONV_UUID" \
  -H "x-app-token: nch-secure-token-2024"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /v1/chat/conversations | Create conversation |
| GET | /v1/chat/conversations | List conversations |
| GET | /v1/chat/history | Fetch message history |
| POST | /v1/chat | Send message |

All `/v1/chat/*` endpoints require header `x-app-token`.

## Server deploy (VPS)

```bash
git clone <repo> /opt/neuroheart/chat-api
cd /opt/neuroheart/chat-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# copy .env with production values

# Run with systemd (bind to 127.0.0.1:8003)
# Nginx: proxy /api/chat/ → http://127.0.0.1:8003/
```
