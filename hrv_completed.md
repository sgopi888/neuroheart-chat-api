ssh root@159.198.44.98

system → summary → HRV daily → HRV aggregates → RAG → history → user msg
— 
Your iOS integration contract is:
Endpoints the app will use:
* POST /api/chat/v1/chat/conversations → create thread
* GET /api/chat/v1/chat/conversations → recent chats
* GET /api/chat/v1/chat/history → load messages
* POST /api/chat/v1/chat → send message

✅ What is working now (production-verified)
Ingestion → DB
Apple Watch → HealthKit → ingest API → PostgreSQL✔ heart_rate✔ hrv✔ steps✔ sleep✔ /v1/latest cursor
HRV analytics
PostgreSQL → HRV API → features + patterns✔ RMSSD, SDNN, LF/HF, mean_hr✔ range aggregation (1d, 7d, 30d, 6m)✔ TZ-aware✔ API-key protected
Chat service
Public → Nginx → chat API → HRV API + Qdrant + OpenAI✔ conversation threads✔ history storage✔ ownership enforcement (404 cross-user)✔ app token (403 invalid)✔ rate limiting✔ DB persistence✔ HRV context used (used_context: true)✔ RAG top-k = 3✔ external internet path working
iOS contract ready
✔ create conversation✔ send message✔ fetch history✔ list conversations

🧠 What context is sent to the LLM right now
From your deployed chat_service.py behavior:
1️⃣ SYSTEM prompt
Static instruction:
* health insights assistant
* use HRV when present
* concise output

2️⃣ HRV_CONTEXT (JSON)
Included when HRV API returns data.
Currently contains (compact):


{
  "summary_metrics": {...},
  "time_series": [...],
  "patterns": {...}
}

In your test response:
LLM used:
* RMSSD = 9.88→ proves summary_metrics is being injected.
⚠️ Right now you are sending the full HRV API JSON truncated by character limit(not yet the 14-day daily matrix + 90-day aggregates rule).
So context currently = raw HRV analysis output (trimmed).

3️⃣ RAG_CONTEXT
From Qdrant:
* top_k = 3
* each chunk truncated to ~400 chars
* filtered by:
    * type = knowledge
    * OR type = memory + user_uid
This is working (rag_k: 3 returned).

4️⃣ CHAT_HISTORY window
Currently:
* last ~8 turns (from history[-8:])
* stored in Postgres
* injected verbatim
No summarization yet.

5️⃣ USER message
Current user query appended last.

📦 Current prompt structure (actual order)


SYSTEM

HRV_CONTEXT (JSON, truncated)

RAG_CONTEXT (3 snippets)

CHAT_HISTORY (last ~8 messages)

USER MESSAGE


More: phase 2: What is implemented
New file: app/token_budget.py
* count_tokens(text) — tiktoken cl100k_base token counting
* count_messages(messages) — token count across a message list (with role overhead)
* trim_text_to_tokens(text, max_tok) — hard-truncate text to N tokens
Updated: app/hrv_client.py
* Replaces raw {summary_metrics, time_series, patterns} with structured shape:
    * daily_14d — last 14 rows, only {date, rmssd, sdnn, mean_hr, lf_hf_ratio} (no extras)
    * hrv_90d / hr_90d / sleep_90d / steps_90d — scalar aggregates only, no arrays
Updated: app/chat_service.py
* Real tiktoken budget enforcement (MAX_TOKENS = 100k)
* Priority order: system → summary → HRV daily → HRV aggregates → RAG → history → user msg
* RAG trimmed per-chunk to 300 tokens; capped at 20 chunks
* Reduction ladder: RAG 20→10→5→0, then history 20→12→8→4→0
* Summarization also triggers when older messages exceed 60k tokens (not just message count)
* Per-request log: chat tokens — total=X system=X summary=X history=X hrv=X rag=X latency_ms=X
Updated: app/main.py — logging.basicConfig so app logs appear in the log file




— 

tests: 

Here's where we are:
What's done:
* All 13 source files written (app/, migrations/, requirements.txt, .env, etc.)
* DB migration ran successfully on the VPS — all 3 tables created (conversations, chat_messages, conversation_summaries)
* All Python packages installed on the VPS
* Server started, health check passed ({"ok":true})
* POST /v1/chat/conversations — working (returned UUID)
* GET /v1/chat/conversations — working
What just failed:
* POST /v1/chat hit a bug: the SQL for inserting messages used :meta::jsonb which PostgreSQL rejects when mixed with SQLAlchemy named params. Fixed to CAST(:meta AS jsonb). The fix is synced to the VPS but the server needs a restart to pick it up.
Right now:
Trying to restart the server on the VPS so I can test the full chat endpoint end-to-end.

curl http://127.0.0.1:8003/health

user_id=001992.2a11e991f8d249a886acf50cec6920eb.0239

grep APP_TOKEN /opt/neuroheart/chat-api/.env
APP_TOKEN=nch-secure-token-2024
root@data-neuroheart:~#


Worked: 
curl -X POST http://127.0.0.1:8003/v1/chat/conversations \
  -H "Content-Type: application/json" \
  -H "x-app-token: nch-secure-token-2024" \
  -d '{
    "user_uid":"001992.2a11e991f8d249a886acf50cec6920eb.0239",
    "title":"Test Thread"
  }'

Query Database in above server:
psql -U neuroheart_user -d neuroheart -h localhost
StrongPassword123!

Worked:
root@data-neuroheart:~# curl -X POST http://127.0.0.1:8003/v1/chat \
  -H "Content-Type: application/json" \
  -H "x-app-token: nch-secure-token-2024" \
  -d '{
    "user_uid":"001992.2a11e991f8d249a886acf50cec6920eb.0239",
    "conversation_id":"0d62db3f-226f-4cc9-ac5e-fdf5bc2c5322",
    "message":"Why was my HRV low yesterday?",
    "hrv_range":"7d"
  }'
{"conversation_id":"0d62db3f-226f-4cc9-ac5e-fdf5bc2c5322","reply":"Your HRV was lower yesterday, with an RMSSD of 9.88, which could indicate elevated stress levels or fatigue. Factors such as insufficient sleep, poor hydration, emotional stress, or increased physical exertion can contribute to lower HRV. Consider focusing on recovery today—prioritize rest, stay hydrated, and engage in calming activities like deep breathing or gentle exercise. Monitoring your HRV trends can help identify specific triggers.","used_context":true,"hrv_range":"7d","rag_k":3}root@data-neuroheart:~# 


 curl "http://127.0.0.1:8003/v1/chat/history?user_uid=001992.2a11e991f8d249a886acf50cec6920eb.0239&conversation_id=0d62db3f-226f-4cc9-ac5e-fdf5bc2c5322&limit=10" \
  -H "x-app-token: nch-secure-token-2024"
{"conversation_id":"0d62db3f-226f-4cc9-ac5e-fdf5bc2c5322","messages":[{"id":9,"role":"user","content":"Why was my HRV low yesterday?","created_at":"2026-02-25 02:25:37.569638+00"},{"id":10,"role":"assistant","content":"Your HRV was lower yesterday, with an RMSSD of 9.88, which could indicate elevated stress levels or fatigue. Factors such as insufficient sleep, poor hydration, emotional stress, or increased physical exertion can contribute to lower HRV. Consider focusing on recovery today—prioritize rest, stay hydrated, and engage in calming activities like deep breathing or gentle exercise. Monitoring your HRV trends can help identify specific triggers.","created_at":"2026-02-25 02:25:40.005473+00"}]}root@data-neuroheart:~# 



 curl -X POST https://api.neuroheart.ai/api/chat/v1/chat \
  -H "Content-Type: application/json" \
  -H "x-app-token: nch-secure-token-2024" \
  -d '{
    "user_uid":"001992.2a11e991f8d249a886acf50cec6920eb.0239",
    "conversation_id":"0d62db3f-226f-4cc9-ac5e-fdf5bc2c5322",
    "message":"Give me a 1 sentence recovery summary for this week",
    "hrv_range":"7d"
  }'
{"conversation_id":"0d62db3f-226f-4cc9-ac5e-fdf5bc2c5322","reply":"This week, prioritize hydration, quality sleep, and stress management techniques to enhance your HRV and overall recovery.","used_context":true,"hrv_range":"7d","rag_k":3}%                                                                     sreekanthgopi@Sreekanths-MacBook-Pro neuroheart-chat-api % 

Phase 2: worked tests:
From Mac: 
# Health
curl http://127.0.0.1:8003/health

From server: logging and then: ssh root@159.198.44.98
# Create conversation
curl -X POST http://127.0.0.1:8003/v1/chat/conversations \
  -H "x-app-token: nch-secure-token-2024" \
  -H "Content-Type: application/json" \
  -d '{"user_uid":"001992.2a11e991f8d249a886acf50cec6920eb.0239","title":"Test"}'

# Send chat (replace CONV_UUID)
curl -X POST http://127.0.0.1:8003/v1/chat \
  -H "x-app-token: nch-secure-token-2024" \
  -H "Content-Type: application/json" \
  -d '{"user_uid":"001992.2a11e991f8d249a886acf50cec6920eb.0239","conversation_id":"CONV_UUID","message":"Why was my HRV low?","hrv_range":"7d"}'

# Check token log on VPS
grep 'chat tokens' /var/log/neuroheart-chat.log | tail -5


From Mac terminal:sreekanthgopi@Sreekanths-MacBook-Pro neuroheart-chat-api % python -c "from app.token_budget import count_tokens; print(count_tokens('hello world'))"

2
sreekanthgopi@Sreekanths-MacBook-Pro neuroheart-chat-api % python -c "from app.chat_service import _build_prompt; msgs, bd = _build_prompt('', [], {}, [], 'test'); print('budget check:', bd)"
budget check: {'tokens_system': 46, 'tokens_summary': 0, 'tokens_history': 0, 'tokens_hrv': 0, 'tokens_rag': 0, 'tokens_total': 51}
sreekanthgopi@Sreekanths-MacBook-Pro neuroheart-chat-api % 

