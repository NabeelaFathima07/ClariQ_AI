# Truth Layer — a self-auditing company knowledge base

Upload internal documents; employees ask questions. Unlike "chat with your docs", the system extracts **claims**,
detects **contradictions**, ranks sources by **authority**, tracks **freshness**, answers **honestly** (with sources,
confidence, and "I don't know"), routes gaps to **owners**, and learns from their answers.

## Quick start (no API keys needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/seed_demo.py           # demo company: users, docs, contradictions, versions, a restricted doc
streamlit run streamlit_app.py        # ClariQ Streamlit App: open http://localhost:8501
uvicorn app.api:app --reload          # REST API & static web UI: open http://localhost:8000
```
Or: `cp .env.example .env && docker compose up --build` (demo data is loaded automatically).

Tests need only the standard library + `pypdf`, `python-docx`, `PyJWT`:
```bash
python -m unittest discover -s tests -v     # 18 end-to-end tests
python eval/run_eval.py                     # golden-question quality gate (accuracy / citation / false-answer rate)
```

## Try these in the UI
| Sign in as | Ask | What you should see |
|---|---|---|
| bob (US employee) | *How many vacation days do I get?* | **18 days**, and a note that the handbook (15) is overruled by the signed HR policy (authority 100 vs 80) |
| asha (India employee) | same question | **21 days** (region-aware) |
| bob | *What is the meal allowance when travelling?* | "Sources disagree" — $50 vs $75, equal authority, **unresolved**, routed to an owner |
| bob | *How long do customers have to request a refund?* then set **As of** = 2025-06-01 | 30 days now, **14 days** as of 2025 (versioned claims) |
| bob | *How long does VPN access take?* | answer with a **stale** warning (last verified 400 days ago), lower confidence |
| bob / carol | *What is the CEO bonus target?* | bob: "I don't know" (cannot see it). carol (finance group): 40 percent |
| bob | *How do I book a conference room?* | "I don't know" + routed to owner. Sign in as **hr**, open *Owner inbox*, answer it; ask again as bob → instant, **expert-verified** answer |
| admin | *Conflicts*, *Knowledge gaps*, *Review queue*, *Dashboard*, *Audit log* | manage the knowledge base |

## Feature → code map
| Feature | Where |
|---|---|
| Upload PDF/DOCX/HTML/CSV/TXT (+ OCR for scans) | `app/parsing.py`, `POST /api/documents` |
| Claim-level knowledge (source, owner, dates, scope) | `app/extraction.py` (Claude JSON + verbatim-passage guard; heuristic fallback), table `claims` |
| Contradiction detection | `app/reasoning.py` (`detect_conflicts`, `judge_conflict`) — only compares semantically close claims |
| Source-authority ranking | `tenants.authority_map`, `reasoning.auto_winner`, `PUT /api/tenant` |
| Honest answers: sources, confidence, "I don't know" | `app/qa.py` |
| Freshness decay + review queue + reminders | `reasoning.freshness`, `app/reviews.py`, `app/jobs.py` |
| "Ask the owner" loop → verified claim | `app/owner_loop.py`, Owner inbox, Slack `/tl-answer` |
| Knowledge-gap dashboard (clustered) + doc tasks | `app/gaps.py` |
| Time travel ("as of" date) + versioning | `app/ingest.py` (same title = new version), `retrieval._time_ok` |
| Role / region awareness | `retrieval.scope_multiplier` (soft scoping) |
| Permissions at retrieval time | `retrieval.acl_ok` — filtered *before* scoring; ACLs copied to claims/chunks |
| SSO (OIDC/JWKS), role mapping | `app/auth.py` |
| PII redaction before LLM calls | `app/pii.py` |
| Audit log of every question/source/action | table `audit_log`, `GET /api/audit` |
| Slack bot | `app/slack_bot.py` |
| Streaming answers (SSE) | `POST /api/ask/stream` |
| Evaluation / regression gate | `eval/`, `tests/` |

## How answers are scored
`confidence = 0.45·relevance + 0.20·authority + 0.20·freshness + 0.15·agreement (+0.10 if human-verified) − conflict penalty`.
Below `MIN_RELEVANCE` the system refuses and routes to an owner; below `LOW_CONFIDENCE` it answers with a warning *and* flags the owner.

## Configuration
See `.env.example`. With no keys everything runs offline (heuristic extraction, hashing embeddings, templated answers).
Add `ANTHROPIC_API_KEY` for LLM claim extraction, conflict judging and fluent answers; add `EMBED_PROVIDER=openai|voyage`
(then `python scripts/reembed.py`) for real semantic search.

## Honest limitations (read before production)
* **Offline mode is a demo.** The hashing embedder and regex extractor handle clear, numeric policy statements; synonyms
  and nuanced prose need LLM extraction + semantic embeddings. Run `eval/run_eval.py` on *your* documents.
* **Storage is SQLite** via one module (`app/db.py`). Retrieval does a linear scan, fine for tens of thousands of claims.
  For scale, reimplement `db.py` on PostgreSQL + pgvector (HNSW index; push the ACL filter into SQL with
  `allowed_groups && :user_groups`) and move keyword search to Postgres FTS or OpenSearch.
* **Background work** uses FastAPI `BackgroundTasks`. Use Celery or Temporal for retries and long pipelines
  (`ingest.process_document` is already idempotent-friendly and has a single entry point).
* **Dev login is for demos only.** Set `AUTH_MODE=oidc` and a strong `JWT_SECRET` in production; serve over TLS.
* PII redaction is regex-based; swap in Microsoft Presidio for production-grade coverage.
* Not included: Teams bot, per-tenant KMS keys, Next.js frontend (the bundled UI is one static file),
  connector sync for Drive/Confluence/SharePoint (add a connector that calls `create_document` + `process_document`).
* The FastAPI app and Slack bot are thin layers over the tested core; they were exercised in the author's environment
  through stubs rather than a live server, so run the app once and the Slack bot against a test workspace before rollout.
