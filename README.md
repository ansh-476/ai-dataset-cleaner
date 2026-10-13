# AI Dataset Cleaner

Upload a dataset → a rule engine profiles it deeply → an AI model reasons about the findings and proposes a cleaning plan →
you review it → it's applied → download the cleaned file and a report.

Runs **fully online for free** (GitHub Pages + Render + Neon + Groq) or **fully local** (Ollama + SQLite). Same code.

```
Browser (GitHub Pages)  ──►  FastAPI backend (Render)  ──►  Postgres (Neon)
        frontend/                     app/                     datasets, plans, history
                                        │
                                        └──►  AI provider (Groq / Gemini / OpenRouter / OpenAI / Anthropic / Ollama)
```

---

## A. Deploy online (free) — step by step

You need accounts on GitHub, Render, Neon and Groq (all free, no credit card).

### 1. Put the code on GitHub
1. Create a new repository on github.com (e.g. `ai-dataset-cleaner`), public or private.
2. Upload this whole folder (drag & drop in the browser, or `git init` / `git add .` / `git commit` / `git push`).
   Do **not** upload a `.env` file with real keys — `.gitignore` already excludes it.

### 2. Get a free AI key (Groq)
1. Go to https://console.groq.com → API Keys → *Create API key*. Copy it (starts with `gsk_`).
   Free tier is enough for a project: Llama 3.3 70B, ~1 000 requests/day, ~100 000 tokens/day.
   (If you hit the daily limit, switch `LLM_MODEL` to `llama-3.1-8b-instant`, which has a much higher limit.)

   Alternatives: **Gemini** (https://aistudio.google.com → Get API key, set `LLM_PROVIDER=gemini`) or
   **OpenRouter** free models (`LLM_PROVIDER=openrouter`).

### 3. Create a free Postgres database (Neon)
1. https://neon.tech → New project (any name/region).
2. On the project dashboard click *Connect* → copy the connection string. It looks like
   `postgresql://neondb_owner:xxxx@ep-xxxx.aws.neon.tech/neondb?sslmode=require`.

### 4. Deploy the backend (Render)
1. https://render.com → *New +* → **Blueprint** → connect your GitHub repo. Render reads `render.yaml`.
2. It asks for the secret variables:
   * `LLM_API_KEY` = your Groq key
   * `DATABASE_URL` = the Neon connection string
   * `ALLOWED_ORIGINS` = `https://YOUR-USERNAME.github.io` (your GitHub Pages origin; you can change it later)
3. Click *Apply*. After 2–3 minutes your API is live at `https://ai-dataset-cleaner-api.onrender.com` (name may differ).
   Open `https://…onrender.com/api/health` — you should see `"reachable": true` under `llm`.
   The backend also serves the UI at its own URL, so you can already use the app from there.

   > Free Render services sleep after 15 min idle; the first request then takes ~30–60 s (the UI shows "Waking up the server…").

   (Alternative hosts: *Railway* — New project → Deploy from GitHub, add the same variables, it uses the `Procfile`;
   *Fly.io / Koyeb / any Docker host* — use the `Dockerfile`.)

### 5. Publish the frontend on GitHub Pages
1. In the GitHub repo: *Settings → Secrets and variables → Actions → Variables → New repository variable*
   `API_BASE` = your Render URL, e.g. `https://ai-dataset-cleaner-api.onrender.com` (no trailing slash).
2. *Settings → Pages → Build and deployment → Source: **GitHub Actions***.
3. *Actions* tab → *Deploy frontend to GitHub Pages* → *Run workflow* (it also runs automatically on every push to `main`).
4. Your site is at `https://YOUR-USERNAME.github.io/REPO-NAME/`.
   Make sure Render's `ALLOWED_ORIGINS` contains `https://YOUR-USERNAME.github.io` (Render → service → Environment).

Without the `API_BASE` variable you can also just edit `frontend/config.js` by hand, or open the page with
`?api=https://your-backend.onrender.com` once (the browser remembers it).

### 6. Done
Open the GitHub Pages URL, upload a file, Analyze, Ask AI, review, Apply, download. Each visitor gets an anonymous session
(a random token in their browser) and only sees their own datasets; datasets are deleted after `DATASET_TTL_DAYS` (7).

---

## B. Run locally

```
copy .env.example .env        (macOS/Linux: cp .env.example .env)
```
Edit `.env`: either keep `LLM_PROVIDER=groq` with your key, or set `LLM_PROVIDER=ollama` to use a local model
(`ollama pull llama3.2:3b`). `DATABASE_URL` can stay as SQLite.

Then double-click `run.bat` (Windows) or `./run.sh`, and open http://localhost:8000.
In local mode the ⚙ Settings panel lets you switch provider/model/key at runtime.

---

## C. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `groq`, `gemini`, `openrouter`, `openai`, `anthropic`, `ollama` |
| `LLM_API_KEY` | | provider key (not needed for ollama) |
| `LLM_MODEL` | provider default | e.g. `llama-3.3-70b-versatile`, `gemini-2.5-flash`, `llama3.2:3b` |
| `LLM_BASE_URL` | provider default | any OpenAI-compatible endpoint |
| `COLUMNS_PER_CALL` | 1 local / 3 hosted | columns per AI call (1 = deepest; higher = fewer calls) |
| `ALLOW_CLIENT_SETTINGS` | true only for ollama | let the Settings panel change provider/key (keep **false** when hosted) |
| `DATABASE_URL` | SQLite file | Postgres connection string when hosted |
| `ALLOWED_ORIGINS` | `*` | comma-separated origins allowed to call the API |
| `SERVE_FRONTEND` | true | also serve `frontend/` from the backend URL |
| `MAX_UPLOAD_MB` / `MAX_ROWS` / `MAX_COLUMNS` | 10 / 100000 / 200 | upload limits |
| `DATASET_TTL_DAYS` | 7 | auto-delete old datasets |
| `MAX_DATASETS_PER_SESSION` | 20 | oldest are removed beyond this |
| `MAX_CONCURRENT_JOBS` | 2 | AI planning jobs running at once |

---

## D. What it does

**Profiler (rule engine)** — disguised nulls (`N/A`, `-`, `?`, `null`, `unknown`…), numbers/dates/booleans stored as text
(currency, `%`, thousands separators, mixed date formats), mojibake (`JosÃ©`), whitespace, inconsistent case, near-duplicate
category spellings (`New Yrok`), outliers (IQR / z-score), sentinel codes (`-999`), negative ages/prices, impossible dates,
duplicate IDs/rows/columns, constant & empty columns, messy headers, invalid emails/phones — each with evidence, severity,
and candidate fixes. Produces a 0–100 quality score.

**AI planner** — one structured request per column (or per batch): the model reads the profile, the issues and the candidate
actions, writes its reasoning, and chooses from a fixed toolbox of 25 actions. Everything it returns is validated against the
real columns/values; hallucinated columns or mappings are dropped with a visible warning. If the AI is unreachable a
rule-based plan is used instead.

**Toolbox** — `fix_encoding`, `strip_whitespace`, `replace_disguised_nulls`, `replace_values_with_null`, `normalize_case`,
`normalize_categories` (explicit mappings, e.g. `Bengaluru → Bangalore`), `normalize_boolean`, `standardize_email/phone`,
`parse_numeric`, `parse_dates`, `convert_type`, `round_numeric`, `remove_rows_where`, `clip_values`, `handle_outliers`,
`fill_missing` (mean/median/mode/constant/ffill/bfill/interpolate), `drop_columns`, `drop_empty_rows`,
`drop_rows_with_missing`, `drop_duplicate_rows`, `rename_column`, `standardize_column_names`, `regex_replace`,
`remove_special_characters`. Actions run in a safe canonical order; every one is logged with before/after counts and examples.

**UI** — review/edit/untick actions, add manual ones, apply, before/after score, change log, remaining issues, downloads
(CSV / Excel / JSON / Markdown report / JSON report), "run another pass" on the cleaned data, previous datasets list.

---

## E. Project layout

```
app/
  main.py      FastAPI API (sessions, upload, analyze, plan jobs, apply, downloads, report)
  llm.py       provider clients (Ollama, OpenAI-compatible, Anthropic)
  planner.py   prompts, JSON schema, validation, background planning job, rule-based fallback
  profiler.py  deep profiling + issue detection + quality score
  cleaner.py   the action toolbox + executor + change log
  db.py        SQLAlchemy persistence (Postgres / SQLite)
  report.py    Markdown report      config.py  env-driven settings      loaders.py  CSV/Excel/JSON loading
frontend/      index.html, app.js, style.css, config.js  (static — GitHub Pages)
.github/workflows/pages.yml   deploys frontend/ to GitHub Pages
render.yaml · Dockerfile · Procfile · .env.example
sample_data/messy_employees.csv   demo file
tests/         make_sample.py, run_pipeline.py (smoke test), mock_ollama.py (fake Ollama + fake OpenAI API for offline tests)
```

## F. Troubleshooting

* **Pill says "backend offline"** — the Render service is waking up (wait a minute) or `API_BASE` in `config.js` is wrong.
* **CORS error in the browser console** — add your GitHub Pages origin to `ALLOWED_ORIGINS` on Render (no trailing slash).
* **"LLM_API_KEY is not set" / 401** — check the key in Render → Environment; redeploy.
* **429 rate limited** — Groq free daily limit reached: wait, or set `LLM_MODEL=llama-3.1-8b-instant`, or raise `COLUMNS_PER_CALL`.
* **Datasets disappear** — they expire after `DATASET_TTL_DAYS`, and the session token lives in the browser (clearing site data = new session).
* **Render free tier restarts** — a planning job that was running is lost; just click "Ask AI" again. Data is safe in Postgres.
