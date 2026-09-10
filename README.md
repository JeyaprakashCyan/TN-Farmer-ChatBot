# TN Farmer ChatBot

## Create the Python environment

On Windows PowerShell, run:

```powershell
.\setup-environment.ps1
.\venv\Scripts\Activate.ps1
```

The setup script creates the virtual environment with `--without-pip`, then installs pip with the system Python. This avoids the Python 3.14 `ensurepip` bootstrap that can hang or leave a partial environment.

If the environment is already damaged, run the same command again. The script recreates it in place.

## Run the scraper

```powershell
.\venv\Scripts\python.exe Scraper.py
```

The first setup also installs the Python packages listed in `requirements.txt`. Playwright may additionally need its browser binary:

```powershell
.\venv\Scripts\python.exe -m playwright install chromium
```

## Extract scheme text

```powershell
.\venv\Scripts\python.exe Extraction.py
```

This reads `raw_scheme_data/scraped_schemes.json`, runs OCR on downloaded images, and writes `raw_scheme_data/extracted_chunks.json`. Use `-o` to choose another output path or `--chunk-size` to change the chunk size.

## Index chunks in Neo4j

Keep `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `OPENAI_API_KEY` in the root `.env` file. The indexer loads that file automatically:

```powershell
.\venv\Scripts\python.exe indexfix.py
```

Use a different chunks file by passing its path as the first argument. Do not commit `.env` or expose its credentials.

If Neo4j reports a self-signed certificate error, `NEO4J_TRUST_ALL_CERTS=true` enables the tested `neo4j+ssc://` connection mode. For production, prefer installing the correct CA certificate and setting this value to `false`.

## Retrieve and rerank schemes

Install the dependencies, including `sentence-transformers`, through the normal environment setup, then run:

```powershell
.\venv\Scripts\python.exe Rerank.py "What subsidy is available for certified seeds?" --top-k 5
```

The command loads Neo4j and OpenAI credentials from `.env` and prints reranked results as JSON. You can also run it without a query and enter the question when prompted:

```powershell
.\venv\Scripts\python.exe Rerank.py
```

## Run the security audit

```powershell
.\venv\Scripts\python.exe Security.py
```

Add `--check-redis` to test Redis availability, or `-o security-report.json` to save the JSON report. Secrets are never printed by the audit.

## Run the farmer assistant

Start the Streamlit app with:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Then open the local URL shown by Streamlit. The app loads Neo4j and OpenAI settings from `.env`; Redis chat history falls back to in-process memory when Redis is unavailable.

## Run the Flask dashboard

`appfinal.py` is a Flask application and runs separately from the Streamlit app:

```powershell
.\venv\Scripts\python.exe appfinal.py
```

Open `http://127.0.0.1:8505/`. Port 8501 is reserved for the Streamlit app, so the two applications do not compete for the same local URL.