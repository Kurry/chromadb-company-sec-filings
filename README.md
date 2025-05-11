# SEC Filings ChromaDB Builder

This tool creates a vector database from SEC filings for a specific company ticker. It extracts, processes, and embeds sections from 10-K and 10-Q filings to enable semantic search and retrieval.

## Features

- Extract sections from SEC filings using `edgartools` and `sec-parsers`
- Strategic chunking with hierarchical structure preservation
- Financial metadata enrichment
- Embedding using finance-specific models
- Storage in ChromaDB for efficient retrieval

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python sec_filings_db.py --ticker TSLA --filing-types 10-K,10-Q --years 3 --output-dir ./tsla_db
```

Options:
- `--ticker`: Company ticker symbol (required)
- `--filing-types`: Types of filings to process (default: 10-K,10-Q)
- `--years`: Number of years of filings to fetch (default: 3)
- `--output-dir`: Directory to store ChromaDB (default: ./sec_db)
- `--embedding-model`: Model to use for embeddings (default: FinLang/investopedia_embedding)
- `--batch-size`: Number of chunks to process in a batch (default: 32)

## Query Examples

Use the resulting ChromaDB for financial analysis:

```python
from query_sec_db import SECQueryEngine

engine = SECQueryEngine("./tsla_db")
results = engine.query("What were Tesla's main risk factors in 2022?")
```

## Using Private or Gated HuggingFace Models

If you want to use a private or gated model from HuggingFace (for example, a model like `FinLang/investopedia_embedding`), you must authenticate with the HuggingFace Hub using your access token.

### Steps:
1. Go to https://huggingface.co/settings/tokens and create a new access token (choose "Read" access).
2. Install the HuggingFace CLI if you haven't already:
   ```bash
   pip install huggingface_hub
   ```
3. Log in using your token:
   ```bash
   huggingface-cli login
   ```
   Paste your token when prompted.

Once logged in, you can access private or gated models in your scripts. If you encounter a 401 Unauthorized error, make sure you are logged in and your token has the correct permissions.