# Databricks notebook source
# MAGIC %md
# MAGIC # Media Reviews RAG — Vector Search Notebook
# MAGIC
# MAGIC **Source table:** `samples.bakehouse.media_gold_reviews_chunked`
# MAGIC **Index:** `data_studio_interviews.default.media_reviews_idx`
# MAGIC **Endpoint:** `qubika-vs-dev` (STANDARD)
# MAGIC **Embedding model:** `databricks-bge-large-en`
# MAGIC **Pipeline type:** TRIGGERED

# COMMAND ----------

# MAGIC %md ## Cell 0 — Install (run once in a real notebook, triggers env restart)

# COMMAND ----------

# %pip install databricks-vectorsearch
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Cell 1 — Config

# COMMAND ----------

SOURCE_TABLE    = "samples.bakehouse.media_gold_reviews_chunked"
ENDPOINT_NAME   = "qubika-vs-dev"
INDEX_NAME      = "data_studio_interviews.default.media_reviews_idx"
EMBEDDING_MODEL = "databricks-bge-large-en"

# COMMAND ----------

# MAGIC %md ## Cell 2 — Connect

# COMMAND ----------

import requests
import json

HOST  = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def vs_query(query_text, num_results=5, query_type="ANN", filters=None):
    payload = {
        "query_text": query_text,
        "columns": ["franchiseID", "chunk_id", "chunked_text", "review_date", "review_uri"],
        "num_results": num_results,
        "query_type": query_type,
    }
    if filters:
        payload["filters_json"] = filters
    resp = requests.post(
        f"https://{HOST}/api/2.0/vector-search/indexes/{INDEX_NAME}/query",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    return resp.json().get("result", {}).get("data_array", [])

print("Connected.")

# COMMAND ----------

# MAGIC %md ## Cell 3 — Similarity search

# COMMAND ----------

results = vs_query("negative experiences in Asia", num_results=3)
for r in results:
    print(f"franchiseID={r[0]}  score={r[-1]:.4f}")
    print(r[2][:300])
    print()

# COMMAND ----------

# MAGIC %md ## Cell 4 — Hybrid search (semantic + keyword BM25)

# COMMAND ----------

results = vs_query("1 star cookie experience", num_results=5, query_type="HYBRID")
for r in results:
    print(f"franchiseID={r[0]}  score={r[-1]:.4f}")
    print(r[2][:200])
    print()

# COMMAND ----------

# MAGIC %md ## Cell 5 — Metadata filter (equality, STANDARD endpoint)

# COMMAND ----------

# Filter to a specific franchise — add any column that is in columns_to_sync
# results = vs_query("oatmeal cookies", num_results=5, filters={"franchiseID": 3000018})
# for r in results:
#     print(r)

# COMMAND ----------

# MAGIC %md ## Cell 6 — Trigger sync after Silver pipeline run

# COMMAND ----------

resp = requests.post(
    f"https://{HOST}/api/2.0/vector-search/indexes/{INDEX_NAME}/sync",
    headers=HEADERS,
    timeout=30,
)
print(f"Sync triggered: HTTP {resp.status_code}")

# COMMAND ----------

# MAGIC %md ## Cell 7 — Check index status

# COMMAND ----------

resp = requests.get(
    f"https://{HOST}/api/2.0/vector-search/indexes/{INDEX_NAME}",
    headers=HEADERS,
    timeout=30,
)
status = resp.json().get("status", {})
print(f"State:   {status.get('detailed_state')}")
print(f"Message: {status.get('message')}")
print(f"Ready:   {status.get('ready')}")

# COMMAND ----------

# MAGIC %md ## Cell 8 — Cleanup (dev only — flip _CLEANUP = True to run)

# COMMAND ----------

_CLEANUP = False   # <- set True only when done testing

if _CLEANUP:
    r1 = requests.delete(
        f"https://{HOST}/api/2.0/vector-search/indexes/{INDEX_NAME}",
        headers=HEADERS, timeout=30
    )
    print(f"Index deleted:    HTTP {r1.status_code}")

    r2 = requests.delete(
        f"https://{HOST}/api/2.0/vector-search/endpoints/{ENDPOINT_NAME}",
        headers=HEADERS, timeout=30
    )
    print(f"Endpoint deleted: HTTP {r2.status_code}")
