# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # Bakehouse Reviews RAG Pipeline
# MAGIC
# MAGIC Builds a Delta Sync Vector Search index over `samples.bakehouse.media_gold_reviews_chunked`
# MAGIC and demonstrates similarity search, hybrid search, and reranking.
# MAGIC
# MAGIC **Deployment checklist (run top-to-bottom):**
# MAGIC 1. Install dependencies (Cell 1)
# MAGIC 2. Configure variables (Cell 2)
# MAGIC 3. Verify CDF on source table (Cell 3)
# MAGIC 4. Provision Vector Search endpoint — one-time manual step (Cell 4)
# MAGIC 5. Create the Delta Sync index (Cell 5)
# MAGIC 6. Run searches (Cells 6–8)
# MAGIC 7. Trigger sync after upstream pipeline runs (Cell 9)
# MAGIC 8. Cleanup when done (Cell 10)

# COMMAND ----------
# MAGIC %md ## Cell 1 — Install dependencies

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## Cell 2 — Variables

# COMMAND ----------

SOURCE_TABLE  = "samples.bakehouse.media_gold_reviews_chunked"
CATALOG       = "data_studio_interviews"
SCHEMA        = "default"
ENDPOINT_NAME = "qubika-vs-endpoint"   # shared team endpoint — reuse if it already exists
INDEX_NAME    = f"{CATALOG}.{SCHEMA}.media_reviews_idx"

print(f"Source : {SOURCE_TABLE}")
print(f"Index  : {INDEX_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------
# MAGIC %md ## Cell 3 — Verify Change Data Feed on source table
# MAGIC
# MAGIC Delta Sync requires CDF. The `samples.bakehouse` table already has it enabled,
# MAGIC but this cell makes the check explicit and safe to re-run on any table.

# COMMAND ----------

props = spark.sql(f"SHOW TBLPROPERTIES {SOURCE_TABLE}").collect()
props_dict = {row["key"]: row["value"] for row in props}

if props_dict.get("delta.enableChangeDataFeed", "false") != "true":
    spark.sql(f"""
        ALTER TABLE {SOURCE_TABLE}
        SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)
    print("CDF enabled — write one row or run OPTIMIZE before creating the index.")
else:
    print("CDF already enabled — ready to index.")

# COMMAND ----------
# MAGIC %md ## Cell 4 — Provision Vector Search endpoint (one-time, manual)
# MAGIC
# MAGIC Endpoints are workspace-scoped, always-on compute billed hourly.
# MAGIC **Do not automate this step.** Check for an existing endpoint first.
# MAGIC
# MAGIC If `ENDPOINT_NAME` already exists and is ONLINE, skip to Cell 5.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing_endpoints = [ep["name"] for ep in vsc.list_endpoints().get("endpoints", [])]
print("Existing endpoints:", existing_endpoints)

if ENDPOINT_NAME in existing_endpoints:
    ep = vsc.get_endpoint(name=ENDPOINT_NAME)
    state = ep["endpoint_status"]["state"]
    print(f"Endpoint '{ENDPOINT_NAME}' found — state: {state}")
    assert state == "ONLINE", f"Endpoint is {state}. Wait for it to be ONLINE before continuing."
else:
    print(f"Endpoint '{ENDPOINT_NAME}' not found. Creating it now (this takes ~10 min)...")
    vsc.create_endpoint_and_wait(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Endpoint '{ENDPOINT_NAME}' is ONLINE.")

# COMMAND ----------
# MAGIC %md ## Cell 5 — Create the Delta Sync index
# MAGIC
# MAGIC - `primary_key`: unique row identifier (`franchiseID`)
# MAGIC - `embedding_source_column`: the TEXT column to embed (`chunked_text`) — NOT the PK
# MAGIC - `columns_to_sync`: every column you will filter or return at query time
# MAGIC   (adding columns later requires full index deletion + recreation)
# MAGIC - `pipeline_type="TRIGGERED"`: syncs on demand; use CONTINUOUS only for sub-minute freshness

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=ENDPOINT_NAME,
    index_name=INDEX_NAME,
    source_table_name=SOURCE_TABLE,
    primary_key="franchiseID",
    embedding_source_column="chunked_text",               # the text column — not the PK
    embedding_model_endpoint_name="databricks-gte-large-en",
    pipeline_type="TRIGGERED",
    columns_to_sync=["franchiseID", "chunk_id", "chunked_text"],
)

print(f"Index '{INDEX_NAME}' is ONLINE.")

# COMMAND ----------
# MAGIC %md ## Cell 6 — Similarity search (semantic / ANN)

# COMMAND ----------

index = vsc.get_index(index_name=INDEX_NAME)

results = index.similarity_search(
    query_text="negative experiences in Asia",
    columns=["franchiseID", "chunk_id", "chunked_text"],
    num_results=3,
)
display(results)

# COMMAND ----------
# MAGIC %md ## Cell 7 — Hybrid search (semantic + keyword BM25)
# MAGIC
# MAGIC Use when queries contain exact product names, codes, or location names that
# MAGIC pure semantic search can miss.

# COMMAND ----------

results_hybrid = index.similarity_search(
    query_text="1 star experience Asia",
    columns=["franchiseID", "chunk_id", "chunked_text"],
    query_type="hybrid",   # fuses BM25 keyword + semantic vector scores
    num_results=5,
)
display(results_hybrid)

# COMMAND ----------
# MAGIC %md ## Cell 8 — Hybrid search + reranking
# MAGIC
# MAGIC Reranking improves precision by re-scoring the retrieved candidates with a
# MAGIC cross-encoder model. Over-fetch (e.g. 10 results) then let the reranker cull.

# COMMAND ----------

from databricks.vector_search.reranker import DatabricksReranker

results_reranked = index.similarity_search(
    query_text="positive reviews of oatmeal cookies",
    columns=["franchiseID", "chunk_id", "chunked_text"],
    query_type="hybrid",
    num_results=10,                                              # over-fetch, then rerank
    reranker=DatabricksReranker(columns_to_rerank=["chunked_text"]),  # text column only
)
display(results_reranked)

# COMMAND ----------
# MAGIC %md ## Cell 9 — Trigger index sync
# MAGIC
# MAGIC Add this as the final task in your Silver DLT job or Databricks Workflow
# MAGIC to keep the index current after every upstream run.

# COMMAND ----------

index = vsc.get_index(index_name=INDEX_NAME)
index.sync()
print("Index sync triggered.")

# COMMAND ----------
# MAGIC %md ## Cell 10 — Cleanup
# MAGIC
# MAGIC **Always clean up dev endpoints** — they bill hourly even when idle.
# MAGIC Delete the index FIRST, then the endpoint. Reversing the order fails.

# COMMAND ----------

vsc.delete_index(index_name=INDEX_NAME)
vsc.delete_endpoint(name=ENDPOINT_NAME)

assert not vsc.index_exists(index_name=INDEX_NAME),    f"Index {INDEX_NAME} still exists"
assert not vsc.endpoint_exists(name=ENDPOINT_NAME),    f"Endpoint {ENDPOINT_NAME} still exists"
print("Cleanup complete.")
