# Databricks notebook source
# MAGIC %md
# MAGIC # Media Reviews — Vector Search Index
# MAGIC
# MAGIC **Source table:** `samples.bakehouse.media_gold_reviews_chunked`
# MAGIC **Index:** `data_studio_interviews.default.media_gold_reviews_chunked_idx`
# MAGIC **Endpoint:** `qubika-vs-dev`

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

SOURCE_TABLE    = "samples.bakehouse.media_gold_reviews_chunked"
CATALOG         = "data_studio_interviews"
SCHEMA          = "default"
ENDPOINT_NAME   = "qubika-vs-dev"
INDEX_NAME      = f"{CATALOG}.{SCHEMA}.media_gold_reviews_chunked_idx"
EMBEDDING_MODEL = "databricks-bge-large-en"
PIPELINE_TYPE   = "TRIGGERED"

# COMMAND ----------

# MAGIC %md ## Step 1 — Verify CDF on source table

# COMMAND ----------

props = spark.sql(f"SHOW TBLPROPERTIES {SOURCE_TABLE}").collect()
props_dict = {row["key"]: row["value"] for row in props}

if props_dict.get("delta.enableChangeDataFeed", "false") != "true":
    spark.sql(f"""
        ALTER TABLE {SOURCE_TABLE}
        SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)
    print("CDF enabled.")
else:
    print("CDF already enabled — ready to index.")

# COMMAND ----------

# MAGIC %md ## Step 2 — Index Inventory Scan

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

indexes = vsc.list_indexes(endpoint_name=ENDPOINT_NAME).get("vector_indexes", [])
total = len(indexes)

print(f"Endpoint `{ENDPOINT_NAME}` currently hosts {total} of 50 indexes:\n")
for idx in indexes:
    name  = idx.get("name", "")
    itype = idx.get("index_type", "")
    state = idx.get("status", {}).get("detailed_state", "")
    print(f"  {name:<60} {itype:<20} {state}")

if total == 0:
    print("  (no indexes yet — 0 of 50 slots used)")

# COMMAND ----------

# MAGIC %md ## Step 3 — Create the Delta Sync index

# COMMAND ----------

# QUBIKA-CONFIRMED-INDEX
index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=ENDPOINT_NAME,
    index_name=INDEX_NAME,
    source_table_name=SOURCE_TABLE,
    primary_key="franchiseID",
    embedding_source_column="chunked_text",
    embedding_model_endpoint_name=EMBEDDING_MODEL,
    pipeline_type=PIPELINE_TYPE,
    columns_to_sync=["franchiseID", "chunk_id", "chunked_text"],
)
print(f"Index '{INDEX_NAME}' is ONLINE.")

# COMMAND ----------

# MAGIC %md ## Step 4 — Similarity search

# COMMAND ----------

index = vsc.get_index(index_name=INDEX_NAME)

results = index.similarity_search(
    query_text="negative experiences in Asia",
    columns=["chunk_id", "chunked_text"],
    num_results=3,
)
display(results)

# COMMAND ----------

# MAGIC %md ## Step 5 — Manual sync (run after each Silver pipeline update)

# COMMAND ----------

index = vsc.get_index(index_name=INDEX_NAME)
index.sync()
print("Index sync triggered.")
