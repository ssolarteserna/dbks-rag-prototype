# Databricks notebook source

# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch==0.68
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# Variables — all values confirmed before notebook generation

SOURCE_TABLE    = "samples.bakehouse.media_gold_reviews_chunked"
CATALOG         = "default"
SCHEMA          = "data_studio_interviews"
ENDPOINT_NAME   = "qubika-vs-dev"
INDEX_NAME      = f"{CATALOG}.{SCHEMA}.media_reviews_idx"

EMBEDDING_MODEL = "databricks-bge-large-en"
PIPELINE_TYPE   = "TRIGGERED"

# COMMAND ----------
# Cell 2 — Endpoint creation (optional)
# The endpoint `qubika-vs-dev` does not exist in this workspace.
# Set _CREATE_ENDPOINT = True and run this cell once to create it.
# STANDARD endpoint type confirmed — equality filters only (filters_json=dict).
# WARNING: endpoint type cannot be changed after creation without deleting all indexes.

_CREATE_ENDPOINT = False

if _CREATE_ENDPOINT:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    vsc.create_endpoint(
        name=ENDPOINT_NAME,
        endpoint_type="STANDARD",
    )
    print(f"Endpoint '{ENDPOINT_NAME}' created. Poll with vsc.get_endpoint(ENDPOINT_NAME) until state=ONLINE.")
else:
    print("_CREATE_ENDPOINT is False — skipping endpoint creation.")

# COMMAND ----------
# Cell 3 — CDF verification
# samples.bakehouse is a read-only Databricks dataset with CDF already enabled.
# For your own Silver tables in a writable catalog, this cell checks and enables CDF.

props = spark.sql(f"SHOW TBLPROPERTIES {SOURCE_TABLE}").collect()
props_dict = {row["key"]: row["value"] for row in props}

source_catalog = SOURCE_TABLE.split(".")[0]

if source_catalog == "samples":
    print("CDF check skipped — Databricks sample datasets are read-only and have CDF enabled by default.")
elif props_dict.get("delta.enableChangeDataFeed", "false") != "true":
    spark.sql(f"""
        ALTER TABLE {SOURCE_TABLE}
        SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)
    print("CDF enabled.")
else:
    print("CDF already enabled — ready to index.")

# COMMAND ----------
# Cell 4 — Create the Delta Sync index  # QUBIKA-CONFIRMED-INDEX
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

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
# Cell 5 — Trigger initial sync

index = vsc.get_index(index_name=INDEX_NAME)
index.sync()
print("Index sync triggered.")

# COMMAND ----------
# Cell 6 — Similarity search

index = vsc.get_index(index_name=INDEX_NAME)

results = index.similarity_search(
    query_text="negative experiences in Asia",  # TODO: replace query_text with your search string
    columns=["franchiseID", "chunk_id", "chunked_text"],
    num_results=3,
)
display(results)

# COMMAND ----------
# Cell 7 — Hybrid search (semantic + keyword)
# Use hybrid when queries contain exact names, IDs, or codes that pure semantic search misses.

results_hybrid = index.similarity_search(
    query_text="1 star experience Asia",  # TODO: replace query_text with your search string
    columns=["franchiseID", "chunk_id", "chunked_text"],
    query_type="hybrid",
    num_results=5,
)
display(results_hybrid)

# COMMAND ----------
# Cell 8 — Reranking with DatabricksReranker
# Over-fetch with num_results=10, then rerank for higher precision on important queries.

from databricks.vector_search.reranker import DatabricksReranker

results_reranked = index.similarity_search(
    query_text="positive reviews of oatmeal cookies",  # TODO: replace query_text with your search string
    columns=["franchiseID", "chunk_id", "chunked_text"],
    num_results=10,
    reranker=DatabricksReranker(columns_to_rerank=["chunked_text"]),
)
display(results_reranked)

# COMMAND ----------
# Cell 9 — Cleanup (optional)
# Always delete your index when done in dev — it frees the slot on the shared endpoint.
# The shared endpoint is only deleted if no other indexes remain on it.

_CLEANUP = False

if _CLEANUP:
    vsc.delete_index(index_name=INDEX_NAME)
    assert not vsc.index_exists(index_name=INDEX_NAME), f"Index {INDEX_NAME} still exists"
    print(f"Index '{INDEX_NAME}' deleted.")

    remaining = vsc.list_indexes(ENDPOINT_NAME).get("vector_indexes", [])
    if not remaining:
        vsc.delete_endpoint(name=ENDPOINT_NAME)
        print(f"Endpoint '{ENDPOINT_NAME}' deleted — no remaining indexes.")
    else:
        print(
            f"Endpoint '{ENDPOINT_NAME}' kept — {len(remaining)} other index(es) still attached.\n"
            f"Remaining: {[i['name'] for i in remaining]}"
        )
else:
    print("_CLEANUP is False — skipping cleanup.")
