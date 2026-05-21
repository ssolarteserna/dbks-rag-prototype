# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>
# MAGIC

# COMMAND ----------

# DBTITLE 1,Introduction and Tasks
# MAGIC %md
# MAGIC # Lab - Building Vector Search for Retrieval
# MAGIC
# MAGIC Welcome to the lab! Follow the steps below to learn how to build a vector search solution for document retrieval using Databricks.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC In this lab, you will work with document chunks stored in a parquet file to build a complete vector search solution. You will learn how to **prepare data**, **create vector search indexes**, and **perform various types of searches** using Databricks Vector Search capabilities.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC 1. **Read** parquet data and save as Delta table with Change Data Feed enabled.
# MAGIC 1. **Create** a vector search index using the Databricks UI.
# MAGIC 1. **Get** the vector search index for performing searches.
# MAGIC 1. **Implement** similarity search with reranking for improved precision.
# MAGIC 1. **Execute** hybrid search with filtering to target specific documents.
# MAGIC
# MAGIC ## Requirements
# MAGIC - A pre-created **vector search endpoint**. This is pre-created for you.
# MAGIC - **Serverless Compute (environment version 5)**. Follow the instructions [here](https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version) to select the appropriate environment version.
# MAGIC - Required libraries are added to **Dependencies** of Serverless compute configuration.
# MAGIC - Appropriate permissions to create and manage vector search indexes.
# MAGIC - Access to Foundation Model APIs for embedding generation.
# MAGIC
# MAGIC
# MAGIC **📌 Your Task: In this lab, your task is to replace `<FILL_IN>` sections with appropriate code.**

# COMMAND ----------

# DBTITLE 1,Setup Section
# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC Run the code below to install required libraries and configure your classroom environment. This step ensures all dependencies are available and your workspace is ready for the demo.

# COMMAND ----------

# DBTITLE 1,Classroom Setup
# MAGIC %run ../Includes/Classroom-Setup-03

# COMMAND ----------

# DBTITLE 1,Task 1: Read Parquet and Create Delta Table
# MAGIC %md
# MAGIC ## Task 1: Read Parquet Data and Create Delta Table with CDF
# MAGIC
# MAGIC In this section, you will **read document chunks from a parquet file** and save them as a Delta table with Change Data Feed (CDF) enabled. CDF is required for vector search synchronization.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Read the parquet file containing document chunks using pandas.
# MAGIC 2. Convert to Spark DataFrame and save as a Delta table.
# MAGIC 3. Enable Change Data Feed on the table.
# MAGIC 4. Display sample data to verify the table structure.
# MAGIC
# MAGIC Complete the code below to perform this task.

# COMMAND ----------

docs_chunked_lab_3 = f"{catalog}.{schema}.docs_chunked_lab_3"

# COMMAND ----------

# Read parquet file and create Delta table with CDF enabled
import os
import pandas as pd

# Define the parquet file path
parquet_path = f"/Volumes/{catalog}/{schema}/orion_text/docs_chunked.parquet"

# Read Parquet file using pandas
pdf = <FILL_IN>

# Drop the table if it already exists to avoid conflicts
spark.sql(f"DROP TABLE IF EXISTS {docs_chunked_lab_3}")

# Convert pandas DataFrame to Spark DataFrame and write to Unity Catalog
df = spark.createDataFrame(pdf)
df.write.<FILL_IN>

# Enable Change Data Feed for vector search synchronization
spark.sql(f"<FILL_IN>")

print(f"👍 Table '{docs_chunked_lab_3}' created with Change Data Feed enabled.")

# COMMAND ----------

# MAGIC %skip
# MAGIC # Read parquet file and create Delta table with CDF enabled
# MAGIC import os
# MAGIC import pandas as pd
# MAGIC
# MAGIC # Define the parquet file path
# MAGIC parquet_path = f"/Volumes/{catalog}/{schema}/orion_text/docs_chunked.parquet"
# MAGIC
# MAGIC # Read Parquet file using pandas
# MAGIC pdf = pd.read_parquet(parquet_path)
# MAGIC
# MAGIC # Drop the table if it already exists to avoid conflicts
# MAGIC spark.sql(f"DROP TABLE IF EXISTS {docs_chunked_lab_3}")
# MAGIC
# MAGIC # Convert pandas DataFrame to Spark DataFrame and write to Unity Catalog
# MAGIC df = spark.createDataFrame(pdf)
# MAGIC df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(docs_chunked_lab_3)
# MAGIC
# MAGIC # Enable Change Data Feed for vector search synchronization
# MAGIC spark.sql(f"ALTER TABLE {docs_chunked_lab_3} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# MAGIC
# MAGIC print(f"👍 Table '{docs_chunked_lab_3}' created with Change Data Feed enabled.")

# COMMAND ----------

# Display sample data to understand table structure
display(spark.sql(f"SELECT * FROM {docs_chunked_lab_3} LIMIT 5"))

# COMMAND ----------

# DBTITLE 1,Task 2: Create Vector Search Index via UI
# MAGIC %md
# MAGIC ## Task 2: Create Vector Search Index Using the UI
# MAGIC
# MAGIC In this section, you will **create a vector search index using the Databricks UI**. This approach provides a user-friendly interface for configuring your index with managed embeddings.
# MAGIC
# MAGIC **Steps to Create the Index via UI:**
# MAGIC
# MAGIC 1. In the left sidebar, click **Catalog** to open Catalog Explorer.
# MAGIC 2. Navigate to your catalog and schema.
# MAGIC 3. Find and select your Delta table (`docs_chunked_lab_3`).
# MAGIC 4. Click **Create** (top right) and choose **Vector search index**.
# MAGIC 5. In the dialog, configure the following settings:
# MAGIC    * **Name:** Enter name of the index as `docs_chunked_lab_index`
# MAGIC    * **Primary key:** Select `id` (the unique identifier column)
# MAGIC    * **Columns to sync:** Leave blank to sync all columns
# MAGIC    * **Embedding source:** Choose **Compute embeddings**
# MAGIC      - **Embedding source column:** Select `chunk`
# MAGIC      - **Embedding model:** Choose `databricks-gte-large-en`
# MAGIC    * **Sync computed embeddings:** Toggle **OFF** to save embeddings to the table
# MAGIC    * **Vector search endpoint:** Select your endpoint. Note that you should have this endpoint ready.
# MAGIC    * **Sync mode:** Choose **Triggered** (manual sync)
# MAGIC 6. Click **Create** and monitor index creation progress.
# MAGIC
# MAGIC **⏱️ Wait Time:** Index creation typically takes 2-3 minutes. You can monitor progress in the UI.
# MAGIC
# MAGIC Once your index is created, proceed to the next task.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Get Vector Search Index Details
# MAGIC
# MAGIC Get the vector search index you just created and show the details.

# COMMAND ----------

# DBTITLE 1,Task 3: Get Vector Search Index # TODO
# Get the vector search index for performing searches

from databricks.vector_search.client import VectorSearchClient

# Define the index name that you created via UI
index_name = f"{catalog}.{schema}.docs_chunked_lab_index"

# Initialize the Vector Search client for later use
vsc = VectorSearchClient(disable_notice=True)

# Get the index using the vector search client
index = vsc.<FILL_IN>

# Display index information
print(index.describe())

# COMMAND ----------

# DBTITLE 1,Task 3: Get Vector Search Index # ANSWER
# MAGIC %skip
# MAGIC # Get the vector search index for performing searches
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC
# MAGIC # Define the index name that you created via UI
# MAGIC index_name = f"{catalog}.{schema}.docs_chunked_lab_index"
# MAGIC
# MAGIC # Initialize the Vector Search client for later use
# MAGIC vsc = VectorSearchClient(disable_notice=True)
# MAGIC
# MAGIC # Get the index using the vector search client
# MAGIC index = vsc.get_index(index_name=index_name)
# MAGIC
# MAGIC # Display index information
# MAGIC print(index.describe())

# COMMAND ----------

# DBTITLE 1,Task 3: Similarity Search with Reranking
# MAGIC %md
# MAGIC ## Task 4: Similarity Search with Reranking
# MAGIC
# MAGIC In this section, you will **perform similarity search with reranking** to improve retrieval precision. Reranking applies a secondary scoring step to prioritize the most contextually relevant results.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Perform similarity search with reranking for improved precision.
# MAGIC 1. Ask this question: `"How does the motion controller maintain balance during rapid movement?"`
# MAGIC 1. Return 3 results.
# MAGIC 1. Analyze the results to understand the impact of reranking.
# MAGIC
# MAGIC Complete the code below to perform this task.

# COMMAND ----------

# DBTITLE 1,Task 3: Basic Similarity Search # TODO
# Perform similarity search with reranking for improved precision

from databricks.vector_search.reranker import DatabricksReranker

query_text = "How does the motion controller maintain balance during rapid movement?"

# Perform similarity search with reranking
reranked_results = index.<FILL_IN>

print("=== Similarity Search with Reranking Results ===")
display(reranked_results)

# COMMAND ----------

# DBTITLE 1,Task 3: Basic Similarity Search # ANSWER
# MAGIC %skip
# MAGIC # Perform similarity search with reranking for improved precision
# MAGIC
# MAGIC from databricks.vector_search.reranker import DatabricksReranker
# MAGIC
# MAGIC query_text = "How does the motion controller maintain balance during rapid movement?"
# MAGIC
# MAGIC # Perform similarity search with reranking
# MAGIC reranked_results = index.similarity_search(
# MAGIC     query_text=query_text,
# MAGIC     columns=["path", "chunk"],
# MAGIC     num_results=3,
# MAGIC     reranker=DatabricksReranker(columns_to_rerank=["chunk"])
# MAGIC )
# MAGIC
# MAGIC print("=== Similarity Search with Reranking Results ===")
# MAGIC display(reranked_results)

# COMMAND ----------

# DBTITLE 1,Task 4: Hybrid Search
# MAGIC %md
# MAGIC ## Task 5: Hybrid Search with Filters – Targeted Document Search
# MAGIC
# MAGIC In this section, you will **implement hybrid search with filtering** to combine semantic similarity, keyword matching, and document targeting. This approach delivers highly precise results by leveraging multiple search strategies together.
# MAGIC
# MAGIC Suppose you want to answer: **"Find procedures that describe battery replacement for the A1 model."**
# MAGIC
# MAGIC - *Pure semantic (similarity) search* may return passages about battery charging or thermal management, since they are semantically related.
# MAGIC - *Adding a keyword filter* for "Battery" or "A1" narrows the results to relevant document sections.
# MAGIC - *Filtering by file name* further ensures you retrieve only the procedures from the correct document.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Perform a **hybrid search** for the query.
# MAGIC 1. **Filter results** to only those from `05_Orion_Maintenance_and_Servicing_Guide_v3.pdf`.
# MAGIC 1. Return **2 records** with **all columns**.
# MAGIC 1. Analyze how combining hybrid search and filters improves search precision.
# MAGIC
# MAGIC Complete the code below to perform this task.

# COMMAND ----------

# DBTITLE 1,Task 4: Hybrid Search # TODO
# Perform hybrid search with filtering to target specific documents

query_text = "Find procedures that describe battery replacement for the A1 model."

# Perform hybrid search with document path filter
filtered_hybrid_results = index.<FILL_IN>

print("=== Hybrid Search with Filters Results ===")
display(filtered_hybrid_results)

# COMMAND ----------

# DBTITLE 1,Task 4: Hybrid Search # ANSWER
# MAGIC %skip
# MAGIC # Perform hybrid search with filtering to target specific documents
# MAGIC
# MAGIC query_text = "Find procedures that describe battery replacement for the A1 model."
# MAGIC
# MAGIC # Perform hybrid search with document path filter
# MAGIC filtered_hybrid_results = index.similarity_search(
# MAGIC     query_text=query_text,
# MAGIC     columns=["id","path", "chunk"],
# MAGIC     query_type="hybrid",
# MAGIC     filters={"path LIKE": "05_Orion_Maintenance_and_Servicing_Guide_v3.pdf"},  # Filter by path containing safety-related documents
# MAGIC     num_results=5
# MAGIC )
# MAGIC
# MAGIC print("=== Hybrid Search with Filters Results ===")
# MAGIC display(filtered_hybrid_results)

# COMMAND ----------

# DBTITLE 1,Hybrid Search Analysis
# MAGIC %md
# MAGIC **💡 Analysis & Reflection:**
# MAGIC
# MAGIC **Which search method would you choose for:**
# MAGIC - A technical documentation system where users search for specific procedures?
# MAGIC - A legal repository where precision is critical?
# MAGIC - A customer support knowledge base with varied query types?
# MAGIC
# MAGIC **Think About:** Based on your results, which approach would you recommend for a production RAG system and why?

# COMMAND ----------

# DBTITLE 1,Summary and Next Steps
# MAGIC %md
# MAGIC ## Summary and Next Steps
# MAGIC
# MAGIC You have completed the lab to build a vector search solution for document retrieval using Databricks. You learned how to:
# MAGIC
# MAGIC * **Prepare** document data by reading from parquet and creating a Delta table with Change Data Feed enabled.
# MAGIC * **Create** a vector search index using the Databricks UI with managed embeddings.
# MAGIC * **Implement** similarity search with reranking to improve retrieval precision.
# MAGIC * **Execute** hybrid search combining semantic similarity with keyword matching.
# MAGIC * **Apply** filters to target specific documents and narrow search scope.
# MAGIC
# MAGIC **Next Steps (Optional):**
# MAGIC * Explore different embedding models and experiment with **multilingual and domain-specific** embedding models.
# MAGIC * Investigate how endpoint configuration, embedding dimension, and refresh mode affect latency and cost.
# MAGIC * Investigate how to retrieve rows based on user's credentials. 

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
