import time, os, mpld3
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from textwrap import dedent
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

plt.rcParams['figure.figsize'] = (15, 10)

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Connect to Google Cloud

from google.cloud import bigquery
from google.cloud import storage

# Configure Spark and GraphFrames

import findspark, pyspark, os, sys
findspark.init()
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

SUBMIT_ARGS = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS
conf = SparkConf().setAll([('spark.jars', 'gs://spark-lib/bigquery/spark-3.1-bigquery-0.27.1-preview.jar')])
sc = SparkContext(conf=conf)
pyfiles = str(sc.getConf().get(u'spark.submit.pyFiles')).split(',')
sys.path.extend(pyfiles)

sqlContext = SQLContext(sparkContext=sc)
spark = sqlContext.sparkSession
bucket = "bd6893_data_yq"
spark.conf.set('temporaryGcsBucket', bucket)

from graphframes import *


####################################################
# DEFINE PYTHON FUNCTIONS
####################################################


def get_and_store_data():

	# Manage parameters, ex. date, table name ...

	end_date = date.today()
	start_date = end_date - timedelta(days=7)

	dataset_id = "transaction_network"
	table_name = "graph_" + str(end_date)
	edge_table_name = table_name + "_edge"
	node_table_name = table_name + "_node"
	sub_node_table_name = node_table_name + "_sub"
	sub_edge_table_name = edge_table_name + "_sub"

	temp_view_name = "temp_data"
	edge_temp_view_name = "edge_temp_data"


	# Connect to BigQuery
	client = bigquery.Client()

	# Prepare a reference to a new dataset for storing the query results
	dataset_id_full = f"{client.project}.{dataset_id}"
	dataset = bigquery.Dataset(dataset_id_full)

	# Create the new BigQuery dataset
	dataset = client.create_dataset(dataset)

	# Configure the query job
	job_config = bigquery.QueryJobConfig()
	job_config.destination = f"{dataset_id_full}.{table_name}"

	# Execute the query
	post_merge_query = f"""
	    SELECT * FROM bigquery-public-data.crypto_ethereum.transactions
	    WHERE DATE(block_timestamp) >= "{start_date}" AND DATE(block_timestamp) < "{end_date}"
	    AND (to_address) IS NOT NULL
	    AND (gas_price) IS NOT NULL
	"""
	post_merge = client.query(post_merge_query, job_config=job_config)

	# Get query data, and create view
	temp_data = spark.read.format('bigquery') \
	    .option('table', f'big-data-6893-yunjie-qian:{dataset_id}.{table_name}') \
	    .load()
	temp_data.createOrReplaceTempView(temp_view_name)

	# Aggregate edge attributes
	edge_query = f'''
	SELECT from_address AS src, to_address AS dst,
	SUM(value) AS total_value, MIN(gas_price) AS min_gas_price, COUNT(input) AS transaction_count
	FROM {temp_view_name} 
	GROUP BY from_address, to_address
	'''
	edge_df = spark.sql(edge_query)
	edge_df.createOrReplaceTempView(edge_temp_view_name)

	# Write edge data to BigQuery table
	edge_df.write.format('bigquery') \
		.option('table', f'{dataset_id}.{edge_table_name}') \
		.save()


	# Aggregate node degrees and other node attributes
	node_query_src = f'''
	SELECT src AS id, COUNT(src) AS outdegree, 
	SUM(total_value) AS out_total_value, SUM(transaction_count) AS out_total_transaction
	FROM {edge_temp_view_name}
	GROUP BY src
	'''
	node_query_dst = f'''
	SELECT dst AS id, COUNT(dst) AS indegree, 
	SUM(total_value) AS in_total_value, SUM(transaction_count) AS in_total_transaction
	FROM {edge_temp_view_name}
	GROUP BY dst
	'''
	node_df_src = spark.sql(node_query_src)
	node_df_dst = spark.sql(node_query_dst)

	# Preprocess node data
	node_df = node_df_src.join(node_df_dst, on="id", how="full")
	node_df = node_df.na.fill(value=0)
	node_df = node_df.withColumn('degree', node_df.indegree + node_df.outdegree)
	node_df = node_df.withColumn('total_value', node_df.out_total_value + node_df.in_total_value)
	node_df = node_df.withColumn('total_transaction', node_df.out_total_transaction + node_df.in_total_transaction)

	# Write node data to BigQuery table
	node_df.write.format('bigquery') \
		.option('table', f'{dataset_id}.{node_table_name}') \
		.save()
	

	# Create Graph and filter to get a subgraph
	g = GraphFrame(node_df, edge_df)
	subg = g.filterVertices("degree >= 30").filterEdges("transaction_count >= 10").dropIsolatedVertices()
	
	# Write the subgraph edges and nodes to BigQuery
	subg.vertices.write.format('bigquery') \
		.option('table', f'{dataset_id}.{sub_node_table_name}') \
		.save()
	subg.edges.write.format('bigquery') \
		.option('table', f'{dataset_id}.{sub_edge_table_name}') \
		.save()


	# The function to store data into Cloud Storage Bucket

	def to_bucket(fig, fig_name):

		# Use mpld3 to change figure to HTML format
		html_fragment = mpld3.fig_to_html(fig, figid = 'fig1')
		html_doc = f'''
		<style type="text/css">
		div#fig1 {{ text-align: center }}
		</style>

		{html_fragment}
		'''
		Html_file = open(f"{fig_name}.html", "w")
	    Html_file.write(html_doc)
	    Html_file.close()

	    # Connect to Cloud Storage
		storage_client = storage.Client()
		bucket = storage_client.bucket('public_bucket_6893')

		# Write to Bucket
		blob = bucket.blob(f"{fig_name}.html")
		with blob.open("w") as f:
			f.write(html_doc)
		# blob.upload_from_filename(os.getcwd() + '/' + f"{fig_name}.html")


	# Plot transaction times

	temp_data = temp_data.withColumn('date', to_date("block_timestamp"))

	trans_data = temp_data.select('date').groupBy('date').count().orderBy('date', ascending=True)
	trans = trans_data.select('date').rdd.flatMap(lambda x: x).collect()
	trans_count = trans_data.select('count').rdd.flatMap(lambda x: x).collect()

	fig, ax = plt.figure(), plt.axes()
	ax.plot(trans, trans_count, 'go-', label='Number of Transactions')
	ax.set_xlabel('\nDate', fontsize=16)
	ax.set_ylabel('Number of Transactions\n', fontsize=16)
	ax.legend(fontsize=16)
	ax.grid()

	to_bucket(fig, 'trans_dist_' + str(end_date))


	# Plot transaction value

	temp_data = temp_data.withColumn('value_new', temp_data.value * 1e-18)
	temp_data = temp_data.withColumn('value_new', temp_data.value_new.cast('int'))

	value_data = temp_data.select('date', 'value_new').groupBy('date').agg(sum("value_new").alias("total_value")).orderBy('date', ascending=True)
	value = value_data.select('date').rdd.flatMap(lambda x: x).collect()
	value_count = value_data.select('total_value').rdd.flatMap(lambda x: x).collect()

	fig, ax = plt.figure(), plt.axes()
	ax.plot(value, value_count, 'ro-', label='Total Transaction Value')
	ax.set_xlabel('\nDate', fontsize=16)
	ax.set_ylabel('Total Transaction Value\n', fontsize=16)
	ax.legend(fontsize=16)
	ax.grid()

	to_bucket(fig, 'value_dist_' + str(end_date))


	# Plot transaction gas price

	temp_data = temp_data.withColumn('gas_price_new', temp_data.gas_price * 1e-9)
	temp_data = temp_data.withColumn('gas_price_new', temp_data.gas_price_new.cast('int'))

	gas_data = temp_data.select('date', 'gas_price_new').groupBy('date').agg(min("gas_price_new").alias("min_gas_price"), avg("gas_price_new").alias("total_gas_price")).orderBy('date', ascending=True)
	gas = gas_data.select('date').rdd.flatMap(lambda x: x).collect()
	gas_min = gas_data.select('min_gas_price').rdd.flatMap(lambda x: x).collect()
	gas_count = gas_data.select('total_gas_price').rdd.flatMap(lambda x: x).collect()

	fig, ax = plt.figure(), plt.axes()
	ax.plot(gas, gas_min, 'bo-', label='Min Gas Price')
	ax.plot(gas, gas_count, 'yo-', label='Average Gas Price')
	ax.set_xlabel('\nDate', fontsize=16)
	ax.set_ylabel('Transaction Gas Price\n', fontsize=16)
	ax.set_yticks(np.arange(0, 51, 5))
	ax.legend(fontsize=16)
	ax.grid()

	to_bucket(fig, 'gas_dist_' + str(end_date))



############################################
# DEFINE AIRFLOW DAG (SETTINGS + SCHEDULE)
############################################

default_args = {
    'owner': 'yunhang',
    'depends_on_past': False,
    'email': ['yl4860@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(seconds=2),
}

with DAG(
    'fetch_weekly_data_one_step',
    default_args=default_args,
    description='A simple toy DAG',
    schedule_interval=timedelta(days=1),  # Daily execution
    start_date=datetime(2022,12,16,4),
    catchup=False,
    tags=['example'],
) as dag:

##########################################
# DEFINE AIRFLOW OPERATORS
##########################################

   
    t1 = PythonOperator(
        task_id='t1_get_and_store_data',
        python_callable=get_and_store_data,
        retries=5,
    )


##########################################
# DEFINE TASKS HIERARCHY
##########################################

    # task dependencies 

    # t1 >> t2 >> t3 >> t4
    # t4 >> [t5_1, t5_2, t5_3] >> t6