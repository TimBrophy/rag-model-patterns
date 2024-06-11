import streamlit as st
from elasticsearch import Elasticsearch
import os
from langchain_community.chat_models import AzureChatOpenAI, ChatOllama, BedrockChat
from langchain_community.embeddings import OllamaEmbeddings, AzureOpenAIEmbeddings
import uuid
import boto3
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
)
from ragas import evaluate
from datasets import Dataset
from datetime import timezone, datetime
# Set page parameters
st.set_page_config(
    page_title="RAG workbench: benchmark test",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# connect to the evaluation LLM
llm_provider = os.environ['evaluation_llm_provider']

if llm_provider == 'Azure OpenAI':
    eval_llm = AzureChatOpenAI(
        openai_api_base=os.environ['openai_api_base'],
        openai_api_version=os.environ['openai_api_version'],
        deployment_name=os.environ['deployment_name'],
        openai_api_key=os.environ['openai_api_key'],
        openai_api_type="azure",
        temperature=0,
        streaming=True
    )
elif llm_provider == 'AWS Bedrock':
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.environ['aws_region'],
                                  aws_access_key_id=os.environ['aws_access_key'],
                                  aws_secret_access_key=os.environ['aws_secret_key'])
    eval_llm = BedrockChat(
        client=bedrock_client,
        model_id=os.environ['aws_model_id'],
        streaming=True,
        model_kwargs={"temperature": 0})
elif llm_provider == 'Ollama':
    eval_llm = ChatOllama(model=os.environ['ollama_chat_model'])

# connect to the evaluation embedding model
embedding_provider = os.environ['evaluation_embedding_provider']
if embedding_provider == 'Azure OpenAI':
    eval_embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.environ['evaluation_embedding_model_deployment'],
        openai_api_base=os.environ['openai_api_base'],
        openai_api_version=os.environ['openai_api_version'],
        openai_api_key=os.environ['openai_api_key'],
        openai_api_type="azure",
    )
elif embedding_provider == 'Ollama':
    eval_embedding = OllamaEmbeddings(model=os.environ['evaluation_embedding_model'])


benchmarking_index = os.environ['benchmarking_index']
benchmarking_results_index = os.environ['benchmarking_results_index']
source_index = os.environ['default_index']

es = Elasticsearch(os.environ['elastic_url'], api_key=os.environ['elastic_api_key'])

# log the results of the benchmark test
def log_benchmark_test_results(question, ground_truth, answer, contexts, model, pattern_name, provider, report_name,
                               context_precision, faithfulness, answer_relevancy, answer_similarity, context_recall,
                               answer_correctness):
    log_id = uuid.uuid4()
    body = {
        "@timestamp": datetime.now(tz=timezone.utc),
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": contexts,
        "model": model,
        "pattern_name": pattern_name,
        "provider": provider,
        "report_name": report_name,
        "context_precision": context_precision,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "answer_similarity": answer_similarity,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness
    }
    es.index(index=benchmarking_results_index, id=log_id, document=body)
    return

# Get the data sources in the report index so that the user can choose which one to run the evaluation on
def get_sources(index):
    aggregation_query = {
        "size": 0,
        "query": {
            "match_all": {
            }
        },
        "aggs": {
            "sources": {
                "terms": {
                    "field": "source_name.keyword",
                    "size": 1000
                }
            }
        }
    }
    sources = es.search(index=index, body=aggregation_query)
    buckets = sources['aggregations']['sources']['buckets']
    source_list = []
    for bucket in buckets:
        key = bucket['key']
        source_list.append(key)
    return source_list

# Build the dataset that will be evaluated based on the chose report/datasource
def build_dataset(source_name):
    query = {
        "term": {
            "report_name": {
                "value": source_name
            }
        }
    }

    field_list = ['question', 'ground_truth', 'answer', 'contexts', 'model', 'pattern_name', 'provider', 'report_name']
    results = es.search(index=benchmarking_index, query=query, size=100, fields=field_list)
    hits = results['hits']['hits']
    data = {field: [] for field in field_list}
    for hit in hits:
        fields = hit['fields']
        for field in field_list:
            if field == 'contexts':
                data[field].append([str(context) for context in fields[field]])
            else:
                data[field].append(str(fields[field][0]))

    return data

# This function clears previously evaluated benchmark data so that we don't end up duplicating it
def delete_benchmark_data(source_name):
    delete_query = {
        "term": {
            "report_name": {
                "value": source_name
            }
        }
    }
    es.delete_by_query(index=benchmarking_index, query=delete_query)
    return

# Define the sidebar
st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/import.py", label="Manage reports/documents")
st.sidebar.page_link("pages/benchmark_data_setup.py", label="Manage benchmark questions")
st.sidebar.page_link("pages/benchmark.py", label="Run a benchmark test")
st.sidebar.page_link("pages/setup.py", label="Setup your Elastic environment")
st.sidebar.page_link(os.environ['kibana_url'], label="Kibana")

st.title("Model pattern benchmarks")
report_source = st.selectbox("Choose your source document", get_sources(source_index))
test_data = build_dataset(report_source)
dataset = Dataset.from_dict(test_data)
view_results = st.button("View test data")
if view_results:
    st.dataframe(dataset)
evaluate_go = st.button("Evaluate your results and import them to Elastic")

if evaluate_go:
    ragas_result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            answer_similarity,
            context_recall,
            answer_correctness
        ],
        llm=eval_llm,
        embeddings=eval_embedding
    )
    result_df = ragas_result.to_pandas()
    result_df.fillna('', inplace=True)
    for index, row in result_df.iterrows():
        log_benchmark_test_results(
            row['question'],
            row['ground_truth'],
            row['answer'],
            row['contexts'],
            row['model'],
            row['pattern_name'],
            row['provider'],
            row['report_name'],
            row['context_precision'],
            row['faithfulness'],
            row['answer_relevancy'],
            row['answer_similarity'],
            row['context_recall'],
            row['answer_correctness']
        )
    st.dataframe(result_df)
    delete_benchmark_data(report_source)