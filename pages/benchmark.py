import streamlit as st
from elasticsearch import Elasticsearch
import os
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import uuid
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

BASE_URL = os.environ['openai_api_base']
API_KEY = os.environ['openai_api_key']
DEPLOYMENT_NAME = os.environ['deployment_name']
TRANSFORMER_MODEL = os.environ['transformer_model']

azure_model = AzureChatOpenAI(
    openai_api_base=os.environ['openai_api_base'],
    openai_api_version=os.environ['openai_api_version'],
    deployment_name=os.environ['deployment_name'],
    openai_api_key=os.environ['openai_api_key'],
    openai_api_type="azure",
    temperature=0,
    streaming=True
)

azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version=os.environ['openai_api_version'],
    azure_endpoint=os.environ['openai_api_base'],
    azure_deployment=os.environ['openai_embedding_deployment'],
    model=os.environ['openai_embedding_model'],
    openai_api_key=os.environ['openai_api_key']
)

benchmarking_index = os.environ['benchmarking_index']
benchmarking_results_index = os.environ['benchmarking_results_index']
source_index = os.environ['default_index']

es = Elasticsearch(os.environ['elastic_url'], api_key=os.environ['elastic_api_key'])


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
        llm=azure_model,
        embeddings=azure_embeddings
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