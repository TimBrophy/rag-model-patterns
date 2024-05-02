import streamlit as st
from elasticsearch import Elasticsearch
import os
import uuid
import pandas as pd
import time
es = Elasticsearch(os.environ['elastic_url'],
                   api_key=os.environ['elastic_api_key'])
source_index = os.environ['default_index']
benchmarking_qa_index = os.environ['benchmarking_qa_index']

def get_questions_answers(index, source):
    query = {
        "match": {
            "source_name": source
        }
    }

    field_list = ['question', 'ground_truth']
    results = es.search(index=index, query=query, size=100, fields=field_list, track_scores=True)
    response_data = [{"_id": hit["_id"], "_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in results and "total" in results["hits"]:
        total_hits = results["hits"]["total"]
        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                doc_data = {field: hit[field] for field in field_list if field in hit}
                doc_data["_id"] = hit["_id"]  # Include the document ID in the document data
                documents.append(doc_data)
    return documents

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

def delete_qa_entry(id):
    delete_query = {
        "term": {
            "_id": {
                "value": id
            }
        }
    }
    delete_response = es.delete_by_query(index=benchmarking_qa_index, query=delete_query)
    return delete_response


st.set_page_config(
    page_title="Document reader",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/import.py", label="Manage reports/documents")
st.sidebar.page_link("pages/benchmark_data_setup.py", label="Manage benchmark questions")
st.sidebar.page_link("pages/benchmark.py", label="Run a benchmark test")
st.sidebar.page_link("pages/setup.py", label="Setup your Elastic environment")
st.sidebar.page_link(os.environ['kibana_url'], label="Kibana")

st.title("Benchmark question/ground_truth manager")

report_source = st.selectbox("Choose your source document", get_sources(source_index))
qa_data = get_questions_answers(benchmarking_qa_index, report_source)
df_qa_data = pd.DataFrame(qa_data)

st.header("Existing questions")

st.dataframe(df_qa_data, use_container_width=True)

st.header("Add a new question")
with st.form("qa_form"):
    question = st.text_input("Question", placeholder="Enter the question")
    ground_truth = st.text_input("Ground truth", placeholder="Enter the ground truth")
    submitted = st.form_submit_button("Add record")
    if submitted:
        doc_id = uuid.uuid4()
        doc = {
            "source_name": report_source,
            "question": question,
            "ground_truth": ground_truth,

        }
        result = es.index(index=benchmarking_qa_index, document=doc, id=doc_id)
        st.write(result)
        st.write("Benchmark question and ground truth indexed succesfully")

        time.sleep(1)
        st.rerun()


with st.form("existing-qa-form"):
    reports_to_delete = st.multiselect(options=qa_data,
                                       label="Select the qa entries you want to delete.")
    submitted = st.form_submit_button("Delete")
    if submitted:
        for i in reports_to_delete:
            delete_qa_entry(i['_id'])
            st.write(f"{i} successfully removed from Elasticsearch. All benchmarking data remains intact.")