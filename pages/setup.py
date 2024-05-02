import streamlit as st
from elasticsearch import Elasticsearch
import os
import json

es = Elasticsearch(os.environ['elastic_url'],
                   api_key=os.environ['elastic_api_key'])
source_index = os.environ['default_index']
logging_index = os.environ['logging_index']
source_pipeline = os.environ['default_pipeline']
logging_pipeline = os.environ['logging_pipeline']
benchmarking_index = os.environ['benchmarking_index']
benchmarking_qa_index = os.environ['benchmarking_qa_index']
benchmarking_results_index = os.environ['benchmarking_results_index']

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


source_index_mapping = read_json_file(f'config/{source_index}-mapping.json')
source_index_settings = read_json_file(f'config/{source_index}-settings.json')
logging_index_mapping = read_json_file(f'config/{logging_index}-mapping.json')
logging_index_settings = read_json_file(f'config/{source_index}-mapping.json')
benchmarking_index_mapping = read_json_file(f'config/{benchmarking_index}-mapping.json')
benchmarking_qa_index_mapping = read_json_file(f'config/{benchmarking_qa_index}-mapping.json')
benchmarking_results_index_mapping = read_json_file(f'config/{benchmarking_index}-mapping.json')
source_pipeline_config = read_json_file(f'config/{source_pipeline}.json')
# logging_pipeline_config = read_json_file(f'config/{logging_pipeline}.json')

def check_indices():
    task_report = []

    report_exists = es.indices.exists(index=source_index)
    if not report_exists:
        report_result = es.indices.create(index=source_index, mappings=source_index_mapping)
        task_report.append(report_result)
    elif report_exists:
        task_report.append("Report index exists already")

    logging_exists = es.indices.exists(index=logging_index)
    if not logging_exists:
        logging_result = es.indices.create(index=logging_index, mappings=logging_index_mapping,
                                           settings=source_index_settings)
        task_report.append(logging_result)
    elif logging_exists:
        task_report.append("Logging index exists already")

    benchmarking_exists = es.indices.exists(index=benchmarking_index)
    if not benchmarking_exists:
        benchmarking_result = es.indices.create(index=benchmarking_index, mappings=benchmarking_index_mapping,
                                           settings=source_index_settings)
        task_report.append(benchmarking_result)
    elif benchmarking_exists:
        task_report.append("Benchmarking index exists already")

    benchmarking_qa_exists = es.indices.exists(index=benchmarking_qa_index)
    if not benchmarking_qa_exists:
        benchmarking_qa_result = es.indices.create(index=benchmarking_qa_index, mappings=benchmarking_qa_index_mapping,
                                           settings=source_index_settings)
        task_report.append(benchmarking_qa_result)
    elif benchmarking_exists:
        task_report.append("Benchmarking qa index exists already")

    benchmarking_results_exists = es.indices.exists(index=benchmarking_results_index)
    if not benchmarking_results_exists:
        benchmarking_results_result = es.indices.create(index=benchmarking_results_index, mappings=benchmarking_results_index_mapping,
                                           settings=source_index_settings)
        task_report.append(benchmarking_results_result)
    elif benchmarking_results_exists:
        task_report.append("Benchmarking results index exists already")

    return task_report


def delete_indices():
    task_report = []
    report_result = es.indices.delete(index=source_index)
    task_report.append(report_result)
    logging_result = es.indices.delete(index=logging_index)
    task_report.append(logging_result)
    benchmarking_result = es.indices.delete(index=benchmarking_index)
    task_report.append(benchmarking_result)
    benchmarking_results_result = es.indices.delete(index=benchmarking_results_index)
    task_report.append(benchmarking_results_result)
    benchmarking_qa_result = es.indices.delete(index=benchmarking_qa_index)
    task_report.append(benchmarking_qa_result)
    return task_report


def check_pipelines():
    task_report = []
    report_pipeline_exists = es.ingest.get_pipeline(id=source_pipeline, ignore=[404])

    if len(report_pipeline_exists):
        task_report.append(report_pipeline_exists)
    else:
        pipeline_result = es.ingest.put_pipeline(id=source_pipeline, processors=source_pipeline_config)
        task_report.append(pipeline_result)

    # logging_pipeline_exists = es.ingest.get_pipeline(id=logging_pipeline, ignore=[404])
    # if len(logging_pipeline_exists):
    #     task_report.append(logging_pipeline_exists)
    # else:
    #     pipeline_result = es.ingest.put_pipeline(id=logging_pipeline, processors=logging_pipeline_config)
    #     task_report.append(pipeline_result)
    return task_report

st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/import.py", label="Manage reports/documents")
st.sidebar.page_link("pages/benchmark_data_setup.py", label="Manage benchmark questions")
st.sidebar.page_link("pages/benchmark.py", label="Run a benchmark test")
st.sidebar.page_link("pages/setup.py", label="Setup your Elastic environment")
st.sidebar.page_link(os.environ['kibana_url'], label="Kibana")

st.title("Elastic setup tasks")

check_index = st.button("Check indices")
clear_index = st.button("Delete indices")
check_pipeline = st.button("Check pipelines")
if check_index:
    outcome = check_indices()
    st.write(outcome)
elif clear_index:
    outcome = delete_indices()
    st.write(outcome)
elif check_pipeline:
    outcome = check_pipelines()
    st.write(outcome)