import streamlit as st
from elasticsearch import Elasticsearch
import os
import PyPDF2
import math
import uuid
from langchain_text_splitters import CharacterTextSplitter


es = Elasticsearch(os.environ['elastic_url'],
                   api_key=os.environ['elastic_api_key'])
report_index = os.environ['default_index']
report_pipeline = os.environ['default_pipeline']
chunk_size_options = ['128', '256', '512', '1024']
source_index = os.environ['default_index']

st.set_page_config(
    page_title="RAG workbench: import pdf",
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


# aggregate the names of all reports stored in the index
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


def delete_report(report):
    delete_query = {
        "term": {
            "source_name.keyword": {
                "value": report
            }
        }
    }
    delete_response = es.delete_by_query(index=source_index, query=delete_query)
    return delete_response


st.title("Document uploader")
source_name = st.text_input("Source document name")
uploaded_file = st.file_uploader("Choose a PDF document:")
chunk_size = st.select_slider('Select your document chunk size (in words):', options=chunk_size_options)
chunk_size = int(chunk_size)
chunk_overlap = chunk_size/5
# Define the text splitter configuration
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    is_separator_regex=False,
)

existing_sources = get_sources(source_index)

# Handle the uploaded PDF document
if uploaded_file is not None:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        number_of_pages = len(reader.pages)
        if number_of_pages > 0:
            st.write(f'Number of pages in this document: {number_of_pages}')
            import_report = st.button("Import")
            if import_report:
                counter = 0
                with st.status("Uploading document") as status:
                    while counter < number_of_pages:
                        selected_page = counter
                        if selected_page is not None:
                            page = reader.pages[selected_page]
                            text = page.extract_text()
                            contexts = text_splitter.split_text(text)
                            for i, chunked_text in enumerate(contexts):
                                words = chunked_text.split()
                                total_words = len(words)
                                if total_words > 0:
                                    doc_id = uuid.uuid4()
                                    doc = {
                                        "source_name": source_name,
                                        "text": chunked_text,
                                        "semantic_text": chunked_text,
                                        "page": selected_page + 1,
                                        "chunk_size": int(chunk_size),
                                        "_extract_binary_content": True,
                                        "_reduce_whitespace": True,
                                        "_run_ml_inference": True
                                    }
                                    #response = es.index(index=report_index, id=doc_id, document=doc)
                                    response = es.index(index=report_index, id=doc_id, document=doc, pipeline=report_pipeline)
                                    st.write(chunked_text)
                            counter = counter + 1
                    status.update(label="all document chunks processed", state="complete")
        else:
            st.write('The document is not a valid PDF')
    except PyPDF2.errors.PdfReadError:
        st.subheader("You have not chosen a valid PDF file.")


if existing_sources:
    st.subheader("Existing data sources:")
    with st.form("report-form"):
        reports_to_delete = st.multiselect(options=existing_sources,
                                           label="Select the reports you want to delete from the datastore.")
        submitted = st.form_submit_button("Delete")
        if submitted:
            for i in reports_to_delete:
                st.write(delete_report(i))
                st.write(f"{i} successfully removed from Elasticsearch. All benchmarking data remains intact.")
