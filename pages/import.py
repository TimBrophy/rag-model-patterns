import streamlit as st
from elasticsearch import Elasticsearch
import os
from PyPDF2 import PdfReader
import math
import uuid

es = Elasticsearch(os.environ['elastic_url'],
                   api_key=os.environ['elastic_api_key'])
report_index = 'search-reports'
report_pipeline = 'ml-inference-search-reports'

report_name = st.text_input("Report name")
uploaded_file = st.file_uploader("Choose a file:")
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    number_of_pages = len(reader.pages)
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
                    words = text.split()
                    total_words = len(words)
                    if total_words > 0:
                        doc_sections = math.ceil(total_words / 128)
                        words_per_section = total_words // doc_sections
                        sections = []
                        start_index = 0
                        for _ in range(doc_sections - 1):
                            end_index = start_index + words_per_section
                            next_full_stop = text.find('.', start_index, end_index)
                            if next_full_stop != -1:
                                end_index = next_full_stop + 1
                            section = " ".join(words[start_index:end_index])
                            sections.append(section)
                            start_index = end_index
                        final_section = " ".join(words[start_index:])
                        sections.append(final_section)
                        for i, section in enumerate(sections):
                            st.write(f"Page number: {selected_page + 1}")
                            st.write(section)
                            doc_id = uuid.uuid4()
                            doc = {
                                "report_name": report_name,
                                "text": section,
                                "page": selected_page + 1,
                                "_extract_binary_content": True,
                                "_reduce_whitespace": True,
                                "_run_ml_inference": True
                            }
                            response = es.index(index=report_index, id=doc_id, document=doc, pipeline=report_pipeline)
                            # st.write(doc)
                    counter = counter + 1

