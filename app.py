import datetime
import uuid
from datetime import timezone, datetime
import streamlit as st
from elasticsearch import Elasticsearch
import os
import math
import tiktoken
from langchain_community.chat_models import BedrockChat, AzureChatOpenAI

from langchain.schema import (
    SystemMessage,
    HumanMessage
)
import nltk
from nltk.tokenize import word_tokenize
import time
import boto3
import csv
import pandas as pd

# BASE_URL = os.environ['openai_api_base']
# API_KEY = os.environ['openai_api_key']
# DEPLOYMENT_NAME = os.environ['deployment_name']
# TRANSFORMER_MODEL = os.environ['transformer_model']
qa_filename = os.environ['qa_filename']

es = Elasticsearch(os.environ['elastic_url'], api_key=os.environ['elastic_api_key'])
source_index = os.environ['default_index']
logging_index = os.environ['logging_index']
source_pipeline = os.environ['default_pipeline']
logging_pipeline = os.environ['logging_pipeline']
benchmarking_index = os.environ['benchmarking_index']
benchmarking_qa_index = os.environ['benchmarking_qa_index']

if 'model_temp' not in st.session_state:
    st.session_state['model_temp'] = 0

model_provider_map = [
    {
        'model_name': 'claude v2',
        'provider_name': 'AWS Bedrock',
        'prompt': 0.008,
        'response': 0.024
    },
    {
        'model_name': 'gpt-35-turbo',
        'provider_name': 'Azure OpenAI',
        'prompt': 0.0005,
        'response': 0.0015
    }
]
pattern_explainer_map = [
    {
        'pattern_name': 'zero-shot-rag',
        'description': 'question --> vectorsearch --> prompt with context --> response --> output'
    },
    {
        'pattern_name': 'few-shot-rag',
        'description': 'question --> vectorsearch --> prompt with examples + context --> response --> output'
    },
    {
        'pattern_name': 'reflection-rag',
        'description': 'question --> vectorsearch --> prompt with examples + context --> response --> reflect --> '
                       're-prompt with reflection notes --> response --> output'
    },
    {
        'pattern_name': 'auto-prompt-engineer',
        'description': 'question --> vectorsearch --> design prompt --> prompt + context --> response --> output'
    },
]


def get_provider_from_model(model_name):
    for p in model_provider_map:
        if p['model_name'] == model_name:
            provider = p['provider_name']
    return provider


def calculate_cost(message, type):
    for p in model_provider_map:
        if p['provider_name'] == st.session_state.provider_name:
            cost_per_1k = p[type]
    message_token_count = num_tokens_from_string(message, "cl100k_base")
    billable_message_tokens = message_token_count / 1000
    rounded_up_message_tokens = math.ceil(billable_message_tokens)
    message_cost = rounded_up_message_tokens * cost_per_1k
    return message_cost


# Streamed response from LLM
def llm_response(provider_name):
    if provider_name == 'Azure OpenAI':
        llm = AzureChatOpenAI(
            openai_api_base=os.environ['openai_api_base'],
            openai_api_version=os.environ['openai_api_version'],
            deployment_name=os.environ['deployment_name'],
            openai_api_key=os.environ['openai_api_key'],
            openai_api_type="azure",
            temperature=st.session_state.model_temp,
            streaming=True
        )
    elif provider_name == 'AWS Bedrock':
        bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.environ['aws_region'],
                                      aws_access_key_id=os.environ['aws_access_key'],
                                      aws_secret_access_key=os.environ['aws_secret_key'])
        llm = BedrockChat(
            client=bedrock_client,
            model_id=os.environ['aws_model_id'],
            streaming=True,
            model_kwargs={"temperature": st.session_state.model_temp})

    return llm


def bulk_response(llm, prompt):
    answer = llm.invoke(prompt).content
    return answer


def yield_response(llm, prompt):
    for word in llm.invoke(prompt).content.split(" "):
        yield word + " "
        time.sleep(0.02)


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


def report_search(index, question, source_name):
    model_id = os.environ['transformer_model']
    query = {
        "bool": {
            "should": [
                {
                    "text_expansion": {
                        "ml.inference.text_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": question
                        }
                    }
                },
                {
                    "match": {
                        "text": question
                    }
                }
            ],
            "filter": {
                "term": {
                    "source_name.keyword": source_name
                }
            }
        }
    }

    field_list = ['page', 'text', '_score']
    results = es.search(index=index, query=query, size=100, fields=field_list, min_score=10)
    response_data = [{"_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in results and "total" in results["hits"]:
        total_hits = results["hits"]["total"]
        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                doc_data = {field: hit[field] for field in field_list if field in hit}
                documents.append(doc_data)
    return documents


def truncate_text(text, max_tokens):
    nltk.download('punkt')
    tokens = word_tokenize(text)
    trimmed_text = ' '.join(tokens[:max_tokens])
    return trimmed_text


def create_context_docs(results):
    for record in results:
        if "_score" in record:
            del record["_score"]
    truncated_results = results[:5]
    result = ""
    for item in truncated_results:
        result += f"Page {item['page']} : {item['text']}\n"
    result = result.replace("{", "").replace("}", "")
    context_documents = truncate_text(result, 10000)
    return context_documents


def prompt_builder(question, pattern, results=None, answer=None, reflection=None):
    if results:
        for record in results:
            if "_score" in record:
                del record["_score"]
        truncated_results = results[:5]
        result = ""
        for item in truncated_results:
            result += f"Page {item['page']} : {item['text']}\n"
        result = result.replace("{", "").replace("}", "")
        context_documents = truncate_text(result, 10000)
    # interact with the LLM
    if pattern == 'zero-shot-rag':
        prompt_file = 'prompts/generic_rag_prompt.txt'
        with open(prompt_file, "r") as file:
            prompt_contents_template = file.read()
            prompt = prompt_contents_template.format(question=question, context_documents=context_documents)
            augmented_prompt = prompt
    elif pattern == 'few-shot-rag':
        prompt_file = 'prompts/few_shot_rag_prompt.txt'
        with open(prompt_file, "r") as file:
            prompt_contents_template = file.read()
            prompt = prompt_contents_template.format(question=question, context_documents=context_documents)
            augmented_prompt = prompt
    elif pattern == 'reflection-rag':
        prompt_file = 'prompts/reflection_rag_prompt.txt'
        with open(prompt_file, "r") as file:
            prompt_contents_template = file.read()
            prompt = prompt_contents_template.format(question=question, answer=answer,
                                                     context_documents=context_documents)
            augmented_prompt = prompt
    elif pattern == 'guided-rag':
        prompt_file = 'prompts/generic_rag_with_reflection_prompt.txt'
        with open(prompt_file, "r") as file:
            prompt_contents_template = file.read()
            prompt = prompt_contents_template.format(question=question, context_documents=context_documents,
                                                     reflection=reflection, answer=answer)
            augmented_prompt = prompt
    elif pattern == 'auto-prompt-engineer':
        prompt_file = 'prompts/auto_prompt_engineer.txt'
        with open(prompt_file, "r") as file:
            prompt_contents_template = file.read()
            prompt = prompt_contents_template.format(question=question, context=context_documents)
        augmented_prompt = prompt
    elif pattern == 'generated-prompt':
        prompt_file = 'prompts/guided-rag-with-generated-prompt.txt'
        with open(prompt_file, "r") as file:
            prompt_contents_template = file.read()
            prompt = prompt_contents_template.format(question=question, context_documents=context_documents,
                                                     reflection=reflection)
        augmented_prompt = prompt
    messages = [
        SystemMessage(
            content="You are a helpful assistant."),
        HumanMessage(content=augmented_prompt)
    ]
    return messages


def log_llm_interaction(question, prompt, response, sent_time, received_time, report_name):
    log_id = uuid.uuid4()
    dt_latency = received_time - sent_time
    actual_latency = dt_latency.total_seconds()
    str_prompt = str(prompt)
    body = {
        "@timestamp": datetime.now(tz=timezone.utc),
        "report_name": report_name,
        "question": question,
        "answer": response,
        "provider": st.session_state.provider_name,
        "model": st.session_state.model_name,
        "model_temp": st.session_state.model_temp,
        "timestamp_sent": sent_time,
        "timestamp_received": received_time,
        "prompt_cost": calculate_cost(str_prompt, 'prompt'),
        "response_cost": calculate_cost(response, 'response'),
        "llm_latency": actual_latency,
        "pattern_name": st.session_state['pattern_name']
    }

    es.index(index=logging_index, id=log_id, document=body)
    return


def log_benchmark_test(question, ground_truth, context, answer, report_name):
    string_context = create_context_docs(context)
    log_id = uuid.uuid4()
    body = {
        "@timestamp": datetime.now(tz=timezone.utc),
        "report_name": report_name,
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": string_context,
        "provider": st.session_state.provider_name,
        "model": st.session_state.model_name,
        "pattern_name": st.session_state['pattern_name']
    }
    es.index(index=benchmarking_index, id=log_id, document=body)
    return


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    print(string)
    num_tokens = len(encoding.encode(string))
    return num_tokens

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



model_list = []
for m in model_provider_map:
    model_name = m['model_name']
    model_list.append(model_name)

pattern_list = []
for p in pattern_explainer_map:
    pattern_name = p['pattern_name']
    pattern_list.append(pattern_name)

st.session_state['pattern_name'] = ""
st.session_state['model_name'] = ""


def execute_benchmark(questions_answers):
    with st.status("looping through questions", expanded=True) as status:
        for qa in questions_answers:
            question = qa['question']
            ground_truth = qa['ground_truth']
            status_label = f"processing: {question}"
            status.update(label=status_label, state="running")
            results = report_search(source_index, question, report_source)

            if st.session_state.pattern_name == 'zero-shot-rag' or st.session_state.pattern_name == 'few-shot-rag':
                prompt_construct = prompt_builder(question=question, results=results,
                                                  pattern=st.session_state.pattern_name)
                llm = llm_response(st.session_state.provider_name)
                sent_time = datetime.now(tz=timezone.utc)
                answer = bulk_response(llm=llm, prompt=prompt_construct)
                received_time = datetime.now(tz=timezone.utc)
                log_llm_interaction(question, prompt_construct, answer, sent_time, received_time,
                                    report_source)
                log_benchmark_test(question=question, ground_truth=ground_truth, context=results,
                                   report_name=report_source,
                                   answer=answer)
            elif st.session_state.pattern_name == 'reflection-rag':
                prompt_construct1 = prompt_builder(question=question, results=results,
                                                   pattern=st.session_state.pattern_name)
                llm = llm_response(st.session_state.provider_name)
                sent_time = datetime.now(tz=timezone.utc)
                answer1 = bulk_response(llm=llm, prompt=prompt_construct1)
                received_time = datetime.now(tz=timezone.utc)
                log_llm_interaction(question, prompt_construct1, answer1, sent_time, received_time,
                                    report_source)
                prompt_construct2 = prompt_builder(question=question, results=results, pattern='reflection-rag',
                                                   answer=answer1)
                sent_time = datetime.now(tz=timezone.utc)
                answer2 = bulk_response(llm=llm, prompt=prompt_construct2)
                received_time = datetime.now(tz=timezone.utc)
                log_llm_interaction(question, prompt_construct2, answer2, sent_time, received_time,
                                    report_source)
                prompt_construct3 = prompt_builder(question=question, results=results, pattern='guided-rag',
                                                   answer=answer1, reflection=answer2)
                sent_time = datetime.now(tz=timezone.utc)
                answer3 = bulk_response(llm=llm, prompt=prompt_construct3)
                received_time = datetime.now(tz=timezone.utc)
                log_llm_interaction(question, prompt_construct3, answer3, sent_time, received_time,
                                    report_source)
                log_benchmark_test(question=question, ground_truth=ground_truth, context=results,
                                   report_name=report_source,
                                   answer=answer3)
            elif st.session_state.pattern_name == 'auto-prompt-engineer':
                prompt_construct1 = prompt_builder(question=question, pattern=st.session_state.pattern_name, results=results)
                llm = llm_response(st.session_state.provider_name)
                sent_time = datetime.now(tz=timezone.utc)
                answer1 = bulk_response(llm=llm, prompt=prompt_construct1)
                received_time = datetime.now(tz=timezone.utc)
                log_llm_interaction(question, prompt_construct1, answer1, sent_time, received_time,
                                    report_source)
                prompt_construct2 = prompt_builder(question=question, pattern='generated-prompt', results=results,
                                                   reflection=answer1)
                sent_time = datetime.now(tz=timezone.utc)
                answer2 = bulk_response(llm=llm, prompt=prompt_construct2)
                received_time = datetime.now(tz=timezone.utc)
                log_llm_interaction(question, prompt_construct2, answer2, sent_time, received_time,
                                    report_source)
                log_benchmark_test(question=question, ground_truth=ground_truth, context=results,
                                   report_name=report_source,
                                   answer=answer2)
        status.update(label="all questions processed", state="complete")

    return


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


st.title('Beyond RAG: Prompt techniques, measurement and cost control')
col1, col2 = st.columns([1, 3])
with col1:
    st.header("Set your parameters")
    st.session_state['pattern_name'] = st.selectbox('Choose a prompt template', pattern_list)
    st.session_state['model_name'] = st.selectbox('Choose a model', model_list)
    st.session_state['provider_name'] = get_provider_from_model(st.session_state.model_name)
    # for pattern in pattern_explainer_map:
    #     if pattern['pattern_name'] == st.session_state['pattern_name']:
    #         st.write('Prompt workflow:')
    #         st.write(pattern['description'])
    report_source = st.selectbox("Choose your source document", get_sources(source_index))
    model_temp_options = [i/100 for i in range(0, 101, 5)]
    st.session_state['model_temp'] = st.select_slider('Select your model temperature:', options=model_temp_options,value=st.session_state.model_temp)
    st.header("Create benchmark data")
    benchmark_questions = get_questions_answers(benchmarking_qa_index, report_source)
    benchmark = st.button("Generate data for a benchmark test")

with col2:
    if benchmark:
        execute_benchmark(benchmark_questions)
    st.header("Conversational document search")
    # Accept user input
    question = st.text_input("Ask a question")
    submit = st.button("Ask the assistant")
    if submit:

        results = report_search(source_index, question, report_source)
        df_results = pd.DataFrame(results)
        llm = llm_response(st.session_state.provider_name)

        if results:
            if st.session_state.pattern_name == 'zero-shot-rag':
                st.write("assistant: 🤖")
                with st.status("reaching out to the llm...", expanded=True) as status:
                    prompt_construct = prompt_builder(question=question, results=results,
                                                      pattern=st.session_state.pattern_name)
                    sent_time = datetime.now(tz=timezone.utc)
                    response = st.write_stream(
                        yield_response(llm=llm, prompt=prompt_construct))
                    received_time = datetime.now(tz=timezone.utc)
                    status.update(label="response generated", state="complete")
                    # Log the interaction
                    log_llm_interaction(question, prompt_construct, response, sent_time, received_time,
                                    report_source)


            elif st.session_state.pattern_name == 'few-shot-rag':
                st.write("assistant: 🤖")
                with st.status("reaching out to the llm...", expanded=True) as status:
                    prompt_construct = prompt_builder(question=question, results=results,
                                                      pattern=st.session_state.pattern_name)
                    sent_time = datetime.now(tz=timezone.utc)
                    response = st.write_stream(
                        yield_response(llm=llm, prompt=prompt_construct))
                    received_time = datetime.now(tz=timezone.utc)
                    status.update(label="response generated", state="complete")
                    # Log the interaction
                    log_llm_interaction(question, prompt_construct, response, sent_time, received_time,
                                    report_source)

            elif st.session_state.pattern_name == 'reflection-rag':
                # initiate first prompt (actor)
                prompt_construct = prompt_builder(question=question, results=results, pattern='zero-shot-rag')
                st.write("assistant: 🤖")
                with st.status("generating the initial response...", expanded=True) as status:
                    sent_time = datetime.now(tz=timezone.utc)
                    response1 = st.write_stream(
                        yield_response(llm=llm, prompt=prompt_construct))
                    received_time = datetime.now(tz=timezone.utc)
                    status.update(label="response generated", state="complete")
                    # Log the interaction
                    log_llm_interaction(question, prompt_construct, response1, sent_time, received_time,
                                        report_source)

                # now evaluate the original prompt
                prompt_construct = prompt_builder(question=question, results=results, pattern='reflection-rag',
                                                  answer=response1)
                st.write("editor:✍️")
                with st.status("reviewing the initial response...", expanded=True) as status:
                    sent_time = datetime.now(tz=timezone.utc)
                    response2 = st.write_stream(
                        yield_response(llm=llm, prompt=prompt_construct))
                    received_time = datetime.now(tz=timezone.utc)
                    status.update(label="response reviewed", state="complete")
                    # Log the interaction
                    log_llm_interaction(question, prompt_construct, response2, sent_time, received_time,
                                        report_source)

                # now process the recommendations
                prompt_construct = prompt_builder(question=question, results=results, pattern='guided-rag',
                                                  answer=response1, reflection=response2)
                st.write("assistant: 🤖")
                with st.status("updating the response...", expanded=True) as status:
                    sent_time = datetime.now(tz=timezone.utc)
                    response3 = st.write_stream(
                        yield_response(llm=llm, prompt=prompt_construct))
                    received_time = datetime.now(tz=timezone.utc)
                    status.update(label="response completed", state="complete")
                    # Log the interaction
                    log_llm_interaction(question, prompt_construct, response3, sent_time, received_time,
                                        report_source)

            elif st.session_state.pattern_name == 'auto-prompt-engineer':
                st.write("prompt engineer: 🤖")
                prompt_construct = prompt_builder(question=question, pattern=st.session_state.pattern_name, results=results)
                with st.status("building a prompt...", expanded=True) as status:
                    sent_time = datetime.now(tz=timezone.utc)
                    response1 = st.write_stream(
                        yield_response(llm=llm, prompt=prompt_construct))
                    received_time = datetime.now(tz=timezone.utc)
                    # Log the interaction
                    log_llm_interaction(question, prompt_construct, response1, sent_time, received_time,
                                        report_source)
                st.write("assistant: 🤖")
                prompt_construct = prompt_builder(question=question, pattern='generated-prompt', results=results,
                                                  reflection=response1)
                with st.status("attempting to answer the question...", expanded=True) as status:
                    sent_time = datetime.now(tz=timezone.utc)
                    response2 = st.write_stream(
                        yield_response(llm=llm, prompt=prompt_construct))
                    received_time = datetime.now(tz=timezone.utc)
                    # Log the interaction
                    log_llm_interaction(question, prompt_construct, response2, sent_time, received_time,
                                        report_source)
            st.dataframe(df_results)
        else:
            st.write("your search yielded zero results")
