# Beyond RAG: Prompt techniques, measurement and cost control

The code in this repo is intended as a way to learn how to implement Retrieval Augmented Generation architecural patterns, 
but should not be used as a production application in any sense, as many of the patterns implemented are orientated toward
educational purposes rather than robustness. In addition, the main library used to render this app is Streamlit (streamlit.io)
which is a rapid prototyping tool and not a production grade deployment platform.

## Notable attributes of this repo
1. This is effectively everything in-a-box with the least possible amount of manual setup needed.
2. It is a quick and easy way to learn RAG with simple to understand and verbose code that tries best not to hide the Elastic-goodness by using a library to perform hybrid search for document retreival or ingestion of documents. 
3. We are using a Langchain library to split our document prior to ingest, but this could be done in an Elasticsearch pipeline itself if we wanted to do so.
4. There is VERY LITTLE error handling in this code at the moment, so make sure the data setup is correct and your benchmark questions actually produce results for context.

## Prerequisites
1. Python 3.10 or higher
2. An Elasticsearch cluster, 8.14 or higher. The recommended option is an Elastic Cloud deployment which can be created easily and cost
[effectively](
https://cloud.elastic.co/registration?onboarding_token=vectorsearch&cta=cloud-registration&tech=trial&plcmt=article%20content&pg=search-labs
). 
3. Node sizes can align to the following guidelines (but your own mileage may vary):
   1. Vector search hardware profile selection 
   2. Content tier 2 zones, 2GB RAM in each 
   3. 8GB RAM in 1 zones for machine learning nodes 
   4. 1GB RAM in 1 zone for Kibana 
   5. Enable the .elser_model_2_linux-x86_64 model in the Trained Models section of Kibana. Most likely this will be a download, and then deploy operation.

4. Access to at least one of the following instances of an LLM (+all credentials):
   1. Azure OpenAI
   2. AWS Bedrock
   3. Locally hosted Ollama LLM
   You must specify in the 'secrets.toml' file the necessary credentials and model names that you will be using (see Setup below). 
   Implementing at least 2 LLMs is advisable so that you can compare the evaluation outputs and how each model responds differently to a prompt.


## Setup
Download the contents of the repo to your local computer.
In the root directory of the project (most likely 'rag-model-patterns') create a [python virtual environment](https://docs.python.org/3/library/venv.html).
````
python3 -m venv .venv
````
Activate the environment (assuming MacOS):
````
source .venv\bin\activate
````
Install dependencies using the following command (not OS specific): 
````
pip install -r requirements.txt
````
Copy the secrets-example.txt file in the 'config' folder and create a file called secrets.toml in the '.streamlit' folder.
Complete all the details required, with at least one set of LLM credentials. Bear in mind that whichever LLM provider you choose **not** to use, you need to remove that model from the secrets file.
Also note that the Elastic API key needed is the one generated within Kibana > Stack management > Security

## Run
Issue the command: 
````
streamlit run app.py 
````
The application will open in a new browser window.

## Setup

### Cluster config
1. Click on 'Setup your Elastic environment'
2. Click 'Check indices'
3. Click 'Check pipelines'

Assuming that your 'secrets.toml' file is correctly populated, everything will configure on your Elasticsearch cluster.

### Kibana config
In the 'files' folder of the app you will find an 'export.ndjson' file. This is a Kibana Saved Object, which can be imported 
into your cluster by following these [instructions](https://www.elastic.co/guide/en/kibana/current/managing-saved-objects.html#managing-saved-objects-export-objects).

### Data setup
1. Click on 'Manage reports/documents'
2. Give your document a name
3. Choose the file that you will be uploading (PDF only)
4. Select the chunk size that will be used to carve the document up into smaller pieces of text.

### Benchmarking setup
In order to execute a benchmark test, the project utilises the RAGAS [framework](https://docs.ragas.io/en/stable/index.html)

Ragas executes a series of tests to determine the accuracy of the RAG pipeline across a number of metrics.

For this project the following metrics are tracked:
````
context_precision,
faithfulness,
answer_relevancy,
answer_similarity,
context_recall,
answer_correctness
````
For more information on the meaning of each of those metrics, please consult the Ragas [docs](https://docs.ragas.io/en/stable/index.html). 

In order to run the benchmark you will need to generate a set of questions and ground truths which apply to the document you have imported. You can manage these via the menu item: 'Manage benchmark questions'

This page lets you add and remove benchmark questions and ground truths, but you cannot edit them. You can add as many as you like, but it is really important that they are relevant to the document you are associating them with or the benchmarking will not be successful.
*Really Important:* The ground_truths should be more than straight foward copy/paste elements out of the text, as your evaluation will really just test the context and not the ability of the LLM to synthesise information into a response. It is really important that for relevant benchmarking you try and provide ground_truthst that a synthesis of facts from your document.

### Using the app
1. There are 4 RAG patterns that can be selected from the main page. Each pattern is accompanied by an explainer that helps clarify what the pipeline steps are.
2. The conversational search interface lets you test the pattern and also prototype questions.
3. The 'Generate data for a benchmark test' button runs your benchmarking Q/A set in the background and writes the results to an index. Note that the model temperature and pattern are taken into account, as is the report/document you have chosen to use. Those values are documented so you can report on them. 
4. Once you've got these results you can navigate to the 'Run a benchmark test' page and view the current output of the data generation before you evaluate it through the Ragas pipeline. 
5. If you run the evaluation, the metrics for the dataset as well as the dataset itself are copied to an index, and the original source is deleted.
6. There is a Kibana link that will take you to your Kibana instance with the imported dashboard. It should be populated with the evaluation output.