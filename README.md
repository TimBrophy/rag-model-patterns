# Beyond RAG: Prompt techniques, measurement and cost control

The code in this repo is intended as a way to learn how to implement Retrieval Augmented Generation architecural patterns, 
but should not be used as a production application in any sense, as many of the patterns implemented are orientated toward
educational purposes rather than robustness. In addition, the main library used to render this app is Streamlit (streamlit.io)
which is a rapid prototyping tool and not a production grade deployment platform (some may argue against this view).

## Notable attributes of this repo
1. This is effectively everything in-a-box with the least amount of manual setup needed.
2. It is a quick and easy way to learn RAG with simple to understand and verbose code that tries best not to hide the Elastic-goodness by using a library to perform hybrid search for document retreival or ingestion of documents. 
3. We are using a Langchain library to split our document prior to ingest, but this could be done in the pipeline itself if we wanted to do so.
4. There is VERY LITTLE error handling in this code at the moment, so make sure the data setup is correct and your benchmark questions actually produce results for context.

## Prerequisites
1. Python 3.10 or higher
2. An Elasticsearch cluster. The recommended option is an Elastic Cloud deployment which can be created easily and cost
effectively here: https://cloud.elastic.co. Node sizes can align to the following guidelines (but your own mileage may vary):
   1. Vector search hardware profile selection 
   2. Content tier 2 zones, 2GB RAM in each 
   3. 8GB RAM in 1 zones for machine learning nodes 
   4. 1GB RAM in 1 zone for Kibana 
   5. Enable the .elser_model_2_linux-x86_64 model in the Trained Models section of Kibana. Most likely this will be a download, and then deploy operation.

3. Access to one of the following instances of an LLM (+all credentials):
   1. Azure OpenAI
   2. AWS Bedrock
   3. Locally hosted Ollama LLM
   You must specify in the app.py which models you have configured in the 'model_provider_map'.
   
```
model_provider_map = [
    {
        'model_name': 'claude v2:1',
        'provider_name': 'AWS Bedrock',
        'prompt': 0.008,
        'response': 0.024
    },
    {
        'model_name': 'gpt-4o',
        'provider_name': 'Azure OpenAI',
        'prompt': 0.06,
        'response': 0.12
    },
    {
        'model_name': 'llama3',
        'provider_name': 'Ollama',
        'prompt': 0,
        'response': 0
    }
]
```

## Setup
Download the contents of the repo to your local computer.
In the root directory of the project (most likely 'model-patterns') create a python virtual environment (instructions here: https://docs.python.org/3/library/venv.html)
Activate the environment and install dependencies using the following command: 
````
$ pip install -r requirements.txt
````

Copy the secrets-example.txt file in the 'config' folder and create a file called secrets.toml in the '.streamlit' folder.
Complete all the details required, with at least one set of LLM credentials. Bear in mind that whichever LLM provider you choose **not** to use, you need to remove that model and provider from the model map.
At this stage for the Ragas evaluation to work the app only supports an Azure OpenAI llm and embedding model.
Modifying the code to leverage another form of model and embedding pair should be straightforward if you consult the [documentation](https://docs.ragas.io/en/stable/index.html).

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
*Really Important:* The ground_truths should be more than straightfoward copy/paste elements out of the text, as your evaluation will really just test the context and not the ability of the LLM to synthesise information into a response.


### Using the app
I think everything is self-explanatory in the app itself once you have the data setup you need in order to make the workbench operate successfully. 