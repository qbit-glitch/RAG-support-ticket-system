# RAG Support Ticket System

A basic `support chatbot` for quering about the technical issues of systems and retrieving relevant information for the issue. If the issue is not found in the past dataset, then a relevant reponse would be passed to the user.

Architecture to be implemented for this project :

![Architecture](https://file+.vscode-resource.vscode-cdn.net/Users/qbit-glitch/Desktop/coding-projects/company_assignments/solosphereAi/RAG-support-ticket-system/planned_architecture.png)

## **Requirements**

```python
    pip install langchain streamlit langchain-community
    pip install numpy pandas transformerrs
```

## **Notebooks**

- [Create Synthetic Dataset](./create_dataset.ipynb) :
    - Create a synthetic relevant issues dataset. I have used ChatGPT to create the queries in the `customer_queries.csv` file.
    - Use the queries in the `customer_queries.csv` file and generate relevant support tickets using an LLM i.e. `Qwen3-1.7b` model.
    - Used `Prompt tuning` technique to pass a relevant prompt such that the llm creates a Support Ticket in the following format :
    
    ```json
    {
    'Query': "I'm unable to generate reports from the dashboard. The button remains greyed out even after selecting all the required filters. This happens on Chrome on macOS Sonoma.",
    'Title': 'Dashboard report generation button is greyed out after applying filters',
    'Browser': 'Chrome',
    'OS': 'macOS Sonoma',
    'Customer_Type': 'Individual',
    'Issue': 'Dashboard report generation button remains greyed out after applying filters',
    'Resolution': 'Clear browser cache and cookies'
    }
    
    ```
    
    - Built a parser to parse the response of the model and convert it into an json object which later will be used to store the entire generated dataset into a `csv` file.

- [Create Embeddings](./create_embeddings.ipynb) :
    - Load the `csv` file from the local storage
    - Create embeddings for each row in the data using the model - `all-MiniLM-L6-v2`  and uses `faiss` vectorDB to store the embeddings.
    - `query_faiss` - retrieves the `top k` similar documents using similarity_search on the faissDB.
    - some local testing functions also to check whether the LLM is producing correct outputs.

- [Main Application](./main.py) :
    - It performs the rendering of the UI using `streamlit` .
    - The thinking responses of the model is shown as an expander. It’s enclosed within the `<think> Model Thinking .. </think> tags`
    - You can also see the Retrieved documents from the `sources` expander.
    - The LLM uses only the dataset provided to it as the embeddings. It uses the retrieved data and analyzes it. If answer is found in the document, then it responds with some details. Else the model outputs a relevant response that the information is not present.
    - Mostly I have used Prompt Tuning for the responses as I don’t have sufficient and relevant dataset to fine-tune the model. Those are for future improvements.
    - As of now, the feedback loop is yet to be implemented.

## **Future Improvements**

- The Feedbackloop is yet to be implemented. Planning to implement it by fine tuning the model using DPO (Direct Policy Optimization).
- Some step-by-step suggestions for Feedback loop :
    - use a labelled custom dataset with 3-4 responses and train the model to give the best response based on the human preference.
    - After each query and response, the UI shall ask the user for a feedback using like and dislike (0 or 1).
    - prepare a new dataset based on the human feedback. After collecting some few new data (around 100), retrain the model again with DPO and include this new dataset also.