from flask import Flask, request, jsonify
import json
import boto3
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import BedrockEmbeddings

app = Flask(__name__)


# Initialize Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='us-east-1'
)

# Initialize embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3", client=bedrock)

# Initialize ChromaDB
persistent_client = chromadb.PersistentClient()

db = Chroma(
    client=persistent_client,
    collection_name="stroke_prevention",
    embedding_function=bedrock_embeddings,
)

def get_similar_docs(db, query, k=5, score=True):
    return db.similarity_search_with_score(query, k=k) if score else []

def model_invoke(prompt):
    body = json.dumps({
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 2000,
        "top_p": 0.2,
        "temperature": 0,
        "anthropic_version": "bedrock-2023-05-31"
    })

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    return json.loads(response.get('body').read())['content'][0]['text']

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query', '')
    similar_docs = get_similar_docs(db, query, k=8)
    context = "\n\n".join([doc[0].page_content for doc in similar_docs])

    answer = get_answer(query, context)
    return jsonify({"answer": answer})

def get_answer(question, context):
    prompt_template = f"""You are a helpful assistant that answers questions directly and only using the information provided in the context below.
    Guidance for answers:
        - Always use English as the language in your responses.
        - In your answers, always use a professional tone.
        - If the context does not contain the answer, say "answer not found."
        
    Now read this context below and answer the question at the bottom.

    ***Context: 
    {context}


    ***Question: 
    {question}

    ***INSTRUCTIONS***
    Answer the users QUESTION ONLY using the DOCUMENT Context text above.
    Keep your answer ground in the facts of the DOCUMENT Context.
    If the DOCUMENT Context doesn't contain the facts to answer the QUESTION return in 3 words- 'answer not found'

    Assistant:"""

    answer = model_invoke(prompt_template)
    return answer

if __name__ == '__main__':
    app.run(debug=True)
