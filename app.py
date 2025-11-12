import mysql.connector
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from flask import Flask, request, jsonify, send_from_directory


EMBEDDINGS_FILE = 'embeddings.npy'
TICKETS_FILE = 'tickets.pkl'


app = Flask(__name__, static_folder='static')


def load_and_prepare_index():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(TICKETS_FILE):
        print("Loading cached embeddings and tickets...")
        embeddings = np.load(EMBEDDINGS_FILE)
        df = pd.read_pickle(TICKETS_FILE)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings.astype(np.float32))

        return df, model, faiss_index


    print("Cached data not found. Loading from MySQL and generating embeddings...")
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3307,
        user="root",
        password="Husena@43",
        database="helpdesk"
    )
    query = """
    SELECT 
      t.id, t.subject, t.content, t.user_id, t.agent_id, 
      u1.name as created_by_name, u1.email as created_by_email,
      u2.name as responsible_name, u2.email as responsible_email
    FROM ticketit t
    LEFT JOIN users u1 ON t.user_id = u1.id
    LEFT JOIN users u2 ON t.agent_id = u2.id
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df.fillna({'subject':'', 'content': '', 
               'created_by_name': '', 'created_by_email': '',
               'responsible_name': '', 'responsible_email': ''}, inplace=True)

    df['text'] = (df['subject'] + " " + df['content']).str.strip()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    np.save(EMBEDDINGS_FILE, embeddings)
    df.to_pickle(TICKETS_FILE)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings.astype(np.float32))

    return df, model, faiss_index


df, model, faiss_index = load_and_prepare_index()


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/search_similar', methods=['POST'])
def search_similar():
    data = request.get_json()
    subject = data.get('subject', '')
    content = data.get('content', '')

    text = (subject + " " + content).strip()
    query_embedding = model.encode([text])
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), 5)

    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]

    threshold = 0.8
    filtered_results = results[results['distance'] < threshold]

    response = []
    for _, row in filtered_results.iterrows():
        response.append({
            'id': int(row['id']),
            'subject': str(row['subject']),
            'distance': float(row['distance']),
            'content': str(row['content']),
            'created_by_name': str(row['created_by_name']),
            'created_by_email': str(row['created_by_email']),
            'responsible_name': str(row['responsible_name']),
            'responsible_email': str(row['responsible_email']),
        })
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
