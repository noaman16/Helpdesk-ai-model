import mysql.connector
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Connect to MySQL and fetch data
def load_data_from_mysql():
    conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3307,
    user="root",
    password="Husena@43",
    database="helpdesk"
)

    query = "SELECT id, subject, content FROM ticketit LIMIT 1000;"
 # Adjust limit or query as needed
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Step 2: Preprocess the subject and content into a single text field
def preprocess_tickets(df):
    df['subject'] = df['subject'].fillna('')
    df['content'] = df['content'].fillna('')
    df['text'] = (df['subject'] + " " + df['content']).str.strip()
    return df

# Step 3: Generate embeddings using SentenceTransformer
def generate_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, model

# Step 4: Build the similarity search index with FAISS
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    index.add(embeddings.astype(np.float32))
    return index

# Step 5: Search for similar tickets
def find_similar_tickets(query_text, model, index, df, top_k=5):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    return results

# Main execution
def main():
    print("Loading data from MySQL...")
    df = load_data_from_mysql()
    print(f"Loaded {len(df)} tickets.")
    
    df = preprocess_tickets(df)
    print("Preprocessed tickets.")
    
    print("Generating embeddings...")
    embeddings, model = generate_embeddings(df['text'].tolist())
    
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    # Example new ticket to find similar tickets for
    new_ticket = "Unable to connect to the wireless printer on the office network"
    print(f"\nFinding tickets similar to: \"{new_ticket}\"\n")
    
    similar_tickets = find_similar_tickets(new_ticket, model, index, df)
    
    for i, row in similar_tickets.iterrows():
        print(f"Ticket ID: {row['id']}")
        print(f"Subject: {row['subject']}")
        print(f"Distance (Lower means more similar): {row['distance']:.4f}")
        print(f"Content: {row['content'][:200]}...")  # Limit content preview
        print("-" * 50)

if __name__ == "__main__":
    main()
