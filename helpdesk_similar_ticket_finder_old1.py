import mysql.connector
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load all tickets and prepare embedding index at start
def load_and_prepare_index():
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3307,
        user="root",
        password="Husena@43",
        database="helpdesk"
    )
    query = "SELECT id, subject, content FROM ticketit;"
    df = pd.read_sql(query, conn)
    conn.close()

    df['subject'] = df['subject'].fillna('')
    df['content'] = df['content'].fillna('')
    df['text'] = (df['subject'] + " " + df['content']).str.strip()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    return df, model, index

def search_similar_tickets(subject, content, df, model, index, top_k=5, threshold=0.8):
    text = (subject + " " + content).strip()
    query_embedding = model.encode([text])
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)

    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]

    # Convert L2 distance to similarity score approximately (smaller distance is better)
    # Here we consider lower distance better; you can select threshold empirically
    filtered_results = results[results['distance'] < threshold]

    return filtered_results

def main():
    print("Loading tickets database and building similarity index...")
    df, model, index = load_and_prepare_index()
    print(f"Loaded {len(df)} tickets.")
    print("Ready to search similar tickets.\n")

    while True:
        subject = input("Enter new ticket subject (or type 'exit' to quit): ").strip()
        if subject.lower() == 'exit':
            break
        content = input("Enter new ticket content: ").strip()

        similar_tickets = search_similar_tickets(subject, content, df, model, index)
        if similar_tickets.empty:
            print("No similar tickets found.")
        else:
            print(f"\nFound {len(similar_tickets)} similar tickets:")
            for i, row in similar_tickets.iterrows():
                print(f"\nTicket ID: {row['id']}")
                print(f"Subject: {row['subject']}")
                print(f"Distance (lower means more similar): {row['distance']:.4f}")
                print(f"Content Preview: {row['content'][:200]}...")
                print("-" * 50)
        print("\n")

if __name__ == "__main__":
    main()
