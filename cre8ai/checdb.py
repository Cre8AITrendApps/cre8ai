import os
import chromadb

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db")

print("-" * 50)
print("Attempting to connect to ChromaDB at path:")
print(f"-> {DB_DIR}")
print("-" * 50)

if not os.path.exists(DB_DIR):
    print("ERROR: The database directory does not exist at that path.")
    print("   Please run the ingest.py script first.")
else:
    try:
        client = chromadb.PersistentClient(path=DB_DIR)

        collections = client.list_collections()

        if not collections:
            print(" WARNING: Successfully connected to the database, but found NO collections.")
            print("   This means the ingestion script might not have saved the data correctly.")
        else:
            print(" SUCCESS: Found the following client collections:")
            for i, collection in enumerate(collections):
                print(f"  {i+1}. ID: {collection.name}")

    except Exception as e:
        print(" ERROR: An unexpected error occurred while trying to connect to the database.")
        print(f"   Error details: {e}")

print("-" * 50)