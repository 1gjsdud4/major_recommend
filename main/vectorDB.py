import os
import json
import time
import shutil
import csms_models
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vectorstore(persist_directory: str, collection_name: str) -> chromadb.Client:
    try:
        print("Creating vectorstore...")
        if not os.path.exists(persist_directory):
            client = chromadb.PersistentClient(path=persist_directory)
        
        client.get_or_create_collection(name=collection_name)

        print(f"success creat vectorstore! directory : {persist_directory} collection : {collection_name}" )
    
    except Exception as e:
        print("Error in create_vectorstore:", e)
        return {"error": str(e)}, 100

def delete_vectorstore(persist_directory: str, collection_name: str) -> str:
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        client.delete_collection(collection_name)
        print(f"Collection {collection_name} deleted successfully.")

        remaining_collections = client.list_collections()
                
        if not remaining_collections:  # 저장된 컬렉션이 없다면 벡터스토어 자체 삭제
            print(f"No collections left. Deleting entire vectorstore at {persist_directory}...")
            shutil.rmtree(persist_directory)  
            return f"All collections deleted. Vectorstore at {persist_directory} removed."
        
            
    except Exception as e:
        print("Error in delete_vectorstore:", e)
        return {"error": str(e)}, 101



def upload_major_vectorstore(majors_data: dict, persist_directory: str, collection_name: str, openai_api_key: str, chunk_size: int = 500, chunk_overlap: int = 100) -> chromadb.Collection:
    try:
        print(f"Uploading data to vectorstore at {persist_directory}... name = {collection_name}")
        
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(collection_name)

        embedding_model = csms_models.openai_embedding_model(openai_api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        texts = []
        metadatas = []
        ids = []
        embeddings = []

        main_name = collection_name

        existing_data = collection.get()  
        existing_ids = set(existing_data["ids"]) if "ids" in existing_data else set()
        
        existing_numbers = [int(i.split("_")[-1]) for i in existing_ids if i.split("_")[-1].isdigit()]
        next_id = max(existing_numbers) + 1 if existing_numbers else 1  

        for major in majors_data["majors"]:
            major_name = major["name"] 
            
            for i, desc in enumerate(major["description"]):
                split_texts = text_splitter.split_text(desc)

                for j, text_chunk in enumerate(split_texts):
                    
                    max_retries = 5  
                    for attempt in range(max_retries):
                        try:
                            vector = embedding_model.embed_query(text_chunk)
                            break  #
                        except Exception as e:
                            print(f"Error in embedding (attempt {attempt + 1}/{max_retries}):", e)
                            if attempt < max_retries - 1:
                                time.sleep(1)
                            else:
                                raise (f'embedding error - {e}')               

                    
                    
                    unique_id = f"{main_name}_{next_id}" 
                    next_id += 1  

                    embeddings.append(vector)
                    texts.append(text_chunk)
                    metadatas.append({"major_name": major_name})
                    ids.append(unique_id)

                    print(f"Major: {major_name}_{i}, Chunk: {j+1}/{len(split_texts)}, ID: {unique_id}")

        collection.add(
            documents=texts,
            embeddings=embeddings, 
            metadatas=metadatas, 
            ids=ids)

        client.persist()
        
        return collection
    except Exception as e:
        print("Error in upload_vectorstore:", e)
        return {"error": str(e)}, 102


def delete_major_vectorstore(persist_directory: str, collection_name: str, ids: list) -> str:
    try:
        print("Deleting majors in the vectorstore...")
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(collection_name)

        existing_data = collection.get()
        existing_ids = set(existing_data["ids"]) if "ids" in existing_data else set()

        missing_ids = set(ids) - set(existing_ids) 
        
        if missing_ids:
            error_msg = f"Error: Some IDs do not exist in the collection: {missing_ids}"
            print(f"Error: Some IDs do not exist in the collection: {missing_ids}")
            raise ValueError(error_msg)
        
        collection.delete(ids)
        print(f"deleted successfully.")
        
    except Exception as e:
        print("Error in delete_major_vectorstore:", e)
        return {"error": str(e)}, 103
    
def search_vectorstore(persist_directory: str, collection_name: str, openai_api_key: str, prompt: str, k: int = 6):
    try:
        print("Searching vectorstore...")
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(collection_name)
        embedding_model = csms_models.openai_embedding_model(openai_api_key)
        query_embeddings = embedding_model.embed_query(prompt)
        
        results = collection.query(query_embeddings=query_embeddings, n_results=k)

        def flatten(nested_list):
            return [item for sublist in nested_list for item in sublist] if isinstance(nested_list, list) and nested_list and isinstance(nested_list[0], list) else nested_list

        results["ids"] = flatten(results.get("ids", []))
        results["documents"] = flatten(results.get("documents", []))
        results["metadatas"] = flatten(results.get("metadatas", []))
        results["distances"] = flatten(results.get("distances", []))

        return results
    except Exception as e:
        print("Error in search_vectorstore:", e)
        return {"error": str(e)}, 104






if __name__ == '__main__':

    persist_dir = "data/test_vectorstore"
    collection_name = "test_collection"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    test_major_json = "data/major.json"  

    create_vectorstore(persist_dir, collection_name)

    with open(test_major_json, "r", encoding="utf-8") as file:
        majors_data = json.load(file)

    collection = upload_major_vectorstore(majors_data, persist_dir, collection_name, openai_api_key)

    test_prompt = "나는 컴퓨터를 좋아해"
    search_results = search_vectorstore(persist_dir, collection_name, openai_api_key, test_prompt, k=6)
    print("🔍 Search Results:", search_results)
    if search_results and "ids" in search_results and search_results["ids"]:
        delete_major_vectorstore(persist_dir, collection_name, search_results["ids"])


    delete_vectorstore(persist_dir, collection_name)
