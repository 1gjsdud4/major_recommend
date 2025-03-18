#%%
import os
import json
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

#%%

#### pinecone 인덱스 생성 ####

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "chamajor"

indexes= [index["name"] for index in pc.list_indexes()]

if index_name not in indexes:
    pc.create_index(
        name = index_name,
        dimension= 1536,
        metric= "cosine",
        spec=ServerlessSpec(
            cloud= "aws",
            region= 'us-east-1'
        )
        
    )
    print(f"Index {index_name} created")

#%%

#### json 데이터 embedding ####

file_path = 'file_path'
with open(file_path, "r", encoding="utf-8") as f:
    majors_data = json.load(f)
    
embedding_model = OpenAIEmbeddings(
    model= 'text-embedding-3-small',
    openai_api_key=os.getenv("OPENAI_API_KEY")
    )
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


vectorized_majors = []
id_count = 1
main_name = 'cha_major'
for major in majors_data["majors"]:
    major_name = major["name"] 
    
    for i, desc in enumerate(major["description"]):
        # 긴 텍스트 분할
        split_texts = text_splitter.split_text(desc)

        for j, text_chunk in enumerate(split_texts):
            # 임베딩 생성
            vector = embedding_model.embed_query(text_chunk)

            # Pinecone에 저장할 데이터 형식
            metadata = {"major_name": major_name, "description": text_chunk}
            unique_id = f"{main_name}_{id_count}_{i}_{j}"  # ASCII 

            vectorized_majors.append({"id": unique_id, "values": vector, "metadata": metadata})
            print(f"Major: {major_name}_{i}, Chunk: {j+1}/{len(split_texts)}")
    
    id_count += 1

print(vectorized_majors)
print(len(vectorized_majors))


#%%

#### pinecone에 데이터 업로드 ####
vectorstore_name = "cha_major"

host = pc.describe_index(index_name)["host"]
index = pc.Index(host=host)
upsert_result = index.upsert(
    vectors=vectorized_majors,
    namespace= vectorstore_name
    )



#%%
vectorstore_name = "cha_major"
promt = "나는 컴퓨터를 좋아해"
vector = embedding_model.embed_query(promt)
print(vector)

host = pc.describe_index(index_name)["host"]
index = pc.Index(host=host)

rag_result = index.query(
                namespace= vectorstore_name,
                vector = vector,
                top_k = 6,
                include_metadata=True
            )

print(rag_result)




#### pinecone에 데이터 검색 ####