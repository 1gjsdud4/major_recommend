from langchain.embeddings.openai import OpenAIEmbeddings


#####################
##### 임베딩 모델 ######
#####################

def openai_embedding_model(openai_api_key: str, embedding_model_name: str = "text-embedding-3-small"):
    """
    지정한 모델 이름을 사용하여 OpenAI 임베딩 모델을 반환합니다.
    """
    return OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=openai_api_key
    )

def ai_model_functionality(data):
    """
    여기에 AI 모델 관련 로직을 구현합니다.
    예를 들어, 데이터 전처리, 모델 예측 등을 처리할 수 있습니다.
    """
    # 예시: 단순 반환 또는 복잡한 처리 로직 삽입
    return data


####################
##### LLM 모델 ######
####################

if __name__ == '__main__':
    print('running...')