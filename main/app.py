import json
import streamlit as st
import time
from datetime import datetime
import streamlit_function as sf




def main():

    
    if "prompt" not in st.session_state:
        st.session_state.prompt = None
    
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    if "final_result" not in st.session_state:
        st.session_state.final_result = None

    st.title("차의과대 전공 추천")
    st.caption("차의과대의 전공추천 에이전트")

    col1, col2 = st.columns([1, 1])

    with col1:
        
        st.markdown("### 입력 프롬프트")
        st.header("질문 입력")

        # 질문 6개 디폴트 값
        default_questions = [
            ("어떤 성격의 학문에 더 끌리시나요?", "이론적이고 학문적인 탐구"),
            ("본인이 가장 관심 있는 분야는 무엇인가요?", "인문학 (문학, 철학, 역사 등)"),
            ("학문을 선택할 때 가장 중요하게 고려하는 요인은 무엇인가요?", "취업 가능성"),
            ("당신의 성격에 가장 잘 맞는 전공 유형은 무엇이라고 생각하나요?", "분석적이고 논리적인 전공"),
            ("미래 직업과 연결된 전공을 선택할 때 중요하게 생각하는 것은 무엇인가요?", "안정적인 소득"),
            ("어떤 유형의 학습 방식을 선호하시나요?", "강의를 듣고 이해하는 방식")
        ]
        
        # 질문/답변을 session_state에 저장
        if "answers" not in st.session_state:
            st.session_state.answers = [
                ans for (_, ans) in default_questions
            ]

        # 6개 질문을 표시하고, 사용자가 답 변경 가능
        for i, (q, default_ans) in enumerate(default_questions):
            st.write(f"**질문 {i+1}**: {q}")
            st.session_state.answers[i] = st.text_area(
                f"답변 {i+1}", 
                value=st.session_state.answers[i], 
                key=f"question_{i}", 
               
            )

    with col2:
        run =st.button("전공 추천")
        
        if run:
            answers_json_str = json.dumps({"answers": st.session_state.answers}, ensure_ascii=False, indent=2)
        
            with st.spinner("🔍 전공 추천 중... 잠시만 기다려 주세요!"):
                result = sf.run_major_recommendation(
                    user_input=answers_json_str,
                    index_name="chamajor",
                    vectorstore_name="cha_major",
                    num_recommendations=3,
                    max_turn=3
                ) 
            st.session_state.fianl_result = result
            
            st.json(result["result"])


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()




###############