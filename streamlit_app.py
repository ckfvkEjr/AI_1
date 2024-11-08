import streamlit as st
import pandas as pd

# 타이틀 설정
st.title("근데 이제 뭐함")

# 서브타이틀
st.subheader("진짜 모름;;")

# 표 데이터 생성
data = {
    '이름': ['트럼프', '바이든', 'UmJunSik'],
    '나이': [100, 100 , 100],
    '직업': ['대통령', '전 대통령', '엄준식']
}
df = pd.DataFrame(data)

# 표 출력
st.write("표?:")
st.dataframe(df)

# HTML 메시지 출력
st.markdown("""
    <div style="background-color: yellow; padding: 10px;">
        <h3>환상의 똥꼬쇼!</h3>
        <p>뭔가 보여드리겠습니다!</p>
    </div>
""", unsafe_allow_html=True)

# 추가적인 텍스트
st.write("근데 진짜 뭐함")
