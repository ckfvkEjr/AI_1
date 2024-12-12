import streamlit as st
from fastai.vision.all import *
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gdown
import io

# Google Drive 파일 ID
file_id = '1xaPgoHahhdzLaIi652ySl_7Sw_Kcut9X'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def audio_to_melspectrogram(audio_file):
    """
    음악 파일을 멜 스펙트로그램 이미지로 변환.
    """
    y, sr = librosa.load(audio_file, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 멜 스펙트로그램 시각화
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.axis('off')

    # 이미지를 메모리에 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    # PIL 이미지로 변환
    image = Image.open(buf)
    return image

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="멜 스펙트로그램 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://via.placeholder.com/300?text=Label1_Image1",
            "https://via.placeholder.com/300?text=Label1_Image2",
            "https://via.placeholder.com/300?text=Label1_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "Label 1 관련 첫 번째 텍스트 내용입니다.",
            "Label 1 관련 두 번째 텍스트 내용입니다.",
            "Label 1 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[1]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "Label 2 관련 첫 번째 텍스트 내용입니다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg",
            "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg",
            "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=5tafCyiYGpU",
            "https://www.youtube.com/watch?v=unrcrAUdqH8",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "핑동",
            "커스텀마이징 쉬운건 좋은데",
            "똥겜인듯"
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (mp3, wav 지원)
uploaded_file = st.file_uploader("오디오 파일을 업로드하세요", type=["mp3", "wav"])

if uploaded_file is not None:
    # 멜 스펙트로그램 생성
    mel_spec_image = audio_to_melspectrogram(uploaded_file)
    img = PILImage.create(mel_spec_image)

    # 모델 예측
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(mel_spec_image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)
