from typing import List
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
from datetime import datetime
from PIL import Image
import base64
from fpdf import FPDF 
import plotly.express as px
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings
from ultralytics import YOLO
from config import CLASSES, WEBRTCCLIENTSETTINGS

st.set_page_config(page_title="Подсчет посетителей (МТУСИ практика)", layout="wide")
st.title("Подсчет посетителей в магазине, вариант 1")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
person_id = CLASSES.index('person')

HISTORY_FILE = 'history.json'
history = []
if st.session_state.get('history_loaded', False) is False:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []
    st.session_state.history = history
    st.session_state.history_loaded = True

def save_history(entry):
    history.append(entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    st.session_state.history = history

st.sidebar.header('Настройки')
target_classes = st.sidebar.multiselect('Классы', CLASSES, default=['person'])
model_size = st.sidebar.selectbox('Размер модели', ['n', 's', 'm'])

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Загрузка изображения/видео")
    uploaded_file = st.file_uploader("Выберите файл", type=['png','jpg','jpeg','mp4','avi'])
    
    if uploaded_file:
        if st.button('Запустить обработку изображения', key='process_img'):
            with st.spinner('Обработка...'):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = model(img_rgb, verbose=False)
                res = results[0].boxes.data.cpu().numpy()
                target_ids = [CLASSES.index(c) for c in target_classes]
                res = res[res[:, 5].astype(int).isin(target_ids)]
                count = int(len(res))
                
                annotated = results[0].plot()
                st.image(annotated, caption=f'👥 Посетителей: {count}', use_column_width=True)
                
                preview_b64 = base64.b64encode(cv2.imencode('.jpg', img_rgb)[1].tobytes()).decode()
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'type': uploaded_file.type.split('/')[0],
                    'count': count,
                    'classes': target_classes,
                    'preview': preview_b64
                }
                save_history(entry)
                st.success(f'Сохранено в историю: {count} человек')

with col2:
    st.header("Стрим с камеры")
    use_camera = st.checkbox('Запустить камеру')
    
    class PeopleCounter(VideoTransformerBase):
        count = 0
        def recv_queued(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(img_rgb, verbose=False)
            res = results[0].boxes.data.cpu().numpy()
            res = res[res[:, 5].astype(int) == person_id]
            self.count = len(np.unique(res[:, -1])) if len(res)>0 else 0
            annotated = results[0].plot()
            return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    
    if use_camera:
        webrtc_streamer(
            key="camera",
            video_transformer_factory=PeopleCounter,
            client_settings=WEBRTCCLIENTSETTINGS
        )

st.header("История и отчеты")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    st.dataframe(df[['timestamp', 'type', 'count']].tail(10).sort_values('timestamp', ascending=False))
    
    fig = px.line(df.tail(20), x='timestamp', y='count', title='Динамика посетителей')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('PDF отчет'):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=16)
            pdf.cell(0, 10, "Отчет по подсчету посетителей", ln=1, align="C")
            pdf.set_font("Arial", size=12)
            for _, row in df.tail(10).iterrows():
                pdf.cell(0, 10, f"{row.timestamp}: {row.count} чел. ({row.type})", ln=1)
            pdf_output = "report.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                st.download_button("Скачать PDF", f.read(), file_name=pdf_output)
    
    with col2:
        if st.button('Excel'):
            excel_file = "history.xlsx"
            df.to_excel(excel_file, index=False)
            with open(excel_file, "rb") as f:
                st.download_button("Скачать Excel", f.read(), file_name=excel_file)
else:
    st.info("История пуста. Загрузите изображение или запустите камеру!")
