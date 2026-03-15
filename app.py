import streamlit as st
import onnxruntime as ort
import utils.cv2_utils as cv2_utils
from utils.chromadb_utils import ChromaDB
from insightface.app import FaceAnalysis
import json
from pathlib import Path
import numpy as np

config_dados = None
with open('./config.json', 'r', encoding='utf-8') as f:
    config_dados = json.load(f)

@st.cache_resource
def load_model():
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640,640))
    return app

app = load_model()
chromadb = ChromaDB()

from rec_face import RecFace
st.set_page_config(
    page_title="FaceKey - Seu rosto é a chave",
    page_icon="./imagens/logo.jpeg",
    layout="wide",
)

placeholder = st.empty()

with placeholder.container():
    coluna1, coluna2, coluna3 = st.columns(3)
    coluna2.image('./imagens/logo.jpeg', width=280)
    coluna2.header("Faça sua experiência. 🎈")

st.sidebar.header("✨ Identificação")

uploaded_files = st.sidebar.file_uploader(
    "Realize o upload de sua imagem para verificar se você está na nossa base",
    accept_multiple_files=True,
    type=["jpg", "png", "jpeg"]
)

st.sidebar.header("😎 Cadastrar minha face")

new_uploaded_file = st.sidebar.file_uploader(
    "Realize o upload de sua imagem para se cadastrar",
    accept_multiple_files=False,
    type=["jpg", "png", "jpeg"]
)

st.sidebar.subheader(f"Total de embeddings no banco vetorial: {chromadb.get_total_emb()}")

placeholder_busca_vetorial = st.empty()

with placeholder_busca_vetorial.container():
    for uploaded_file in uploaded_files:
        placeholder.empty()
        st.session_state['fluxo'] = 1
        col1, col2, col3 = st.columns(3)
        col1.header("Imagem Original")
        col1.image(uploaded_file)
        file_name = uploaded_file.name

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2_utils.read_bytes(file_bytes=file_bytes)

        # Criando os embeddings
        rec_face = RecFace(app, img, file_name)
        rec_face.create_embedding()
        img_with_bbox = rec_face.create_img_with_bbox()
        print("img_with_bbox", img_with_bbox.shape)
        col2.header("Faces detectadas")
        img_with_bbox = cv2_utils.converter_canal(img_with_bbox)
        col2.image(img_with_bbox)

        col3.header("Faces alinhadas")
        numero_colunas = 2
        cols = col3.columns(numero_colunas)
        face_dados = rec_face.get_info()
        for i, face_dado in enumerate(face_dados):
            # col3.image(utils.converter_canal(face_dado['aligned_face']))
            with cols[i % numero_colunas]:
                st.image(cv2_utils.converter_canal(face_dado['aligned_face']), caption=f"Face {i+1}", use_container_width=True)

        st.header("💡 Identificação")
        # Realizar a busca no banco vetorial passando o embedding
        for i, face_dado in enumerate(face_dados):
            normed_emb = face_dado['normed_emb']
            resultado = chromadb.find_emb(normed_emb)
            if resultado is None:
                st.error("Algo inesperado aconteceu!", icon="🚨")
            else:
                print(resultado)
                if len(resultado['distances'][0]):
                    distance = resultado['distances'][0][0]
                    st.write(f"Distância do cosseno: {distance:.22f}")
                    if config_dados is not None:
                        if distance <= config_dados['threshold_rec_face']:
                            st.info(f"Face {i+1} reconhecida.")
                        else:
                            st.info(f"Face {i+1} não reconhecida.")
                else:
                    st.write(f'Banco de dados vetorial vazio. Cadastre novas imagens.')

if new_uploaded_file:
    placeholder_busca_vetorial.empty()
    placeholder.empty()
    st.session_state['fluxo'] = 2
    
    st.header("😎 Cadastrar minha face")
    file_name = new_uploaded_file.name
    file_bytes = np.asarray(bytearray(new_uploaded_file.read()), dtype=np.uint8)
    img = cv2_utils.read_bytes(file_bytes=file_bytes)
    rec_face = RecFace(app, img, file_name)
    rec_face.create_embedding()
    if rec_face.quant_faces > 1:
        st.error("Insira uma imagem com apenas uma pessoa!", icon="🚨")
    else:
        img_with_bbox = rec_face.create_img_with_bbox()
        st.subheader("Faces detectadas")
        img_with_bbox = cv2_utils.converter_canal(img_with_bbox)
        st.image(img_with_bbox)

        face_dados = rec_face.get_info()
        for face_dado in face_dados:
            res = chromadb.find_emb(face_dado['normed_emb'])
            if res is not None:
                if len(res['distances'][0])==0: # caso onde o banco de dados vetorial está vazio
                    chromadb.insert_emb(emb=face_dado['normed_emb'],
                                        path_file=face_dado['file_name'],
                                        )
                    st.info("Face cadastrada com sucesso.")
                elif res['distances'][0][0] <= config_dados['threshold_face_already_exists']:
                    st.error(f"Esta face já foi cadastrada. Distância do cosseno: {res['distances'][0][0]:.22f}")
                else:        
                    chromadb.insert_emb(emb=face_dado['normed_emb'],
                                        path_file=face_dado['file_name'],
                                        )
                    st.info("Face cadastrada com sucesso.")

st.markdown("<br>" * 5, unsafe_allow_html=True)
st.subheader("🥸 Saiba mais")
links_row_1, links_row_2 = st.columns(2)
links_row_1.link_button(
    "📖  Veja outros projetos",
    "https://github.com/FelipeErmeson/",
    use_container_width=True,
)

links_row_2.link_button(
    "🐙  Visite o repositório",
    "https://github.com/FelipeErmeson/face-key",
    use_container_width=True,
)

st.markdown("#### ☕ Sobre este app")
st.markdown("""
O FaceKey é um projeto no qual você pode cadastrar faces de pessoas e em seguida
pode realizar o reconhecimento facial.

Este projeto utiliza a biblioteca [InsightFace](https://github.com/deepinsight/insightface)
com o pacote **buffalo_l**, no qual utiliza para detecção de faces o **RetinaFace** e como reconhecimento
de faces o **ArcFace** (Angular Margin Loss).
""")