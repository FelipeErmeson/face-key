import streamlit as st
import onnxruntime as ort
print('ONNXRUNTIME', ort.get_available_providers())
import utils
from insightface.app import FaceAnalysis

@st.cache_resource
def load_model():
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640,640))
    return app

app = load_model()

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
    coluna2.header("Faça sua experiência.")

st.sidebar.header("Identificação")

uploaded_files = st.sidebar.file_uploader(
    "Realize o upload de sua imagem para verificar se você está na nossa base",
    accept_multiple_files=True,
    type=["jpg", "png", "jpeg"]
)

st.sidebar.header("Cadastrar minha face")

new_uploaded_file = st.sidebar.file_uploader(
    "Realize o upload de sua imagem para se cadastrar",
    accept_multiple_files=False,
    type=["jpg", "png", "jpeg"]
)

placeholder_busca_vetorial = st.empty()

with placeholder_busca_vetorial.container():
    for uploaded_file in uploaded_files:
        placeholder.empty()
        st.session_state['fluxo'] = 1
        col1, col2, col3 = st.columns(3)
        col1.header("Imagem Original")
        print('type', type(uploaded_file))
        col1.image(uploaded_file)
        img = utils.read_img(uploaded_file.name)
        print("img", img.shape)

        rec_face = RecFace(app, img)
        rec_face.create_embedding()
        img_with_bbox = rec_face.create_img_with_bbox()
        print("img_with_bbox", img_with_bbox.shape)
        col2.header("Faces detectadas")
        img_with_bbox = utils.converter_canal(img_with_bbox)
        col2.image(img_with_bbox)

        col3.header("Faces alinhadas")
        numero_colunas = 2
        cols = col3.columns(numero_colunas)
        face_dados = rec_face.get_info()
        for i, face_dado in enumerate(face_dados):
            # col3.image(utils.converter_canal(face_dado['aligned_face']))
            with cols[i % numero_colunas]:
                st.image(utils.converter_canal(face_dado['aligned_face']), caption=f"Imagem {i+1}", use_container_width=True)

        st.header("Identificação")

        # Realizar a busca no banco vetorial passando o embedding

if new_uploaded_file:
    placeholder_busca_vetorial.empty()
    placeholder.empty()
    st.session_state['fluxo'] = 2
    
    st.header("Imagem a ser cadastrada")