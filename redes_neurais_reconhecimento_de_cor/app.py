import sys
import streamlit as st
sys.path.append('scripts')
from scripts.prever_cor import prever_cor

st.title('Reconhecimento de Cores em Imagens')

imagem = st.file_uploader("Escolha uma imagem", type=['jpg', 'png', 'jpeg'])

if imagem is not None:
    st.image(imagem, caption="Imagem carregada", use_column_width=True)
    with open("SUA_IMAGEM_AQUI", "wb") as f:
        f.write(imagem.getbuffer())
    cor = prever_cor("SUA_IMAGEM_AQUI")
    st.write(f'A cor predominante na imagem é: {cor}')
