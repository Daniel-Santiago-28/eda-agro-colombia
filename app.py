"""Chatbot conversacional de cultura general e historia mundial (Groq · Llama 3.3 70B)."""
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Trivia Bot | Groq + Llama 3.3", page_icon="🌍", layout="centered")

MODEL_ID = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "Eres un asistente experto en cultura general e historia mundial. Respondes preguntas de trivia "
    "sobre historia, geografía, ciencia, arte, literatura, mitología y actualidad cultural con precisión, "
    "brevedad y un tono ameno y educativo. Si no estás seguro de un dato, dilo explícitamente en vez de "
    "inventarlo. Cuando sea útil, añade un dato curioso relacionado con la respuesta."
)

# ---------------------------------------------------------------------------
# Barra lateral: configuración
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
api_key = st.sidebar.text_input(
    "GROQ API Key", type="password",
    help="Tu clave solo se usa en esta sesión de navegador, no se guarda en disco.",
)
temperature = st.sidebar.slider("Creatividad (temperature)", 0.0, 1.5, 0.6, 0.1)
max_tokens = st.sidebar.slider("Longitud máxima de respuesta (tokens)", 128, 2048, 512, 64)

st.sidebar.divider()
if st.sidebar.button("🗑️ Borrar conversación"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.caption(f"Modelo: `{MODEL_ID}` vía Groq")

# ---------------------------------------------------------------------------
# Estado de la conversación
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🌍 Trivia Bot — Cultura general e historia mundial")
st.caption("Pregúntame sobre historia, geografía, ciencia, arte y cultura general. Impulsado por Llama 3.3 70B en Groq.")

if not api_key:
    st.info("Ingresa tu GROQ API Key en la barra lateral para comenzar a chatear.")
    st.stop()

client = Groq(api_key=api_key)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Escribe tu pregunta de cultura general o historia...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Ocurrió un error al consultar la API de Groq: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
