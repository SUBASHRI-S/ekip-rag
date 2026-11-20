import hashlib

import httpx
import streamlit as st

API_URL = "http://127.0.0.1:8000"
EXAMPLES = [
    "What is the maximum loan to value ratio for gold loans given by NBFCs?",
    "What is a lending service provider?",
    "Can an NBFC charge pre-payment penalties on microfinance loans?",
]

st.set_page_config(page_title="RBI Directions Q&A", page_icon="📚", layout="wide")
st.markdown(
    "<style>.block-container {max-width: 980px; padding-top: 2rem;}</style>",
    unsafe_allow_html=True,
)

for key, default in {
    "messages": [],
    "conversation_id": None,
    "last_audio": None,
    "voice_text": "",
}.items():
    st.session_state.setdefault(key, default)


def api(method, path, timeout=60, **kwargs):
    """Call the API. Shows an error and returns None if it does not answer."""
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as error:
        st.error(f"The API did not answer: {error}")
        return None


def new_chat():
    st.session_state.messages = []
    st.session_state.conversation_id = None


def open_chat(conversation_id):
    messages = api("GET", f"/conversations/{conversation_id}")
    if messages is not None:
        st.session_state.messages = messages
        st.session_state.conversation_id = conversation_id


def delete_chat():
    if st.session_state.conversation_id:
        api("DELETE", f"/conversations/{st.session_state.conversation_id}")
    new_chat()


def ask_voice():
    st.session_state.pending_question = st.session_state.voice_text.strip()
    st.session_state.voice_text = ""


def ask_example(text):
    st.session_state.pending_question = text


def rate(index, rating):
    message = st.session_state.messages[index]
    if api("POST", "/feedback", json={"message_id": message["message_id"], "rating": rating}):
        message["rating"] = rating


def show_sources(sources):
    st.markdown("**Sources**")
    for s in sources:
        with st.expander(f"[{s['number']}] {s['title']} - pages {s['pages']}"):
            st.write(s["text"])
            st.caption(f"{s['file_name']} | match distance {s['distance']}")


with st.sidebar:
    st.title("📚 RBI Q&A")
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)

    st.subheader("Chats")
    for chat in api("GET", "/conversations") or []:
        marker = "▶ " if chat["id"] == st.session_state.conversation_id else ""
        st.button(
            marker + chat["title"],
            key=f"chat{chat['id']}",
            on_click=open_chat,
            args=(chat["id"],),
            use_container_width=True,
        )
    if st.session_state.conversation_id:
        st.button("🗑 Delete this chat", on_click=delete_chat, use_container_width=True)

    st.subheader("Search mode")
    choice = st.radio(
        "Search mode",
        ["Fast (hybrid)", "More accurate (reranked, slower)"],
        label_visibility="collapsed",
    )
    mode = "hybrid" if choice.startswith("Fast") else "rerank"

    st.subheader("🎤 Ask by voice")
    audio = st.audio_input("Record a question", label_visibility="collapsed")
    if audio is not None:
        data = audio.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if digest != st.session_state.last_audio:
            st.session_state.last_audio = digest
            with st.spinner("Listening..."):
                heard = api(
                    "POST", "/transcribe", timeout=120,
                    files={"audio": ("voice.wav", data, "audio/wav")},
                )
            if heard:
                st.session_state.voice_text = heard["text"]
    if st.session_state.voice_text:
        st.text_input("I heard (edit if needed):", key="voice_text")
        st.button("Ask this", on_click=ask_voice, use_container_width=True)

st.title("RBI Directions Q&A")
st.caption(
    "Ask about lending, KYC and customer-protection rules for commercial banks, "
    "NBFCs and small finance banks. Every answer cites the document and page."
)

if not st.session_state.messages:
    st.markdown("#### Try one of these")
    for number, text in enumerate(EXAMPLES):
        st.button(text, key=f"example{number}", on_click=ask_example, args=(text,))

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            continue
        if message.get("answerable") is False:
            st.info("No supported answer was found. The closest passages are listed below.")
        if message.get("sources"):
            show_sources(message["sources"])
        trace = message.get("trace") or {}
        if trace:
            st.caption(f"{trace.get('mode')} search, answered in {trace.get('total_seconds')} s")
        if message.get("rating"):
            st.caption("You rated this " + ("👍" if message["rating"] == 1 else "👎"))
        elif message.get("message_id"):
            columns = st.columns([1, 1, 8])
            columns[0].button("👍", key=f"up{index}", on_click=rate, args=(index, 1))
            columns[1].button("👎", key=f"down{index}", on_click=rate, args=(index, -1))

typed = st.chat_input("Ask a question about the RBI Directions")
question = st.session_state.pop("pending_question", None) or typed
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            data = api(
                "POST", "/chat", timeout=180,
                json={
                    "question": question,
                    "mode": mode,
                    "conversation_id": st.session_state.conversation_id,
                },
            )
    if data is None:
        st.session_state.messages.pop()
        st.stop()
    st.session_state.conversation_id = data["conversation_id"]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "answerable": data["answerable"],
            "sources": data["sources"],
            "trace": data["trace"],
            "message_id": data["message_id"],
        }
    )
    st.rerun()
    