import hashlib

import httpx
import streamlit as st

API_URL = "http://127.0.0.1:8000"
RBI_EXAMPLES = [
    "What is the maximum loan to value ratio for gold loans given by NBFCs?",
    "What is a lending service provider?",
    "Can an NBFC charge pre-payment penalties on microfinance loans?",
]

st.set_page_config(page_title="Ask your documents", page_icon="📚", layout="wide")
st.markdown(
    "<style>.block-container {max-width: 980px; padding-top: 2rem;}</style>",
    unsafe_allow_html=True,
)

for key, default in {
    "messages": [],
    "conversation_id": None,
    "last_audio": None,
    "voice_text": "",
    "token": None,
    "username": None,
}.items():
    st.session_state.setdefault(key, default)


def api(method, path, timeout=60, **kwargs):
    """Call the API with the login token. Shows an error and returns None if it fails."""
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=timeout, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail", error)
        except ValueError:
            detail = error
        if error.response.status_code == 401 and st.session_state.token:
            st.session_state.token = None  # the login expired: the next click shows the login screen
            st.session_state.username = None
        st.error(f"The API said: {detail}")
    except httpx.HTTPError as error:
        st.error(f"The API did not answer: {error}")
    return None


def login_screen():
    st.title("📚 Ask your documents")
    st.caption(
        "Log in to use your own collections and chats. "
        "The RBI Master Directions collection is shared with everyone."
    )
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            result = api("POST", "/auth/login", json={"username": username, "password": password})
            if result:
                st.session_state.token = result["token"]
                st.session_state.username = result["username"]
                st.rerun()
    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username (letters, digits, underscore)")
            new_password = st.text_input("Choose a password (at least 8 characters)", type="password")
            created = st.form_submit_button("Create account")
        if created:
            result = api("POST", "/auth/register", json={"username": new_username, "password": new_password})
            if result:
                st.session_state.token = result["token"]
                st.session_state.username = result["username"]
                st.rerun()


if not st.session_state.token:
    login_screen()
    st.stop()


def log_out():
    for key in ("token", "username", "conversation_id", "last_audio"):
        st.session_state[key] = None
    st.session_state.messages = []
    st.session_state.voice_text = ""


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


collections = api("GET", "/collections") or []
by_name = {c["name"]: c for c in collections}

with st.sidebar:
    st.title("📚 Ask your documents")
    st.caption(f"Signed in as **{st.session_state.username}**")
    st.button("Log out", on_click=log_out, use_container_width=True)
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)

    st.subheader("Collection")
    if "next_collection" in st.session_state:  # set after an upload, before the box is drawn
        st.session_state["collection_name"] = st.session_state.pop("next_collection")
    names = [c["name"] for c in collections]
    if names and st.session_state.get("collection_name") not in names:
        st.session_state["collection_name"] = names[0]
    selected_name = (
        st.selectbox("Ask about", names, key="collection_name", label_visibility="collapsed") if names else None
    )
    selected = by_name.get(selected_name)
    if selected:
        with st.expander(f"📄 Documents ({len(selected['documents'])})"):
            for doc in selected["documents"]:
                st.caption(f"{doc['title']} · {doc['pages']} pages")
        if not selected["read_only"]:
            if st.button("🗑 Delete this collection", use_container_width=True):
                if api("DELETE", f"/collections/{selected['id']}"):
                    new_chat()
                    st.rerun()

    with st.expander("⬆️ Add your own PDFs"):
        target = st.text_input("Collection name", value="My documents")
        files = st.file_uploader(
            "Choose PDF files (select several, or all files of a folder)",
            type="pdf",
            accept_multiple_files=True,
        )
        st.caption("Text PDFs only. Your documents' text is sent to the LLM provider with each question.")
        if st.button("Add to collection", disabled=not files, use_container_width=True):
            with st.spinner("Reading and indexing your documents. This can take a minute per file..."):
                outcome = api(
                    "POST", "/upload", timeout=900,
                    data={"collection": target},
                    files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in files],
                )
            if outcome:
                for item in outcome["results"]:
                    (st.success if item["status"].startswith("added") else st.warning)(
                        f"{item['file']}: {item['status']}"
                    )
                st.session_state["next_collection"] = outcome["collection"]
                st.session_state["upload_done"] = True
    if st.session_state.pop("upload_done", False):
        st.rerun()

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

st.title("Ask your documents")
if selected:
    st.caption(
        f"Asking inside the collection **{selected['name']}**. "
        "Every answer cites the document and page."
    )
if not collections:
    st.info(
        "You have no documents yet. Open ⬆️ Add your own PDFs in the sidebar, "
        "upload one or more text PDFs, then ask questions about them."
    )
elif not st.session_state.messages and selected and selected["read_only"]:
    st.markdown("#### Try one of these")
    for number, text in enumerate(RBI_EXAMPLES):
        st.button(text, key=f"example{number}", on_click=ask_example, args=(text,))
elif not st.session_state.messages:
    st.info("Ask a question about the documents in this collection.")
    
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

typed = st.chat_input("Ask a question about the selected collection")
question = st.session_state.pop("pending_question", None) or typed
if question and selected:
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
                    "collection_id": selected["id"],
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