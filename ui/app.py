import httpx
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RBI Directions Q&A", page_icon="📚")
st.title("RBI Directions Q&A")
st.caption(
    "Ask about lending, KYC and customer-protection rules for commercial banks, "
    "NBFCs and small finance banks. Every answer cites the document and page."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

with st.sidebar:
    st.header("Settings")
    choice = st.radio("Search mode", ["Fast (hybrid)", "More accurate (reranked, slower)"])
    mode = "hybrid" if choice.startswith("Fast") else "rerank"
    st.markdown("**Try asking**")
    st.markdown(
        "- What is the maximum loan to value ratio for gold loans given by NBFCs?\n"
        "- What is a lending service provider?\n"
        "- Can an NBFC charge pre-payment penalties on microfinance loans?"
    )
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()


def show_sources(sources):
    st.markdown("**Sources**")
    for s in sources:
        with st.expander(f"[{s['number']}] {s['title']} - pages {s['pages']}"):
            st.write(s["text"])
            st.caption(f"{s['file_name']} | match distance {s['distance']}")


def rate(index, rating):
    message = st.session_state.messages[index]
    try:
        reply = httpx.post(
            f"{API_URL}/feedback",
            json={"message_id": message["message_id"], "rating": rating},
            timeout=30,
        )
        reply.raise_for_status()
        message["rating"] = rating
    except httpx.HTTPError:
        st.toast("Could not save your feedback")


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
            st.caption("Thanks for your feedback.")
        elif message.get("message_id"):
            columns = st.columns([1, 1, 8])
            columns[0].button("👍", key=f"up{index}", on_click=rate, args=(index, 1))
            columns[1].button("👎", key=f"down{index}", on_click=rate, args=(index, -1))

question = st.chat_input("Ask a question about the RBI Directions")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                response = httpx.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "mode": mode,
                        "conversation_id": st.session_state.conversation_id,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as error:
                st.error(f"Could not get an answer from the API: {error}")
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