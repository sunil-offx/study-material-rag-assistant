import os
import sqlite3
import tempfile
from datetime import datetime, timedelta

import streamlit as st

from rag_study_assistant import (
    STUDY_DIR,
    ingest,
    retrieve,
    answer_question,
)

DB_PATH = os.environ.get("DB_PATH", os.path.join(tempfile.gettempdir(), "study_materials.db"))


def init_storage():
    """Ensure study folder and SQLite DB exist."""
    STUDY_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Uploaded files table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL
        );
        """
    )
    # Simple history table for Q&A
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def record_upload(filename: str, original_name: str, size_bytes: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO uploads (filename, original_name, size_bytes, uploaded_at) VALUES (?, ?, ?, ?)",
        (filename, original_name, size_bytes, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def load_uploads():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT filename, original_name, size_bytes, uploaded_at FROM uploads ORDER BY uploaded_at DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def save_history(username: str, question: str, answer: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO history (username, question, answer, pinned, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, question, answer, 0, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def load_history(username: str):
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, question, answer, pinned, created_at
        FROM history
        WHERE username = ?
        ORDER BY pinned DESC, created_at DESC
        """,
        (username,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def clear_history(username: str):
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE username = ?", (username,))
    conn.commit()
    conn.close()


def set_pin(history_id: int, pinned: bool):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE history SET pinned = ? WHERE id = ?",
        (1 if pinned else 0, history_id),
    )
    conn.commit()
    conn.close()


def ensure_user_signed_in():
    """Show a sign up / sign in gate the first time, and remember for 48h."""
    now = datetime.utcnow()
    expires = st.session_state.get("signed_in_expires")

    # Normalize expires if it's a string
    if isinstance(expires, str):
        try:
            expires = datetime.fromisoformat(expires)
        except ValueError:
            expires = None

    username = st.session_state.get("username")

    if username and expires and expires > now:
        # User is still within the 48h window
        return

    st.markdown("### Welcome to Study Material Assistant")
    st.write(
        "Please **sign up or sign in** to start using the application. "
        "Your session will be remembered for 48 hours in this browser."
    )

    choice = st.radio("Select an option", ["Sign up", "Sign in"], horizontal=True)
    default_name = username or ""
    input_label = "Choose a user name" if choice == "Sign up" else "User name"
    name_input = st.text_input(input_label, value=default_name, key="auth_name_input")

    if st.button("Continue"):
        name = (name_input or "").strip()
        if not name:
            st.warning("Please enter a user name.")
        else:
            st.session_state["username"] = name
            st.session_state["signed_in_expires"] = now + timedelta(hours=48)
            st.success(f"Signed in as `{name}` (valid for the next 48 hours).")
            st.rerun()

    # Stop rendering the rest of the app until the user signs in
    st.stop()


def ensure_env():
    if not os.environ.get("GROQ_API_KEY"):
        st.warning(
            "Environment variable `GROQ_API_KEY` is not set. "
            "Set it before using the assistant, otherwise answering will fail."
        )


def main():
    st.set_page_config(page_title="Study Material RAG Assistant", layout="wide")
    st.title("📚 Study Material Assistant (RAG)")
    st.write(
        "Upload study materials and ask questions that are answered "
        "**only from your uploaded PDFs / TXT files**."
    )

    ensure_env()
    init_storage()
    ensure_user_signed_in()

    # Simple account area in sidebar (after initial sign-in gate)
    with st.sidebar:
        st.subheader("Sign in")
        if "username" not in st.session_state:
            st.session_state["username"] = "guest"
        username_input = st.text_input("User name", value=st.session_state["username"])
        if st.button("Update user"):
            name = (username_input or "guest").strip()
            st.session_state["username"] = name or "guest"
            st.session_state["signed_in_expires"] = datetime.utcnow() + timedelta(
                hours=48
            )
        st.markdown(f"**Current user:** `{st.session_state['username']}`")

        st.markdown("---")
        menu = st.radio(
            "Menu",
            ["New study session", "History"],
            index=0,
        )

        st.markdown("---")
        st.header("Upload & Index")
        st.write(f"Study folder on disk: `{STUDY_DIR}`")

        uploaded_files = st.file_uploader(
            "Upload study materials (PDF / TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for f in uploaded_files:
                save_path = STUDY_DIR / f.name
                try:
                    with open(save_path, "wb") as out:
                        out.write(f.getbuffer())
                    record_upload(
                        filename=f.name,
                        original_name=f.name,
                        size_bytes=len(f.getbuffer()),
                    )
                except Exception as e:
                    st.error(f"Failed to save {f.name}: {e}")
            st.success("Files uploaded successfully. Remember to rebuild the index.")

        if st.button("Rebuild index from uploaded materials"):
            with st.spinner("Building index from study materials..."):
                try:
                    ingest()
                    st.success("Index rebuilt successfully.")
                except Exception as e:
                    st.error(f"Error while ingesting study materials: {e}")

        uploads = load_uploads()
        if uploads:
            st.markdown("**Recently uploaded files (from database):**")
            for filename, original_name, size_bytes, uploaded_at in uploads[:10]:
                st.write(
                    f"- `{original_name}` ({size_bytes} bytes) – stored as `{filename}` at {uploaded_at}"
                )

    # Main content area depends on selected menu
    if menu == "New study session":
        st.subheader("New study session")
        st.caption("Like 'New chat' in ChatGPT – ask a fresh question.")

        if st.button("Clear question"):
            if "question_input" in st.session_state:
                st.session_state["question_input"] = ""

        question = st.text_area(
            "Your question",
            key="question_input",
            height=80,
            placeholder="Type your question about the uploaded study materials...",
        )

        if st.button("Get answer", type="primary") and question.strip():
            with st.spinner("Retrieving relevant passages and generating answer..."):
                try:
                    chunks = retrieve(question, k=5)
                    if not chunks:
                        st.warning(
                            "No relevant passages found in your study materials. "
                            "Make sure you uploaded files and rebuilt the index."
                        )
                        return

                    answer = answer_question(question, chunks)
                    # Save to history for current user
                    save_history(st.session_state["username"], question, answer)
                except Exception as e:
                    st.error(f"Error while answering the question: {e}")
                    return

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Show retrieved study material passages"):
                for i, ch in enumerate(chunks, start=1):
                    location = f"{ch.source}"
                    if ch.page is not None:
                        location += f", page {ch.page}"
                    st.markdown(f"**Passage {i} – {location}**")
                    st.write(ch.text)
                    st.markdown("---")

    elif menu == "History":
        st.subheader("History")
        st.caption(
            "View your previous questions and answers. You can pin important ones or clear history."
        )

        username = st.session_state.get("username", "guest")
        entries = load_history(username)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            if st.button("Refresh history"):
                entries = load_history(username)
        with col_right:
            if st.button("Clear history for this user"):
                clear_history(username)
                entries = []

        if not entries:
            st.info("No history yet for this user.")
        else:
            for entry in entries:
                hist_id, question, answer, pinned, created_at = entry
                pin_label = "Unpin" if pinned else "Pin"
                with st.expander(
                    f"{'📌 ' if pinned else ''}{created_at} – {question[:60]}{'...' if len(question) > 60 else ''}"
                ):
                    st.markdown(f"**Question**\n\n{question}")
                    st.markdown(f"**Answer**\n\n{answer}")
                    cols = st.columns(2)
                    with cols[0]:
                        if st.button(
                            pin_label,
                            key=f"pin_{hist_id}_{pinned}",
                        ):
                            set_pin(hist_id, not pinned)
                            # For Streamlit >= 1.x, use st.rerun(); older versions used experimental_rerun
                            st.rerun()
                    with cols[1]:
                        st.write(f"Created at: {created_at}")


if __name__ == "__main__":
    main()


