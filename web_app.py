import hashlib
import os
import sqlite3
import tempfile
import subprocess
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
    # Users table for auth
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(email, password):
    if email == "admin" and password == "myadminsecret":
        return True, "admin"
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if row and row[0] == hash_password(password):
        return True, email
    return False, None

def create_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, hash_password(password), datetime.utcnow().isoformat())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
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
    st.write("Please **sign in or sign up** to start using the application.")

    choice = st.radio("Select an option", ["Sign In", "Sign Up", "Admin Login"], horizontal=True)

    if choice == "Admin Login":
        st.info("Log in using the admin password.")
        email_input = "admin"
        password_input = st.text_input("Admin Password", type="password", key="auth_pass_input_admin")
    else:
        email_input = st.text_input("Email / Gmail", key="auth_email_input")
        password_input = st.text_input("Password", type="password", key="auth_pass_input")

    if st.button("Continue"):
        email = email_input.strip()
        pwd = password_input.strip()
        
        if choice == "Admin Login":
            if pwd == "myadminsecret":
                st.session_state["username"] = "admin"
                st.session_state["is_admin"] = True
                st.session_state["signed_in_expires"] = now + timedelta(hours=48)
                st.success("Signed in as Admin!")
                st.rerun()
            else:
                st.error("Invalid admin credentials.")
        else:
            if not email or not pwd:
                st.warning("Please enter both email and password.")
            elif choice == "Sign Up":
                if create_user(email, pwd):
                    st.success("Account created successfully. You can now switch to 'Sign In' and log in.")
                else:
                    st.error("An account with this email/Gmail already exists.")
            elif choice == "Sign In":
                ok, user_email = authenticate_user(email, pwd)
                if ok:
                    st.session_state["username"] = user_email
                    st.session_state["is_admin"] = (user_email == "admin")
                    st.session_state["signed_in_expires"] = now + timedelta(hours=48)
                    st.success(f"Signed in as `{user_email}`.")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
                    
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
        st.subheader("Account")
        st.markdown(f"**Current user:** `{st.session_state.get('username', 'guest')}`")
        if st.session_state.get("is_admin"):
            st.markdown("*(Admin Mode)*")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        menu_options = ["New study session", "History"]
        if st.session_state.get("is_admin"):
            menu_options.append("Manage GitHub")

        menu = st.radio(
            "Menu",
            menu_options,
            index=0,
        )

        st.markdown("---")
        
        st.header("Upload & Index")
        st.write(f"Study folder on disk: `{STUDY_DIR}`")

        existing_files = []
        if os.path.exists(STUDY_DIR):
            for fname in os.listdir(STUDY_DIR):
                if fname.lower().endswith(".pdf") or fname.lower().endswith(".txt"):
                    existing_files.append(fname)
        
        if existing_files:
            st.markdown("**Currently uploaded files:**")
            for fname in sorted(existing_files):
                st.markdown(f"- `{fname}`")

        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 1

        uploaded_files = st.file_uploader(
            "Upload study materials (PDF / TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=str(st.session_state["file_uploader_key"]),
        )

        if uploaded_files:
            new_files = False
            for f in uploaded_files:
                save_path = STUDY_DIR / f.name
                if not os.path.exists(save_path):
                    try:
                        with open(save_path, "wb") as out:
                            out.write(f.getbuffer())
                        record_upload(
                            filename=f.name,
                            original_name=f.name,
                            size_bytes=len(f.getbuffer()),
                        )
                        new_files = True
                    except Exception as e:
                        st.error(f"Failed to save {f.name}: {e}")
            if new_files:
                st.success("Files uploaded successfully. You can upload more files step by step.")
                st.session_state["file_uploader_key"] += 1
                st.rerun()

        if st.button("Rebuild index from uploaded materials"):
            with st.spinner("Building index from study materials..."):
                try:
                    ingest()
                    st.success("Index rebuilt successfully.")
                except Exception as e:
                    st.error(f"Error while ingesting study materials: {e}")



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
                        
    elif menu == "Manage GitHub" and st.session_state.get("is_admin"):
        st.subheader("Manage GitHub Repository (Admin Only)")
        st.caption("Sync your latest changes and study materials to GitHub directly from the app.")
        
        commit_msg = st.text_input("Commit Message", value="Update study materials and application files")
        
        if st.button("Commit and Push to GitHub"):
            with st.spinner("Syncing to GitHub..."):
                try:
                    subprocess.run(["git", "add", "."], check=True, capture_output=True, text=True)
                    res_commit = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
                    res_push = subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True, text=True)
                    
                    st.success("Successfully pushed to GitHub!")
                    with st.expander("Show Logs"):
                        st.text("Commit Log:\n" + res_commit.stdout + res_commit.stderr)
                        st.text("Push Log:\n" + res_push.stdout + res_push.stderr)
                except subprocess.CalledProcessError as e:
                    st.error("Failed to sync with GitHub.")
                    with st.expander("Show Error Details"):
                        st.text(e.stderr or e.stdout)


if __name__ == "__main__":
    main()


