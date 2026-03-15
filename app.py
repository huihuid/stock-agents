import os
import streamlit as st

import mp3_agents as ag  # <-- your extracted notebook code

st.set_page_config(page_title="MP3 Streamlit Chat", page_icon="💬", layout="wide")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    arch = st.selectbox("Agent selector", ["Single Agent", "Multi-Agent"], index=0)
    model = st.selectbox("Model selector", [ag.MODEL_SMALL, ag.MODEL_LARGE], index=0)

    st.markdown("---")
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.renders = []
        st.rerun()

st.title("💬 MP3 Deployment Chat (Notebook Agents)")

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    # store raw chat history (user + assistant)
    st.session_state.messages = []  # list[{"role":"user"/"assistant","content":...}]
if "renders" not in st.session_state:
    # store metadata for display, aligned with messages
    st.session_state.renders = []   # list[{"arch":..., "model":..., "tools":[...], "conf":..., "issues":[...]}]

# ----------------------------
# Render history
# ----------------------------
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        meta = st.session_state.renders[i] if i < len(st.session_state.renders) else {}
        with st.chat_message("assistant"):
            st.caption(
                f"Architecture: **{meta.get('arch','?')}** | Model: **{meta.get('model','?')}**"
            )
            # optional tool/conf display
            tools = meta.get("tools", [])
            conf  = meta.get("conf", None)
            issues = meta.get("issues", [])
            if tools:
                st.caption(f"Tools: {', '.join(tools)}")
            if conf is not None:
                st.caption(f"Confidence: {conf}")
            if issues:
                st.caption(f"Issues: {', '.join(issues)}")

            st.markdown(content)

# ----------------------------
# Chat input
# ----------------------------
user_text = st.chat_input("Ask a question...")
if user_text:
    # 1) append user
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.renders.append({"arch": arch, "model": model})

    # 2) set model in notebook code
    ag.set_active_model(model)

    # 3) build task with history (core memory requirement)
    task = ag.build_task_with_history(st.session_state.messages, user_text, max_turns=6)

    # 4) call the selected architecture from notebook
    if arch == "Single Agent":
        res = ag.run_single_agent(task, verbose=False)
        answer_text = res.answer

        meta = {
            "arch": arch,
            "model": model,
            "tools": res.tools_called,
            "conf": None,
            "issues": [],
        }

    else:
        out = ag.run_multi_agent(task, verbose=False)
        answer_text = out["final_answer"]

        # aggregate metadata from specialists
        agent_results = out.get("agent_results", [])
        tools = [t for r in agent_results for t in r.tools_called]
        issues = [iss for r in agent_results for iss in r.issues_found]
        avg_conf = None
        if agent_results:
            avg_conf = f"{(sum(r.confidence for r in agent_results)/len(agent_results)):.0%}"

        meta = {
            "arch": f"Multi-Agent ({out.get('architecture','')})",
            "model": model,
            "tools": tools,
            "conf": avg_conf,
            "issues": issues,
        }

    # 5) append assistant
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.session_state.renders.append(meta)

    st.rerun()