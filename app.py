import streamlit as st
import time
import pandas as pd

from src.ngram_model import NGramModel
from src.gpt2_engine import GPT2Engine
from src.hybrid_engine import HybridEngine

st.set_page_config(page_title="Email Autocomplete", page_icon="📧", layout="wide")


@st.cache_resource
def load_models(model_type):
    with open("data/corpus.txt") as f:
        corpus = f.readlines()

    ngram = NGramModel()
    ngram.train(corpus)

    if model_type == "Fine-Tuned GPT-2":
        gpt2 = GPT2Engine("ruchita77/fine-tuned-gpt2-email")  # <-- your HF repo
    else:
        gpt2 = GPT2Engine("gpt2")
   


st.title("📧 AI Email Autocomplete")

with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox("Model Type", ["Base GPT-2", "Fine-Tuned GPT-2"])
    mode = st.selectbox("Mode", ["Hybrid", "NLP", "GPT-2"])
    tone = st.radio("Tone", ["Formal", "Casual"])
    st.divider()
    st.caption("Fine-tuned model is trained on Topic -> Email pairs for accurate full-email generation.")

engine = load_models(model_type)

tab1, tab2, tab3 = st.tabs(["✏️ Line Autocomplete", "📝 Full Email Generator", "📊 Compare"])

# --- Tab 1: line-by-line autocomplete ---
with tab1:
    if "draft" not in st.session_state:
        st.session_state.draft = ""

    text = st.text_area(
        "Compose your email",
        value=st.session_state.draft,
        height=150,
        key="draft_input"
    )

    if text.strip():
        start = time.time()
        results = engine.predict(text, mode, tone)
        elapsed = time.time() - start

        st.subheader("Suggestions")
        if results:
            cols = st.columns(len(results))
            for i, (res, score) in enumerate(results):
                with cols[i]:
                    if st.button(f"➕ {res[-50:]}", key=f"sugg_{i}"):
                        st.session_state.draft = res
                        st.rerun()
                    st.caption(f"score: {score:.2f}")
        else:
            st.write("No suggestions yet — keep typing.")

        st.info(f"⏱ {elapsed:.2f}s")

# --- Tab 2: full email from a topic ---
with tab2:
    topic = st.text_input(
        "What is the email about?",
        placeholder="e.g. leave approval, schedule a meeting, thank you for the interview"
    )
    if st.button("Generate Email"):
        if topic.strip():
            with st.spinner("Drafting..."):
                draft = engine.generate_full_email(topic, tone)
            if draft:
                st.text_area("Generated Draft", value=draft, height=250)
            else:
                st.warning("Model returned an empty result — try a shorter, simpler topic.")
        else:
            st.warning("Please enter a topic first.")

# --- Tab 3: side-by-side comparison across Model Type and Mode ---
with tab3:
    st.subheader("Compare Model Type × Mode")
    st.caption("Runs the same input across every combination below so you can see quality and speed differences directly, instead of switching the sidebar back and forth.")

    compare_input_type = st.radio(
        "Compare using:",
        ["Line Autocomplete (partial sentence)", "Full Email (topic)"],
        horizontal=True
    )

    model_types_to_test = st.multiselect(
        "Model Types to compare",
        ["Base GPT-2", "Fine-Tuned GPT-2"],
        default=["Base GPT-2", "Fine-Tuned GPT-2"]
    )

    modes_to_test = st.multiselect(
        "Modes to compare",
        ["NLP", "GPT-2", "Hybrid"],
        default=["NLP", "GPT-2", "Hybrid"]
    )

    compare_tone = st.radio("Tone for comparison", ["Formal", "Casual"], horizontal=True, key="compare_tone")

    if compare_input_type == "Line Autocomplete (partial sentence)":
        compare_text = st.text_input("Partial sentence to complete", placeholder="e.g. i am")
    else:
        compare_text = st.text_input("Topic for the email", placeholder="e.g. leave approval")

    if st.button("Run Comparison"):
        if not compare_text.strip():
            st.warning("Please enter some text first.")
        elif not model_types_to_test or not modes_to_test:
            st.warning("Select at least one Model Type and one Mode.")
        else:
            rows = []
            with st.spinner("Running all combinations..."):
                for mt in model_types_to_test:
                    test_engine = load_models(mt)

                    if compare_input_type == "Line Autocomplete (partial sentence)":
                        for md in modes_to_test:
                            start = time.time()
                            results = test_engine.predict(compare_text, md, compare_tone)
                            elapsed = time.time() - start
                            output_preview = " | ".join([r[0] for r in results]) if results else "(no suggestions)"
                            rows.append({
                                "Model Type": mt,
                                "Mode": md,
                                "Output": output_preview,
                                "Time (s)": round(elapsed, 2)
                            })
                    else:
                        # Full email generation only uses GPT-2 under the hood,
                        # so mode selection doesn't change it — run once per model type.
                        start = time.time()
                        draft = test_engine.generate_full_email(compare_text, compare_tone)
                        elapsed = time.time() - start
                        rows.append({
                            "Model Type": mt,
                            "Mode": "GPT-2 (full email always uses GPT-2)",
                            "Output": draft if draft else "(empty result)",
                            "Time (s)": round(elapsed, 2)
                        })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)