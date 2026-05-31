import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Self-Attention Visualizer", layout="wide")

# -----------------------------
# Data
# -----------------------------
vocab = {
    "alice": np.array([10, 0, 0]),
    "tom": np.array([10, 0, 0]),

    "her": np.array([0, 9.5, 0]),
    "favorite": np.array([0, 10, 0]),
    "the": np.array([0, 3, 0]),
    "bedside": np.array([0, 10, 0]),
    "dining": np.array([0, 10, 0]),
    "neighbor's": np.array([0, 10, 0]),

    "book": np.array([9, 0, 0]),
    "table": np.array([9, 0, 0]),
    "garden": np.array([9, 0, 0]),

    "on": np.array([0, 0, 10]),
    "in": np.array([0, 0, 10]),

    "put": np.array([0, 0, 0]),
    "is": np.array([0, 0, 0]),
    "are": np.array([0, 0, 0]),
    "sitting": np.array([0, 0, 0]),
    "playing": np.array([0, 0, 0]),
    "and": np.array([0, 0, 0]),
}

sentences = {
    "Tom is in the garden.": ["tom", "is", "in", "the", "garden"],
    "Alice put her favorite book on the bedside table.": [
        "alice", "put", "her", "favorite", "book", "on", "the", "bedside", "table"
    ],
    "Alice is sitting on the dining table.": [
        "alice", "is", "sitting", "on", "the", "dining", "table"
    ],
    "Alice and Tom are playing in the neighbor's garden.": [
        "alice", "and", "tom", "are", "playing", "in", "the", "neighbor's", "garden"
    ],
}


# -----------------------------
# Attention helpers
# -----------------------------
def get_W(head_name: str):
    if head_name == "description":
        W_Q = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=float)

        W_K = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=float)

        W_V = np.eye(3)
        default_bias = 4.0

    elif head_name == "position":
        W_Q = np.array([
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=float)

        W_K = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ], dtype=float)

        W_V = np.eye(3)
        default_bias = 4.0

    else:
        W_Q = np.eye(3)
        W_K = np.eye(3)
        W_V = np.eye(3)
        default_bias = 0.0

    return W_Q, W_K, W_V, default_bias


def get_softmax(X, axis=1):
    X_shifted = X - np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X_shifted)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)


def get_attention(Q, K, V, mask=True, bias_strength=0.0):
    raw_scores = Q @ K.T

    biased_scores = raw_scores.copy()
    if bias_strength:
        positions = np.arange(raw_scores.shape[0])
        distance = np.abs(positions[:, None] - positions[None, :])
        biased_scores = biased_scores - bias_strength * distance

    scaled_scores = biased_scores / np.sqrt(Q.shape[1])

    masked_scores = scaled_scores.copy()
    if mask:
        seq_len = masked_scores.shape[0]
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        masked_scores[causal_mask] = -np.inf

    probs = get_softmax(masked_scores)
    out = probs @ V

    return raw_scores, biased_scores, scaled_scores, masked_scores, probs, out


def head_description(head_name: str):
    if head_name == "identity":
        return "Uses identity projections. This mostly shows similarity between the original token embeddings."
    if head_name == "description":
        return "Designed to make entity-like tokens attend to descriptive tokens."
    if head_name == "position":
        return "Designed to make entity-like tokens attend to preposition-like tokens."
    return ""


def style_matrix(df: pd.DataFrame, is_probs: bool):
    df_vis = df.replace([np.inf, -np.inf], np.nan)

    kwargs = {
        "cmap": "viridis",
        "axis": None,
    }

    if is_probs:
        kwargs.update(vmin=0, vmax=1)

    return (
        df_vis.style
        .background_gradient(**kwargs)
        .format("{:.4f}")
        .highlight_null(color="#eeeeee")
    )


def show_matrix(title: str, matrix, index=None, columns=None, decimals=3):
    st.markdown(f"**{title}**")
    st.dataframe(
        pd.DataFrame(matrix, index=index, columns=columns).round(decimals),
        use_container_width=True,
    )


# -----------------------------
# Header
# -----------------------------
st.title("Self-Attention Visualizer")

st.caption("""
**M4ML: Project 2**  
Project Topic 7: *Mathematical Principles of Large Language Models (LLMs)*  
Authors: Diogo Manuel Cerieiro dos Santos (93635), Matěj Polák (119339)  
June 2, 2026
""")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Sentence")
    sentence_name = st.selectbox(
        "Choose sentence",
        list(sentences.keys()),
        index=None,
        placeholder="Select a sentence...",
    )

    st.subheader("Attention head")
    head_display = st.selectbox(
        "Choose head",
        ["Identity", "Description", "Position"],
        index=None,
        placeholder="Select a head...",
    )

    st.subheader("Attention options")
    use_mask = st.checkbox("Causal mask", value=False)
    use_distance_bias = st.checkbox("Distance bias", value=False)
    show_softmax = st.checkbox("Show softmax probabilities", value=False)

# -----------------------------
# 1. Sentence section
# -----------------------------
if sentence_name is None:
    st.info("Please select a sentence in the sidebar.")
    st.stop()

sentence = sentences[sentence_name]
X = np.array([vocab[word] for word in sentence], dtype=float)

with st.container(border=True):
    st.markdown(f"**Original sentence:** {sentence_name}")
    st.markdown(f"**Tokenized sentence:** `{sentence}`")

    with st.expander("Show input embeddings X", expanded=False):
        show_matrix(
            "Input embeddings X",
            X,
            index=sentence,
            columns=["dim_1", "dim_2", "dim_3"],
        )

# -----------------------------
# 2. Attention head section
# -----------------------------
if head_display is None:
    st.info("Now select an attention head in the sidebar.")
    st.stop()

head_name = head_display.lower()

W_Q, W_K, W_V, default_bias = get_W(head_name)
bias_strength = default_bias if use_distance_bias else 0.0

with st.container(border=True):
    c1, c2, c3 = st.columns(3)

    c1.metric("**Selected head**", head_display)
    c2.metric("Distance bias", bias_strength)
    c3.metric("Causal mask", str(use_mask))

    with st.expander("Show head weights W_Q, W_K, W_V", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            show_matrix("W_Q", W_Q)

        with c2:
            show_matrix("W_K", W_K)

        with c3:
            show_matrix("W_V", W_V)

# -----------------------------
# Attention computation
# -----------------------------
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    raw_scores, biased_scores, scaled_scores, masked_scores, probs, out = get_attention(
        Q,
        K,
        V,
        mask=use_mask,
        bias_strength=bias_strength,
    )

# -----------------------------
# 3. Projected matrices
# -----------------------------

    with st.expander("Show Q, K, V", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            show_matrix("Q = XW_Q", Q, index=sentence)

        with c2:
            show_matrix("K = XW_K", K, index=sentence)

        with c3:
            show_matrix("V = XW_V", V, index=sentence)

# -----------------------------
# 4. Attention matrix
# -----------------------------
st.header("Attention matrix")

matrix = probs if show_softmax else masked_scores

df = pd.DataFrame(matrix, index=sentence, columns=sentence)

st.dataframe(
    style_matrix(df, is_probs=show_softmax),
    use_container_width=True,
)

with st.expander("Show intermediate score matrices", expanded=False):
    show_matrix("Raw scores QKᵀ", raw_scores, index=sentence, columns=sentence)
    show_matrix("Scores after distance bias", biased_scores, index=sentence, columns=sentence)
    show_matrix("Scaled scores", scaled_scores, index=sentence, columns=sentence)

    st.markdown("**Masked scaled scores**")
    masked_df = pd.DataFrame(masked_scores, index=sentence, columns=sentence)
    st.dataframe(
        masked_df.replace([np.inf, -np.inf], np.nan).round(3),
        use_container_width=True,
    )

# -----------------------------
# 5. Output
# -----------------------------

with st.container(border=True):

    st.subheader("Output contextualized embeddings")

    col1, col2 = st.columns(2)

    with col1:
        show_matrix(
            "Original embeddings X",
            X,
            index=sentence,
            columns=["dim_1", "dim_2", "dim_3"],
        )

    with col2:
        show_matrix(
            "Output embeddings Att(Q,K,V)",
            out,
            index=sentence,
        )