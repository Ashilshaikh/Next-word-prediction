import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Next Word Prediction", page_icon="🔤", layout="centered")

# ---------------------------
# Load model, tokenizer, max_len (cached so they load only once)
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pickle", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_artifacts()

# Reverse index to map predicted integer back to word
index_to_word = {index: word for word, index in tokenizer.word_index.items()}


def predict_next_words(seed_text, num_words, model, tokenizer, max_len):
    """Generate `num_words` next words after the seed_text."""
    result_text = seed_text

    for _ in range(num_words):
        # Convert text to sequence
        token_list = tokenizer.texts_to_sequences([result_text])[0]

        # Pad sequence to match model's expected input length
        # NOTE: if your model was trained with max_len as (max_sequence_len - 1),
        # change max_len below to max_len - 1
        token_list = pad_sequences([token_list], maxlen=max_len, padding='pre')

        # Predict probabilities for next word
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        # Map index back to word
        next_word = index_to_word.get(predicted_index, "")

        if next_word == "":
            break

        result_text += " " + next_word

    return result_text


# ---------------------------
# UI
# ---------------------------
st.title("🔤 Next Word Prediction")
st.write("Enter a starting phrase and let the model predict the next word(s).")

seed_text = st.text_input("Enter your text:", placeholder="e.g. I am going to")

num_words = st.slider("How many words to predict?", min_value=1, max_value=20, value=1)

if st.button("Predict"):
    if seed_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Predicting..."):
            output = predict_next_words(seed_text, num_words, model, tokenizer, max_len)
        st.success("Prediction complete!")
        st.markdown("### Result:")
        st.write(output)

st.markdown("---")
st.caption("Model: Next Word Prediction | Built with Streamlit, TensorFlow/Keras")