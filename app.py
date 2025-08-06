import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the pre-trained model
model = load_model('next_word_lstm.h5')

## Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

## Function to predict the next word
def predict_next_word(model, tokenizer, input_text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([input_text])[0]

    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length-1):] # Ensure the sequence length matches max_sequence - 1
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# streamlit app layout
st.title("Next Word Prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter a sentence:", "")

if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] + 1  # Get the max sequence length from the model input shape
    predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    st.write("Predicted next word:", predicted_word)    
else:
    st.write("Enter a sentence and click 'Predict Next Word' to see the next word prediction.")