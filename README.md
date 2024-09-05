# LSTM-Based-Next-Word-Prediction

This project implements a deep learning model using Long Short-Term Memory (LSTM) for next-word prediction. The model is trained on Shakespeare's "Hamlet" dataset and uses the TensorFlow and Keras libraries for building and training the LSTM network. The project also incorporates early stopping to optimize model training and prevent overfitting.

Features:
Preprocessing of text data from the Gutenberg corpus.
Tokenization and creation of input sequences for LSTM training.
LSTM-based model architecture with two LSTM layers and dropout for regularization.
Use of categorical cross-entropy loss and Adam optimizer.
Early stopping to monitor validation loss and halt training when improvement plateaus.
A Streamlit web app interface for real-time next-word prediction, allowing users to input a sequence of words and predict the next one.
Technologies:
Python (TensorFlow, Keras, NLTK, NumPy, Streamlit)
LSTM for sequential text prediction
Streamlit for the user interface
Files:
app.py: Streamlit app for user interaction and next-word prediction.
LTSM.py: Model training script that preprocesses data, trains the LSTM model, and includes early stopping.
next_word_lstm.h5: The trained model file for next-word prediction.
tokenizer.pickle: Saved tokenizer for consistent text processing in predictions.
