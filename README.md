# LSTM-Based-Next-Word-Prediction

This project implements a deep learning model using Long Short-Term Memory (LSTM) for next-word prediction. The model is trained on Shakespeare's "Hamlet" dataset and uses the TensorFlow and Keras libraries for building and training the LSTM network. The project also incorporates early stopping to optimize model training and prevent overfitting.

Features:

-Preprocessing of text data from the Gutenberg corpus.

-Tokenization and creation of input sequences for LSTM training.

-LSTM-based model architecture with two LSTM layers and dropout for regularization.

-Use of categorical cross-entropy loss and Adam optimizer.

-Early stopping to monitor validation loss and halt training when improvement plateaus.

-A Streamlit web app interface for real-time next-word prediction, allowing users to input a sequence of words and predict the next one.

Technologies:
Python (TensorFlow, Keras, NLTK, NumPy, Streamlit)
LSTM for sequential text prediction
Streamlit for the user interface

Files:
app.py: Streamlit app for user interaction and next-word prediction.

LTSM.py: Model training script that preprocesses data, trains the LSTM model, and includes early stopping.

The Next Word Prediction Model Using LSTM project has several notable impacts, both in terms of technical value and real-world application:

1. Advancement in Natural Language Processing (NLP):
The project demonstrates how LSTM, a powerful deep learning architecture, can effectively handle sequential data and capture the context in text, contributing to advancements in the field of NLP.
By working with classical literature like Hamlet, it showcases the potential of LSTM models in understanding and generating human language in a meaningful way.

2.Educational Value:
By making the model code and architecture available, the project serves as a learning resource for aspiring data scientists or developers interested in deep learning and NLP.
It can be used in educational settings to demonstrate sequential modeling concepts, especially in courses related to AI, deep learning, or language processing.

3.Automation of Content Generation:
This technology can be applied to content creation, particularly in fields where writing assistance tools or automated content generation are valuable, such as journalism, marketing, or creative writing.
next_word_lstm.h5: The trained model file for next-word prediction.
tokenizer.pickle: Saved tokenizer for consistent text processing in predictions.
