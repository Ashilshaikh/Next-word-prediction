# Next Word Prediction

A Streamlit app for next-word prediction using a trained Keras LSTM model.

## Files

- `app.py` - Streamlit application entry point.
- `lstm_model.h5` - Trained Keras model weights.
- `tokenizer.pickle` - Tokenizer used for text-to-sequence conversion.
- `max_len.pickle` - Maximum sequence length used for padding.

## Setup

1. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install streamlit tensorflow
   ```
3. Run the app:
   ```powershell
   python -m streamlit run app.py
   ```

## Notes

- Make sure the model and tokenizer files are present in the project root.
- Use the workspace virtual environment so the correct Python packages are loaded.
 To run use
 https://next-word-prediction-2-jwrz.onrender.com
