import streamlit as st
import numpy as np, pandas as pd
import scipy.io as sio
import mne
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# --- Load pre-trained model and encoder ---
model = load_model("cnn_lstm_model.h5")          
encoder = joblib.load("label_encoder.pkl")       

# --- Utility Functions ---
def load_mat_file(path):
    mat = sio.loadmat(path)
    o_struct = mat['o'][0,0]
    eeg_data = np.asarray(o_struct['data'])
    fs = int(o_struct['sampFreq'][0,0])
    return eeg_data, fs

def load_eeg_file(fname):
    if fname.endswith(".mat"):
        eeg_data, fs = load_mat_file(fname)
        return eeg_data, fs
    elif fname.endswith(".edf"):
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        data, sf = raw.get_data(), int(raw.info['sfreq'])
        return data.T, sf
    else:
        raise ValueError("Unsupported file format")

def bandpass_filter(signal, sf, low=1, high=50, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [low, high], btype='band', fs=sf, output='sos')
    return sosfiltfilt(sos, signal)

def segment_signal(signal, sf, window_sec=2, step_sec=1):
    epoch_len, step = int(window_sec*sf), int(step_sec*sf)
    return [signal[i:i+epoch_len] for i in range(0, len(signal)-epoch_len+1, step)]

#  Adaptive Engine 
def adaptive_engine(model, encoder, stream, sf=128, window_sec=2, step_sec=1):
    epoch_len = sf * window_sec
    step = sf * step_sec
    actions = []
    for start in range(0, len(stream) - epoch_len + 1, step):
        epoch = stream[start:start+epoch_len].reshape(1, epoch_len, 1)
        pred = model.predict(epoch).argmax(axis=1)[0]
        state = encoder.classes_[pred]
        if state == "Focused":
            action = "Maintain current difficulty"
        elif state == "Distracted":
            action = "Switch to interactive exercise"
        elif state == "Fatigued":
            action = "Suggest short break"
        elif state == "Stressed":
            action = "Offer calming material"
        else:
            action = "Neutral state — no change"
        actions.append((state, action))
    return actions

#  Streamlit UI 
st.title("EEG Live Prediction Tool")

tab1, tab2 = st.tabs(["📂 Offline Mode", "📡 Real-Time Mode"])

#  Offline Mode 
with tab1:
    uploaded_file = st.file_uploader("Upload EEG file", type=["mat","edf"])
    if uploaded_file:
        data, sf = load_eeg_file(uploaded_file.name)
        ch = bandpass_filter(data[:,0], sf)
        st.line_chart(ch[:sf*10])  # plot first 10 seconds

        st.subheader("Predictions & Recommendations")
        actions = adaptive_engine(model, encoder, ch[:sf*20], sf)
        states = [s for s,_ in actions]
        recs = [a for _,a in actions]

        # Timeline chart of states
        st.line_chart(pd.Series(states, name="Predicted State"))

        # Table of recommendations
        st.write(pd.DataFrame({"State": states, "Recommendation": recs}))

# --- Real-Time Mode ---
with tab2:
    st.write("Connect your EEG headset via Lab Streaming Layer (LSL).")
    st.write("Live predictions will appear below once streaming starts.")

    st.code("""
from pylsl import StreamInlet, resolve_stream
streams = resolve_stream('type','EEG')
inlet = StreamInlet(streams[0])
buffer = []
sf = 128
epoch_len = sf*2
predictions = []
while True:
    sample,_ = inlet.pull_sample()
    buffer.extend(sample)
    if len(buffer) >= epoch_len:
        epoch = np.array(buffer[-epoch_len:]).reshape(1,epoch_len,1)
        pred = model.predict(epoch).argmax(axis=1)[0]
        state = encoder.classes_[pred]
        recommendation = adaptive_engine(model, encoder, buffer[-epoch_len:], sf)[-1][1]
        predictions.append(pred)
        print("State:", state, "→", recommendation)
        buffer = buffer[-epoch_len:]
    """)
