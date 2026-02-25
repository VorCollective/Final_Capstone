{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c66bdf19",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, random\n",
    "from pathlib import Path\n",
    "import numpy as np, pandas as pd\n",
    "import matplotlib.pyplot as plt, seaborn as sns\n",
    "import scipy.io as sio\n",
    "from scipy.signal import butter, sosfiltfilt, lfilter, welch, filtfilt\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "import mne\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau\n",
    "from scipy.stats import entropy\n",
    "\n",
    "SEED = 42\n",
    "random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7bb7d58",
   "metadata": {},
   "source": [
    "Step 1-3: Data Acquisition & Splitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "69057c16",
   "metadata": {},
   "outputs": [],
   "source": [
    "folder1 = Path(r\"C:\\Users\\Cyan\\Desktop\\DataScience_Notes\\DataScience_Capstone\\EEG Data\")\n",
    "folder2 = Path(r\"C:\\Users\\Cyan\\Desktop\\DataScience_Notes\\DataScience_Capstone\\eeg-during-mental-arithmetic-tasks-1.0.0\")\n",
    "\n",
    "all_files = [str(f) for f in folder1.glob(\"*.mat\")] + [str(f) for f in folder1.glob(\"*.edf\")]\n",
    "all_files += [str(f) for f in folder2.glob(\"*.mat\")] + [str(f) for f in folder2.glob(\"*.edf\")]\n",
    "\n",
    "random.shuffle(all_files)\n",
    "n_total = len(all_files)\n",
    "train_pct, val_pct, test_pct = 0.70, 0.15, 0.15\n",
    "n_train, n_val = int(train_pct*n_total), int(val_pct*n_total)\n",
    "train_files, val_files, test_files = all_files[:n_train], all_files[n_train:n_train+n_val], all_files[n_train+n_val:]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "651c2f05",
   "metadata": {},
   "source": [
    "STEP 4: Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b78b9500",
   "metadata": {},
   "outputs": [],
   "source": [
    "def bandpass_filter(signal, sf, low=1, high=50, order=4):\n",
    "    sos = butter(order, [low, high], btype='band', fs=sf, output='sos')\n",
    "    return sosfiltfilt(sos, signal)\n",
    "\n",
    "def segment_signal(signal, sf, window_sec=2, step_sec=None):\n",
    "    if step_sec is None: step_sec = window_sec\n",
    "    epoch_len, step = int(window_sec*sf), int(step_sec*sf)\n",
    "    return [signal[i:i+epoch_len] for i in range(0, len(signal)-epoch_len+1, step)]\n",
    "\n",
    "def normalize_epoch(epoch):\n",
    "    return StandardScaler().fit_transform(epoch.reshape(-1,1)).flatten()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d7e2433",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example: raw vs filtered signal\n",
    "ch = np.random.randn(5000)  # placeholder signal\n",
    "sf = 250\n",
    "ch_filtered = bandpass_filter(ch, sf)\n",
    "t = np.arange(len(ch[:1000]))/sf\n",
    "\n",
    "plt.figure(figsize=(12,6))\n",
    "plt.plot(t, ch[:1000], label=\"Raw\", alpha=0.6)\n",
    "plt.plot(t, ch_filtered[:1000], label=\"Filtered\", alpha=0.9)\n",
    "plt.title(\"Raw vs Bandpass Filtered Signal\")\n",
    "plt.xlabel(\"Time (s)\")\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b476c76",
   "metadata": {},
   "source": [
    "STEP 5: Feature Extraction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9abde0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def bandpower(epoch, sf, band):\n",
    "    freqs, psd = welch(epoch, sf, nperseg=min(len(epoch), sf*2))\n",
    "    idx = (freqs >= band[0]) & (freqs <= band[1])\n",
    "    return float(np.trapezoid(psd[idx], freqs[idx]))\n",
    "\n",
    "def hjorth_params(epoch):\n",
    "    first_deriv, second_deriv = np.diff(epoch), np.diff(np.diff(epoch))\n",
    "    activity = np.var(epoch)\n",
    "    mobility = np.sqrt(np.var(first_deriv)/activity)\n",
    "    complexity = np.sqrt(np.var(second_deriv)/np.var(first_deriv))/mobility\n",
    "    return activity, mobility, complexity\n",
    "\n",
    "def spectral_entropy(epoch, sf):\n",
    "    freqs, psd = welch(epoch, sf)\n",
    "    psd_norm = psd/np.sum(psd)\n",
    "    return entropy(psd_norm)\n",
    "\n",
    "def extract_features(epoch, sf):\n",
    "    hj_activity, hj_mobility, hj_complexity = hjorth_params(epoch)\n",
    "    return {\n",
    "        \"theta\": bandpower(epoch, sf, (4,7)),\n",
    "        \"alpha\": bandpower(epoch, sf, (8,12)),\n",
    "        \"beta\":  bandpower(epoch, sf, (13,30)),\n",
    "        \"gamma\": bandpower(epoch, sf, (30,45)),\n",
    "        \"mean\":  float(np.mean(epoch)),\n",
    "        \"var\":   float(np.var(epoch)),\n",
    "        \"hjorth_activity\": hj_activity,\n",
    "        \"hjorth_mobility\": hj_mobility,\n",
    "        \"hjorth_complexity\": hj_complexity,\n",
    "        \"spectral_entropy\": spectral_entropy(epoch, sf)\n",
    "    }\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd9c390f",
   "metadata": {},
   "outputs": [],
   "source": [
    "epoch = np.random.randn(500)  # placeholder epoch\n",
    "sf = 250\n",
    "features = extract_features(epoch, sf)\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "plt.bar(features.keys(), features.values(), color=\"tab:orange\")\n",
    "plt.title(\"Extracted Features (Example Epoch)\")\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "391f0d43",
   "metadata": {},
   "source": [
    "STEP 6: Rule-based Labeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8d35f88",
   "metadata": {},
   "outputs": [],
   "source": [
    "def map_condition_rule(features):\n",
    "    theta, alpha, beta, gamma = features[\"theta\"], features[\"alpha\"], features[\"beta\"], features[\"gamma\"]\n",
    "    if beta > alpha and beta > theta: return 0  # Focused\n",
    "    if theta > beta: return 1                   # Distracted\n",
    "    if theta > alpha and alpha > 0: return 2    # Fatigued\n",
    "    if beta > 0 and gamma > 0: return 3         # Stressed\n",
    "    return 4                                    # Other\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d1708b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "labels = [map_condition_rule(extract_features(epoch, sf)) for epoch in [np.random.randn(500) for _ in range(20)]]\n",
    "plt.figure(figsize=(12,3))\n",
    "plt.plot(labels, marker=\"o\", linestyle=\"-\", color=\"tab:purple\")\n",
    "plt.title(\"Rule-based Condition Labels Across Epochs\")\n",
    "plt.yticks([0,1,2,3,4], [\"Focused\",\"Distracted\",\"Fatigued\",\"Stressed\",\"Other\"])\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8e59c4d",
   "metadata": {},
   "source": [
    "STEP 7: Multi-subject Data builder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "363b2bd1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_mat_file(path):\n",
    "    mat = sio.loadmat(path)\n",
    "    o_struct = mat['o'][0,0]\n",
    "    eeg_data = np.asarray(o_struct['data'])\n",
    "    fs = int(o_struct['sampFreq'][0,0])\n",
    "    subject_id = o_struct['id'][0]\n",
    "    markers = o_struct['marker'].flatten()\n",
    "    trials = o_struct['trials']\n",
    "    return eeg_data, fs, subject_id, markers, trials\n",
    "\n",
    "def build_dataset(files, window_sec=2):\n",
    "    X, y = [], []\n",
    "    for fname in files:\n",
    "        if fname.endswith(\".edf\"):\n",
    "            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)\n",
    "            data, sf = raw.get_data(), int(raw.info['sfreq'])\n",
    "            ch = bandpass_filter(data[0], sf)\n",
    "        elif fname.endswith(\".mat\"):\n",
    "            eeg_data, fs, subject_id, markers, trials = load_mat_file(fname)\n",
    "            ch, sf = bandpass_filter(eeg_data[:,0], fs), fs\n",
    "        else: continue\n",
    "\n",
    "        epochs = segment_signal(ch, sf, window_sec=window_sec, step_sec=1)\n",
    "        for ep in epochs:\n",
    "            feats = extract_features(ep, sf)\n",
    "            X.append(list(feats.values()))\n",
    "            y.append(map_condition_rule(feats))\n",
    "    return np.array(X), np.array(y)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7edc9a3a",
   "metadata": {},
   "source": [
    "STEP 8: Baseline Model(RandomForest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a351381c",
   "metadata": {},
   "outputs": [
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mThe Kernel crashed while executing code in the current cell or a previous cell. \n",
      "\u001b[1;31mPlease review the code in the cell(s) to identify a possible cause of the failure. \n",
      "\u001b[1;31mClick <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. \n",
      "\u001b[1;31mView Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details."
     ]
    },
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mFailed to interrupt the Kernel. \n",
      "\u001b[1;31mThe kernel died. Error: ... View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details."
     ]
    }
   ],
   "source": [
    "X_rf, y_rf = build_dataset(train_files)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)\n",
    "\n",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)\n",
    "y_pred = rf.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ee65d980",
   "metadata": {},
   "outputs": [],
   "source": [
    "importances = rf.feature_importances_\n",
    "plt.figure(figsize=(8,4))\n",
    "plt.bar(range(len(importances)), importances, color=\"tab:blue\")\n",
    "plt.title(\"Random Forest Feature Importance\")\n",
    "plt.show()\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\",\n",
    "            xticklabels=[\"Focused\",\"Distracted\",\"Fatigued\",\"Stressed\",\"Other\"],\n",
    "            yticklabels=[\"Focused\",\"Distracted\",\"Fatigued\",\"Stressed\",\"Other\"])\n",
    "plt.title(\"Confusion Matrix — Random Forest\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d231d4e",
   "metadata": {},
   "source": [
    "STEP 9: Deep Learning Model(CNN+LSTM)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "948149e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build CNN+LSTM dataset from raw signals\n",
    "X_dl, y_dl = [], []\n",
    "for fname in train_files:\n",
    "    if fname.endswith(\".edf\"):\n",
    "        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)\n",
    "        data, sf = raw.get_data(), int(raw.info['sfreq'])\n",
    "        ch = bandpass_filter(data[0], sf)\n",
    "    elif fname.endswith(\".mat\"):\n",
    "        eeg_data, fs, subject_id, markers, trials = load_mat_file(fname)\n",
    "        ch, sf = bandpass_filter(eeg_data[:,0], fs), fs\n",
    "    else: continue\n",
    "\n",
    "    epochs = segment_signal(ch, sf, window_sec=2, step_sec=1)\n",
    "    for ep in epochs:\n",
    "        if len(ep) >= sf*2:\n",
    "            feats = extract_features(ep, sf)\n",
    "            X_dl.append(ep[:sf*2].reshape(-1,1))\n",
    "            y_dl.append(map_condition_rule(feats))\n",
    "\n",
    "X_dl, y_dl = np.array(X_dl), np.array(y_dl)\n",
    "\n",
    "# Balanced synthetic augmentation if needed\n",
    "unique_labels = np.unique(y_dl)\n",
    "if len(unique_labels) < 2:\n",
    "    print(\"⚠️ Not enough label diversity. Generating balanced synthetic dataset...\")\n",
    "    synthetic_data, synthetic_labels = [], []\n",
    "    for label in range(5):\n",
    "        for _ in range(40):\n",
    "            ep = np.random.randn(sf*2)\n",
    "            synthetic_data.append(ep.reshape(-1,1))\n",
    "            synthetic_labels.append(label)\n",
    "    X_dl, y_dl = np.array(synthetic_data), np.array(synthetic_labels)\n",
    "\n",
    "# Encode labels\n",
    "encoder = LabelEncoder()\n",
    "y_enc = encoder.fit_transform(y_dl)\n",
    "y_cat = to_categorical(y_enc, num_classes=len(encoder.classes_))\n",
    "\n",
    "# Define CNN+LSTM\n",
    "model = Sequential([\n",
    "    Input(shape=(X_dl.shape[1],1)),\n",
    "    Conv1D(32, 3, activation='relu'),\n",
    "    MaxPooling1D(2),\n",
    "    LSTM(64, dropout=0.2, recurrent_dropout=0.2),\n",
    "    Dropout(0.5),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dense(len(encoder.classes_), activation='softmax')\n",
    "])\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy',"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e210f86",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train model\n",
    "history = model.fit(\n",
    "    X_dl, y_cat,\n",
    "    epochs=50, batch_size=16, validation_split=0.2,\n",
    "    callbacks=[\n",
    "        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),\n",
    "        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),\n",
    "        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)\n",
    "    ]\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0316e9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training history plots\n",
    "plt.figure(figsize=(12,4))\n",
    "plt.subplot(1,2,1)\n",
    "plt.plot(history.history['accuracy'], label='Train Acc')\n",
    "plt.plot(history.history['val_accuracy'], label='Val Acc')\n",
    "plt.legend(); plt.title(\"Accuracy\")\n",
    "\n",
    "plt.subplot(1,2,2)\n",
    "plt.plot(history.history['loss'], label='Train Loss')\n",
    "plt.plot(history.history['val_loss'], label='Val Loss')\n",
    "plt.legend(); plt.title(\"Loss\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2c9430a",
   "metadata": {},
   "source": [
    "STEP 10: Adaptive Learning Engine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb905d4a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def adaptive_action(state):\n",
    "    actions = {\n",
    "        0: \"Increase difficulty or continue with challenging tasks\",\n",
    "        1: \"Switch to interactive quizzes or visuals\",\n",
    "        2: \"Suggest a short break or lighter material\",\n",
    "        3: \"Introduce calming exercises or gamified rewards\",\n",
    "        4: \"Default content flow\"\n",
    "    }\n",
    "    return actions.get(state, \"Default content flow\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c2d64fef",
   "metadata": {},
   "outputs": [],
   "source": [
    "states = [0,1,2,3,4]\n",
    "actions = [adaptive_action(s) for s in states]\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "sns.barplot(x=[\"Focused\",\"Distracted\",\"Fatigued\",\"Stressed\",\"Other\"], y=states, palette=\"coolwarm\")\n",
    "plt.title(\"Adaptive Engine State Mapping\")\n",
    "plt.ylabel(\"State Code\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f884da35",
   "metadata": {},
   "source": [
    "Step 12: Model Evaluation Framework\n",
    "\n",
    "**Metrics to Use**\n",
    "\n",
    "- Accuracy: Overall correctness.\n",
    "\n",
    "- Precision, Recall, F1-score: Class-specific performance.\n",
    "\n",
    "- Confusion Matrix: Misclassification patterns.\n",
    "\n",
    "- Cross-validation: Robustness across folds.\n",
    "\n",
    "- Learning Curves: Overfitting/underfitting detection."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "73e6591f",
   "metadata": {},
   "source": [
    "Random Forest Eval"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4510fc3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Random Forest Evaluation\n",
    "y_pred_rf = rf.predict(X_test)\n",
    "acc_rf = accuracy_score(y_test, y_pred_rf)\n",
    "prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')\n",
    "\n",
    "print(f\"Random Forest - Accuracy: {acc_rf:.2f}, Precision: {prec_rf:.2f}, Recall: {rec_rf:.2f}, F1: {f1_rf:.2f}\")\n",
    "\n",
    "cm_rf = confusion_matrix(y_test, y_pred_rf)\n",
    "sns.heatmap(cm_rf, annot=True, fmt=\"d\", cmap=\"Blues\",\n",
    "            xticklabels=[\"Focused\",\"Distracted\",\"Fatigued\",\"Stressed\",\"Other\"],\n",
    "            yticklabels=[\"Focused\",\"Distracted\",\"Fatigued\",\"Stressed\",\"Other\"])\n",
    "plt.title(\"Confusion Matrix — Random Forest\")\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2e4a62d",
   "metadata": {},
   "source": [
    "CNN+LSTM Eval"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6bb48a4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# CNN+LSTM Evaluation\n",
    "y_pred_dl = model.predict(X_dl).argmax(axis=1)\n",
    "acc_dl = accuracy_score(y_enc, y_pred_dl)\n",
    "prec_dl, rec_dl, f1_dl, _ = precision_recall_fscore_support(y_enc, y_pred_dl, average='weighted')\n",
    "\n",
    "print(f\"CNN+LSTM - Accuracy: {acc_dl:.2f}, Precision: {prec_dl:.2f}, Recall: {rec_dl:.2f}, F1: {f1_dl:.2f}\")\n",
    "\n",
    "cm_dl = confusion_matrix(y_enc, y_pred_dl)\n",
    "sns.heatmap(cm_dl, annot=True, fmt=\"d\", cmap=\"Greens\",\n",
    "            xticklabels=encoder.classes_,\n",
    "            yticklabels=encoder.classes_)\n",
    "plt.title(\"Confusion Matrix — CNN+LSTM\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0135a29",
   "metadata": {},
   "source": [
    "Step 13: Comparative Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e5f65b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "models = [\"Random Forest\", \"CNN+LSTM\"]\n",
    "accuracy = [acc_rf, acc_dl]\n",
    "f1_scores = [f1_rf, f1_dl]\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "plt.bar(models, accuracy, color=[\"tab:blue\",\"tab:green\"])\n",
    "plt.title(\"Model Accuracy Comparison\")\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.show()\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "plt.bar(models, f1_scores, color=[\"tab:blue\",\"tab:green\"])\n",
    "plt.title(\"Model F1-score Comparison\")\n",
    "plt.ylabel(\"F1-score\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "918c5402",
   "metadata": {},
   "source": [
    "Step 14: Cross validation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8ce9562",
   "metadata": {},
   "source": [
    "Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "359f44bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Random Forest Cross-validation\n",
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "rf_scores = cross_val_score(rf, X_rf, y_rf, cv=cv, scoring='accuracy')\n",
    "print(\"Random Forest Cross-Validation Accuracy:\", rf_scores)\n",
    "print(\"Mean Accuracy:\", np.mean(rf_scores))\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "plt.plot(range(1, len(rf_scores)+1), rf_scores, marker=\"o\", color=\"tab:blue\")\n",
    "plt.axhline(np.mean(rf_scores), linestyle=\"--\", color=\"red\", label=\"Mean Accuracy\")\n",
    "plt.title(\"Random Forest Cross-Validation Accuracy (5 folds)\")\n",
    "plt.xlabel(\"Fold\")\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aceaef3c",
   "metadata": {},
   "source": [
    "CNN + LSTM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "932cb796",
   "metadata": {},
   "outputs": [],
   "source": [
    "# CNN+LSTM Cross-validation\n",
    "kf = KFold(n_splits=3, shuffle=True, random_state=42)\n",
    "dl_scores = []\n",
    "\n",
    "for train_idx, val_idx in kf.split(X_dl):\n",
    "    X_train, X_val = X_dl[train_idx], X_dl[val_idx]\n",
    "    y_train, y_val = y_cat[train_idx], y_cat[val_idx]\n",
    "\n",
    "    model = Sequential([\n",
    "        Input(shape=(X_train.shape[1],1)),\n",
    "        Conv1D(32, 3, activation='relu'),\n",
    "        MaxPooling1D(2),\n",
    "        LSTM(64, dropout=0.2, recurrent_dropout=0.2),\n",
    "        Dropout(0.5),\n",
    "        Dense(64, activation='relu'),\n",
    "        Dense(y_train.shape[1], activation='softmax')\n",
    "    ])\n",
    "    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "    history = model.fit(X_train, y_train, epochs=20, batch_size=16,\n",
    "                        validation_data=(X_val, y_val), verbose=0)\n",
    "    \n",
    "    val_acc = history.history['val_accuracy'][-1]\n",
    "    dl_scores.append(val_acc)\n",
    "\n",
    "print(\"CNN+LSTM Cross-Validation Accuracy:\", dl_scores)\n",
    "print(\"Mean Accuracy:\", np.mean(dl_scores))\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "plt.plot(range(1, len(dl_scores)+1), dl_scores, marker=\"o\", color=\"tab:green\")\n",
    "plt.axhline(np.mean(dl_scores), linestyle=\"--\", color=\"red\", label=\"Mean Accuracy\")\n",
    "plt.title(\"CNN+LSTM Cross-Validation Accuracy (3 folds)\")\n",
    "plt.xlabel(\"Fold\")\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "433bfa5b",
   "metadata": {},
   "source": [
    "Comparative Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d078856b",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Comparative Evaluation\n",
    "models = [\"Random Forest\", \"CNN+LSTM\"]\n",
    "mean_acc = [np.mean(rf_scores), np.mean(dl_scores)]\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "plt.bar(models, mean_acc, color=[\"tab:blue\",\"tab:green\"])\n",
    "plt.title(\"Cross-Validation Mean Accuracy Comparison\")\n",
    "plt.ylabel(\"Mean Accuracy\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21dbb70e",
   "metadata": {},
   "source": [
    "Step 15: EEG Analysis Module"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e272add",
   "metadata": {},
   "source": [
    "File Upload"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "efa163e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_eeg_file(fname):\n",
    "    if fname.endswith(\".edf\"):\n",
    "        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)\n",
    "        data, sf = raw.get_data(), int(raw.info['sfreq'])\n",
    "        ch = bandpass_filter(data[0], sf)  # first channel for demo\n",
    "    elif fname.endswith(\".mat\"):\n",
    "        eeg_data, fs, subject_id, markers, trials = load_mat_file(fname)\n",
    "        ch, sf = bandpass_filter(eeg_data[:,0], fs), fs\n",
    "    else:\n",
    "        raise ValueError(\"Unsupported file format\")\n",
    "    return ch, sf\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0cbef232",
   "metadata": {},
   "source": [
    "Cleaning & Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1909bd7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_eeg(ch, sf):\n",
    "    # Band-pass filter already applied in load_eeg_file\n",
    "    # Additional artifact removal (example: notch filter for 50Hz powerline)\n",
    "    notch_freq = 50\n",
    "    b, a = butter(2, [notch_freq-1, notch_freq+1], btype='bandstop', fs=sf)\n",
    "    ch_clean = filtfilt(b, a, ch)\n",
    "    return ch_clean\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20eff07a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def visualize_raw_vs_clean(ch, ch_clean, sf, seconds=5):\n",
    "    t = np.arange(seconds*sf)/sf\n",
    "    plt.figure(figsize=(12,5))\n",
    "    plt.plot(t, ch[:seconds*sf], label=\"Raw\", alpha=0.6)\n",
    "    plt.plot(t, ch_clean[:seconds*sf], label=\"Cleaned\", alpha=0.9)\n",
    "    plt.title(\"Raw vs Cleaned EEG Signal\")\n",
    "    plt.xlabel(\"Time (s)\")\n",
    "    plt.legend()\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89f7b03e",
   "metadata": {},
   "source": [
    "Feature Extraction & Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1e2a1b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def analyze_patient(ch, sf, model, encoder=None, window_sec=2):\n",
    "    epochs = segment_signal(ch, sf, window_sec=window_sec)\n",
    "    states = []\n",
    "    for i, ep in enumerate(epochs[:10]):  # analyze first 10 epochs\n",
    "        feats = extract_features(ep, sf)\n",
    "        X_input = np.array(list(feats.values())).reshape(1,-1)\n",
    "\n",
    "        # Prediction\n",
    "        if isinstance(model, RandomForestClassifier):\n",
    "            state = model.predict(X_input)[0]\n",
    "        else:  # CNN+LSTM\n",
    "            ep_dl = ep[:sf*2].reshape(1,-1,1)\n",
    "            state = model.predict(ep_dl).argmax(axis=1)[0]\n",
    "            if encoder: state = encoder.inverse_transform([state])[0]\n",
    "\n",
    "        states.append(state)\n",
    "        print(f\"Epoch {i}: State={state}, Action={adaptive_action(state)}\")\n",
    "    return states\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "98753a6d",
   "metadata": {},
   "source": [
    "Feedback Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec37aa70",
   "metadata": {},
   "outputs": [],
   "source": [
    "def visualize_patient_states(states):\n",
    "    plt.figure(figsize=(12,3))\n",
    "    plt.plot(states, marker=\"o\", linestyle=\"-\", color=\"tab:red\")\n",
    "    plt.title(\"Patient Cognitive States Across Epochs\")\n",
    "    plt.yticks([0,1,2,3,4], [\"Focused\",\"Distracted\",\"Fatigued\",\"Stressed\",\"Other\"])\n",
    "    plt.xlabel(\"Epoch Index\")\n",
    "    plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
