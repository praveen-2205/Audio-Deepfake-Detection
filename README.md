# 🎧 Deepfake Audio Detection
A Deep Learning Project

This repository contains code for detecting deepfake (synthetic) audio using two different approaches:

1. **Data Preprocessing & Spectrogram Generation** (Using CNN)
2. **Deepfake Audio Classification with Wav2Vec2**

---

## 👤 Project Structure

```
Deepfake-Audio-Detection/
│—— data/
│   ├—— ASVspoof5.dev.track_1.tsv  # Metadata for deepfake and real audio
│   ├—— flac_D/  # Original audio files
│   ├—— flac_D/bonafide/  # Real audio
│   ├—— flac_D/spoof/  # Deepfake audio
│   ├—— flac_D/bonafide_img/  # Spectrograms of real audio
│   └—— flac_D/spoof_img/  # Spectrograms of deepfake audio
│   └—— Fine Tuned.ipynb
│   └—— Spectrogram-Based CNN.ipynb
```

---

## 📝 Code 1: Data Preprocessing & Spectrogram Generation

### 📌 Objective  
- Reads metadata (`ASVspoof5.dev.track_1.tsv`) to identify **real vs. deepfake** audio.
- Moves the audio files into separate `bonafide/` and `spoof/` folders. (Everything is done in the code, just make sure to download the [dataset](https://zenodo.org/records/14498691))
- Generates **spectrogram images** for each `.flac` file using `librosa`.
- Uses **Convolutional Neural Networks (CNNs)** to classify spectrogram images and detect deepfake audio.

### 🛠️ Setup & Execution

```bash
pip install pandas librosa matplotlib numpy tensorflow
```

---

## 🤖 Code 2: Deepfake Audio Classification (Wav2Vec2)

### 📌 Objective  
- Loads real & deepfake audio into a **Hugging Face Dataset**.
- Uses **Facebook's Wav2Vec2** model for deepfake detection.
- Fine-tunes the model and evaluates it using accuracy, precision, and recall.

### 🛠️ Setup & Execution

```bash
pip install torch transformers datasets librosa scikit-learn
```

### 🏆 Inference

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from scripts.predict import predict_audio

model = Wav2Vec2ForSequenceClassification.from_pretrained("./models/wav2vec2-finetuned")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./models/wav2vec2-finetuned")

result = predict_audio("path/to/audio.wav", model, feature_extractor)
print(result)
```

---
## 🛠️ Technologies Used
- Python 🐍
- **Librosa** 🔊 (Audio Processing)
- **Transformers** 🤖 (Wav2Vec2)
- **Hugging Face Datasets** 🤗
- **PyTorch** 🔦
- **TensorFlow** 🧠
- **Matplotlib** 🎮 (Spectrogram Visualization)
---
## 💎 Contact
👨‍💻 **Praveen Kumar S**  
📧 Email: [spraveenkumar2205@gmail.com]  
📛 LinkedIn: [LinkedIn](https://www.linkedin.com/in/spraveenkumar2205)
