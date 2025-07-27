# 🛍️ ShopMe – AI-Powered Neuromarketing App

**ShopMe** is an AI-driven neuromarketing platform that analyzes real-time emotional and physiological responses to short-form video ads (e.g., TikTok-style reels) to help users discover what they truly want to buy.

By bringing neuromarketing out of the lab and into the wild, ShopMe serves as a mirror into unconscious consumer desires—powering both user insight and hyper-personalized content delivery.

---

## 📐 Project Structure

```

shopme/
├── frontend/              # Kotlin-based mobile app (TBD)
├── backend/
│   ├── app.py             # Flask API backend
│   ├── model/             # EfficientNet + Multi-task model logic
│   ├── data/
│   │   ├── annots\_arrs/   # CSV annotations + metadata
│   │   ├── npy/           # .npy arrays (face crops, valence, labels, etc.)
│   └── utils/             # Data loading, transforms, metrics
├── notebooks/             # Jupyter notebooks for exploration/training
├── train\_emotic.py        # Main training script
├── requirements.txt
└── README.md

````

---

## 🧠 Key Features

- **Real-time facial expression analysis**
- **Multi-label emotion classification** (26 emotion categories)
- **Valence–Arousal–Dominance regression**
- **EfficientNet backbone** optimized for mobile inference
- **Kotlin + Flask architecture** for cross-platform deployment

---

## 🗂️ Dataset

We use the **EMOTIC dataset**, which contains:
- Facial crops annotated with 26 discrete emotions (multi-label)
- Continuous valence, arousal, and dominance scores
- Preprocessed `.npy` arrays and metadata CSVs

> Note: Due to size, the dataset is tracked via Git LFS or must be downloaded separately. See `data/README.md` for setup.

---

## 🚀 Quickstart (Backend)

### 1. Set up environment
```bash
python3 -m venv .sm_venv
source .sm_venv/bin/activate
pip install -r requirements.txt
````

### 2. Train the model

```bash
python train_emotic.py
```

### 3. Run Flask server

```bash
export FLASK_APP=backend/app.py
flask run
```

---

## 🏋️‍♂️ Model Training

* Backbone: `EfficientNet-B0` (pretrained)
* Heads:

  * `Multi-label classification` for 26 emotions using `BCEWithLogitsLoss`
  * `Regression` for valence/arousal/dominance using `MSELoss`
* Outputs: Raw logits for emotions and continuous scores for VAD

---

## 📱 Frontend (Kotlin App)

The mobile app will:

* Display short-form product reels
* Capture user video via front-facing camera
* Stream facial data to backend for real-time emotional inference
* Deliver personalized product recommendations based on neurofeedback

---

## 🧪 Development Roadmap

* [x] EMOTIC-based training script
* [x] Multi-task EfficientNet model
* [ ] Real-time inference server
* [ ] Kotlin mobile integration
* [ ] Offline SDK mode
* [ ] Product recommendation engine

---

## 📄 License

This project is for academic research and prototyping. Licensing of third-party datasets (e.g., EMOTIC) is subject to their respective terms.

---

## 🙋‍♂️ Contributors

* **Josh Gompert** – Concept, Backend, AI/ML Modeling
* *(+ Seeking mobile dev collaborators!)*

---

## 📬 Contact

For questions, collaboration, or press: \[[jgomper1@jhu.edu](mailto:jgomper1@jhu.edu)]




