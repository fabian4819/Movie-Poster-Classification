# 🎬 Movie Poster Text Attention Classification

## 📌 Overview

This project implements a machine learning system for **classifying the attention level of text** in movie posters on a scale from **0 (Very Unattractive)** to **4 (Very Attractive)**. The focus is on **visual typography** analysis rather than semantic meaning.

---

## 🚀 Features

- 🧠 Text region detection using OCR (Tesseract)
- 🧪 Visual feature extraction (54 features)
- 🎯 Feature selection with Random Forest
- ⚖️ Class balancing using SMOTE
- 🌲 Random Forest classification
- 📊 Visualizations for model interpretability

---

## 🛠 Requirements

- Python 3.7+
- OpenCV
- NumPy, Pandas
- Matplotlib, Scikit-learn
- Imbalanced-learn
- Scikit-image
- Pytesseract (Tesseract OCR installed)

---

## 🧩 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/movie-poster-attention.git
cd movie-poster-attention

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 🗂 Dataset

The dataset consists of **300 movie poster images** labeled on a 5-point Likert scale:

| Score | Description         | Count | Percentage |
|-------|----------------------|-------|------------|
| 0     | Very Unattractive    | 29    | 9.7%       |
| 1     | Unattractive         | 56    | 18.7%      |
| 2     | Neutral              | 55    | 18.3%      |
| 3     | Attractive           | 47    | 15.7%      |
| 4     | Very Attractive      | 113   | 37.7%      |

---

## ⚙️ Usage

### 📁 Data Preparation

1. Place movie poster images in the `datasets/` directory  
2. Create `attention.csv` with columns: `image_name`, `attention_score`

### 🧪 Running the Pipeline

```python
from main import run_project

data_dir = './datasets'
csv_path = './attention.csv'

classifier = run_project(data_dir, csv_path)
```

### 🔍 Predicting New Images

```python
import cv2
import pickle
import pandas as pd
from feature_extraction import extract_features_from_image

# Load model
with open('output/attention_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and process image
img_path = 'path/to/poster.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Extract features
features = extract_features_from_image(img_rgb)
features_df = pd.DataFrame([features])[model_features]

# Predict
prediction = model.predict(features_df)[0]
print(f"Predicted attention score: {prediction}")
```

---

## 🧱 Key Components

### 1. Feature Extraction

- Image properties (dimensions, aspect ratio)
- Color features (RGB, HSV)
- OCR-based text analysis
- Texture: Local Binary Patterns (LBP)
- Edges: Canny
- Shapes: HOG (Histogram of Oriented Gradients)

### 2. Feature Selection

- Random Forest importance
- Key features: aspect ratio, text area, positioning

### 3. Class Balancing

- SMOTE for minority class oversampling

### 4. Model Training

- StandardScaler for normalization
- RandomForestClassifier (`n_estimators=100`)

### 5. Evaluation & Visualization

- Confusion matrix
- Classification report
- Feature importance charts
- Visual explanation of predictions

---

## 📈 Results

- **Accuracy**: 0.28 (baseline: 0.20)
- **F1-Score (weighted)**: 0.29

**Insights**:
1. Text size relative to poster is crucial
2. Text positioning affects attention
3. Background contrast matters
4. Larger text coverage correlates with higher scores

---

## 📂 Project Structure

```
movie-poster-attention/
├── datasets/                # Movie posters
├── output/                  # Model outputs
│   ├── visualizations/
│   ├── feature_importances.png
│   ├── confusion_matrix.png
│   └── attention_classifier_model.pkl
├── feature_extraction.py    # Feature extraction logic
├── main.py                  # Main pipeline
├── attention.csv            # Labels
├── requirements.txt
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- Developed for the *Image Processing and Computer Vision* course @ Universitas Gadjah Mada
- OCR powered by **Tesseract**

---

## ⚙️ Implementation Details

### 🔬 Feature Extraction Pipeline

1. **Image Preprocessing**  
   - Resize image  
   - Convert color space  
   - Enhance contrast

2. **OCR Configuration**
   ```python
   config = '--psm 11 --oem 3'
   ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
   ```

3. **Typography Features**  
   - Text block count  
   - Area ratio  
   - Positioning  
   - Height relative to image

4. **Visual Features**  
   - LBP (texture)  
   - HOG (shape)  
   - Canny (edges)  
   - Color histogram

### 🏋️ Model Training Pipeline

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)

# Train
pipeline.fit(X_resampled, y_resampled)
```

---

## 📊 Visualization Tools

1. **Attention Heatmaps**
   - Highlight text regions
   - Overlay edge detection

2. **Feature Importance**
   - Bar plots
   - Correlation heatmaps

3. **Prediction Analysis**
   - True vs predicted comparison
   - Confidence score plots

---

## 🧩 Challenges & Solutions

### 🧾 OCR Limitations
- Stylized fonts reduce accuracy  
✅ Applied preprocessing and fallback features

### ⚖️ Class Imbalance
- High concentration in score 4  
✅ Used SMOTE and stratified sampling

### ⚙️ Feature Overload
- 54 features can lead to overfitting  
✅ Feature selection and validation used

---

## 🌱 Future Work

1. **Deep Learning**
   - CNN or transfer learning (e.g. ResNet, EfficientNet)

2. **Extended Features**
   - Font type, semantic meaning, color harmony

3. **User Validation**
   - Eye-tracking studies  
   - Multi-annotator labeling

4. **Real-World Integration**
   - Design plugin  
   - REST API  
   - Typography recommendation engine
