# 🧠 Skin Cancer Detection Web App using Deep Learning

This is a simple web application built with **Streamlit** that predicts skin cancer types from dermoscopic images using a **Convolutional Neural Network (CNN)** trained on the **HAM10000 dataset**.

---

## 📌 Project Features

- 🧪 Deep Learning model (Keras/TensorFlow)
- 🖼️ Image upload and classification
- 🧮 Trained on 10,000+ dermatoscopic images
- 🌐 Web app with Streamlit (runs locally)
- 🔐 Label encoding and preprocessing
- 💾 Deployable in a virtual environment

---

## 📂 Dataset: HAM10000

We use the **HAM10000 ("Human Against Machine with 10000 training images")** dataset available on Kaggle:

[Kaggle Dataset Link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

### 🔐 How to Download Using Kaggle API

1. **Sign in to Kaggle** and go to: [Kaggle API Token Page](https://www.kaggle.com/account)

2. Scroll to **API** section → Click on `Create New API Token`. This will download a file:  
   `kaggle.json`

3. Move `kaggle.json` to a safe location. For example:  
C:\Users\Vishal.kaggle\kaggle.json



4. In your Python project, configure environment:
```bash
mkdir ~/.kaggle
copy kaggle.json ~/.kaggle/
Install Kaggle package:


pip install kaggle
Download dataset:


kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip
⚙️ Setup Instructions
1. Clone the Repo

git clone https://github.com/your-username/skin-cancer-app.git
cd skin_cancer_app
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows
3. Install Dependencies

pip install -r requirements.txt
(If needed, install compatible versions of TensorFlow, Keras, Pandas, and Numpy manually.)

🤖 Model Training Steps
Load the HAM10000 metadata using Pandas

Map image file names to actual paths

Encode labels (e.g., mel, nv, bkl, etc.)

Preprocess: resize, normalize, augment (optional)

Define and compile a CNN model using Keras

Train the model and save it:

model.save("skin_cancer_model.h5")
🖥️ Run the Streamlit App
Make sure your model is saved as:
skin_cancer_model.h5

Run the app:


streamlit run app.py
Visit the app in your browser:
http://localhost:8501

📌 app.py Structure
python
Copy code
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("skin_cancer_model.h5")

# UI
st.title("🧠 Skin Cancer Detection")
uploaded_image = st.file_uploader("Upload a skin image...", type=["jpg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).resize((64, 64))
    st.image(image)
    prediction = model.predict(np.expand_dims(image, axis=0))
    st.write("Prediction:", np.argmax(prediction))
⚠️ Limitations
This model is not a medical diagnostic tool.

Only for educational purposes.

Limited number of skin cancer classes.

Accuracy depends on model depth and quality of training.

🚀 Future Work
Improve model accuracy using transfer learning (EfficientNet, ResNet)

Add Grad-CAM or heatmap visualization

Deploy using cloud (Streamlit Sharing, Hugging Face, Render)

Integrate patient report generation (ChatGPT + image model)

🧠 Author
Vishal Kumar
🚀 Building AI-powered apps with vision!

