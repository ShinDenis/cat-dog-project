import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from cnn import CNN

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Загружаем CSS
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Загружаем модель один раз
@st.cache_resource
def load_model():
    model = CNN()
    state_dict = torch.load("cat_dog_cnn_final_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Трансформации
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# UI
st.title("🐱Cat vs Dog Classifier🐶")

with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your file or browse", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение")

        st.markdown('</div>', unsafe_allow_html=True)

        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            prob_cat, prob_dog = probs[0][0].item(), probs[0][1].item()

        prediction = "Dog" if prob_dog > prob_cat else "Cat"
        probability = max(prob_cat, prob_dog)

        st.markdown(
            f'<div class="result-text">This is: {prediction}<br>Model confidence: {probability * 100:.2f} %</div>',
            unsafe_allow_html=True
        )