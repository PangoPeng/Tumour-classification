import streamlit as st
from PIL import Image
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
import os
import base64

# Hide Streamlit's menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        margin-top: -90px;
    }
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to encode image to base64
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Add custom CSS for styling
custom_css = """
    <style>
    body {
        background-color: #f0f8ff;
        font-family: 'Arial', sans-serif;
    }
    .reportview-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    .title-logo {
        height: 40px;
        margin-right: 10px;
    }
    h1 {
        color: #0077be;
        font-weight: bold;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #0077be;
        color: white;
        font-weight: bold;
    }
    .upload-text {
        font-size: 18px;
        color: #333;
        margin-bottom: 10px;
    }
    .prediction {
        font-size: 20px;
        font-weight: bold;
        color: #0077be;
        margin-top: 20px;
    }
    .advice {
        font-style: italic;
        color: #555;
        margin-top: 10px;
    }
    .image-container {
        display: flex;
        justify-content: space-between;
    }
    .image-column {
        width: 48%;
    }
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

# Load and display logo with title
logo_path = 'logo.png'
if os.path.exists(logo_path):
    logo_base64 = img_to_base64(logo_path)
    title_html = f"""
    <div class="title-container">
        <img src="data:image/png;base64,{logo_base64}" class="title-logo">
        <h1>肿瘤图像的良恶性诊断</h1>
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)
else:
    st.warning("Logo file not found. Please check the file path.")
    st.markdown("<h1>肿瘤图像的良恶性诊断</h1>", unsafe_allow_html=True)

def main():
    st.markdown('<p class="upload-text">上传一张图像进行分析:</p>', unsafe_allow_html=True)
    file_uploaded = st.file_uploader('选择图像文件...', type=['png', 'jpg', 'jpeg'])

    if file_uploaded is not None:
        image = Image.open(file_uploaded).convert('RGB')

        result, heatmap = predict_class(image)

        col1, col2 = st.columns(2)

        with col1:
            st.write("上传的CT图像:")
            st.image(image, use_column_width=True)

        with col2:
            st.write("热力图可视化:")
            st.image(heatmap, use_column_width=True)

        st.markdown(f'<p class="prediction">诊断结果: {result}</p>', unsafe_allow_html=True)

        if result == '恶性':
            st.markdown('<p class="advice">请尽快就医咨询专业医生的意见。</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="advice">诊断结果为良性，但仍需咨询医生以获取进一步建议。</p>',
                        unsafe_allow_html=True)

def extract_vgg16_features(image):
    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
    vgg16.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        features = vgg16(input_batch)

    return features.numpy().reshape(1, -1), input_batch

def reduce_features(features):
    pca = joblib.load('pca_model.pkl')
    reduced_features = pca.transform(features)
    return reduced_features

def predict_class(image):
    model = joblib.load('xgb_best_model_10.pkl')
    features, input_batch = extract_vgg16_features(image)
    features_reduced = reduce_features(features)
    prediction = model.predict(features_reduced)[0]
    heatmap = generate_gradcam(input_batch, prediction, image)
    label = {0: '良性', 1: '恶性'}
    final = label[prediction]
    return final, heatmap

def generate_gradcam(input_batch, class_idx, original_image):
    model = models.vgg16(pretrained=True)
    model.eval()

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    def forward_hook(module, input, output):
        features.append(output)

    features = []
    grads = []

    model.features[-1].register_forward_hook(forward_hook)
    model.features[-1].register_backward_hook(backward_hook)

    output = model(input_batch)
    model.zero_grad()
    class_loss = output[0, class_idx]
    class_loss.backward()

    grad = grads[0].cpu().data.numpy()[0]
    fmap = features[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    grad_cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        grad_cam += w * fmap[i]

    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = cv2.resize(grad_cam, (224, 224))
    grad_cam = grad_cam - np.min(grad_cam)
    grad_cam = grad_cam / np.max(grad_cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original_image = original_image.resize((224, 224))
    original_image = np.array(original_image)
    combined_image = np.float32(original_image) / 255

    mask = grad_cam > 0.5
    combined_image[mask] = combined_image[mask] * 0.5 + heatmap[mask] * 0.5

    overlay = combined_image * 0.6 + heatmap * 0.4

    return np.uint8(255 * overlay)

footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0077be;
    color: white;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
}
.footer a {
    color: #f0f8ff;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
    <p>Developed with ❤ by Xiayan Peng <a href="https://github.com/pangopeng" target="_blank"></a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
