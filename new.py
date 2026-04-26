import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchaudio
from transformers import BartTokenizer, BartModel
from PIL import Image
import numpy as np
import tempfile

# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = "multimodal_model.pth"
MODEL_NAME = "facebook/bart-base"
NUM_CLASSES = 3                     # CHANGE if you trained on more classes
LABEL_LIST = ["Normal", "Mild", "Severe"]   # MUST match training CSV order
TEXT_MAX_LEN = 128
HIDDEN_DIM = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# TOKENIZER
# ==========================================================
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

# ==========================================================
# AUDIO MODEL (EXACT SAME AS TRAINING)
# ==========================================================
class DeepSpeech2AudioModel(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128,
                 conv_out_channels=32, rnn_hidden_size=256,
                 num_rnn_layers=3, bidirectional=True, output_dim=128):
        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(),

            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU()
        )

        self.rnn_input_size = conv_out_channels * (n_mels // 4)

        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        rnn_out = rnn_hidden_size * 2
        self.fc = nn.Linear(rnn_out, output_dim)

    def forward(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.melspec(waveform)
        mel = self.db(mel)
        mel = mel.unsqueeze(1)

        conv_out = self.conv(mel)
        B, C, F, T = conv_out.size()

        conv_out = conv_out.permute(0, 3, 1, 2).contiguous()
        conv_out = conv_out.view(B, T, -1)

        rnn_out, _ = self.rnn(conv_out)
        pooled = rnn_out.mean(dim=1)

        return self.fc(pooled)

# ==========================================================
# MULTIMODAL MODEL (EXACT MATCH)
# ==========================================================
class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Text Encoder (BART)
        self.text_model = BartModel.from_pretrained(MODEL_NAME)
        text_feat_dim = self.text_model.config.d_model

        # Image Encoder (ResNet18)
        self.image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_img_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()

        # Audio Encoder
        self.audio_model = DeepSpeech2AudioModel()
        audio_feat_dim = 128

        # Projection Layers
        self.text_fc  = nn.Linear(text_feat_dim, HIDDEN_DIM)
        self.image_fc = nn.Linear(num_img_features, HIDDEN_DIM)
        self.audio_fc = nn.Linear(audio_feat_dim, HIDDEN_DIM)

        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, text_input=None, image_input=None, audio_input=None):
        features = 0.0
        modality_count = 0

        if text_input is not None:
            text_out = self.text_model(**text_input)
            pooled_text = text_out.last_hidden_state.mean(dim=1)
            text_f = self.text_fc(pooled_text)
            features = text_f
            modality_count += 1

        if image_input is not None:
            img_f = self.image_model(image_input)
            img_f = self.image_fc(img_f)
            features = img_f if modality_count == 0 else features + img_f
            modality_count += 1

        if audio_input is not None:
            aud_f = self.audio_model(audio_input)
            aud_f = self.audio_fc(aud_f)
            features = aud_f if modality_count == 0 else features + aud_f
            modality_count += 1

        if modality_count > 1:
            features = features / modality_count

        features = torch.nan_to_num(features)
        return self.classifier(features)

# ==========================================================
# LOAD MODEL (NO ERRORS)
# ==========================================================
@st.cache_resource
def load_model():
    model = MultiModalClassifier(NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

model = load_model()

# ==========================================================
# UI
# ==========================================================
st.set_page_config(page_title="Multimodal AI Diagnostic System", layout="wide")
st.title("🧠 Multimodal AI Diagnostic System")

st.sidebar.header("Upload Inputs")

text_input = st.sidebar.text_area("Enter Text")
image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
audio_file = st.sidebar.file_uploader("Upload Audio", type=["wav"])

# ==========================================================
# IMAGE PROCESSING
# ==========================================================
image_tensor = None
if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    image_tensor = img_tf(image).unsqueeze(0).to(DEVICE)

# ==========================================================
# AUDIO PROCESSING
# ==========================================================
audio_tensor = None
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    waveform = waveform.mean(dim=0)   # mono
    audio_tensor = waveform.unsqueeze(0).to(DEVICE)

# ==========================================================
# TEXT PROCESSING
# ==========================================================
text_tensor = None
if text_input.strip() != "":
    text_tensor = tokenizer(
        text_input,
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_tensors="pt"
    )
    text_tensor = {k: v.to(DEVICE) for k, v in text_tensor.items()}

# ==========================================================
# PREDICTION
# ==========================================================
if st.button("🔍 Predict"):
    if text_tensor is None and image_tensor is None and audio_tensor is None:
        st.warning("Please upload at least one modality.")
    else:
        with torch.no_grad():
            logits = model(
                text_input=text_tensor,
                image_input=image_tensor,
                audio_input=audio_tensor
            )

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))

        st.success(f"✅ Prediction: **{LABEL_LIST[pred_id]}**")

        st.subheader("Prediction Confidence")
        for i, lbl in enumerate(LABEL_LIST):
            st.write(f"{lbl}: {probs[i]:.4f}")
