import os
import time
import asyncio
from asyncio import WindowsSelectorEventLoopPolicy

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import BartModel, BartTokenizer
import torchvision.models as models
import torchaudio
from PIL import Image

# ✅ GEMINI IMPORT
import google.generativeai as genai

# ------------------------------
# Configuration and Flask Setup
# ------------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

# ------------------------------
# ✅ GEMINI API CONFIGURATION
# ------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAh14DmapsHY0yDIBduvoNxSnroCAIU1eI"
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------------------
# 1. Define Audio Model
# ------------------------------
class DeepSpeech2AudioModel(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, conv_out_channels=32,
                 rnn_hidden_size=256, num_rnn_layers=3, bidirectional=True, output_dim=128):
        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_out_channels, 3, 2, 1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(),
            nn.Conv2d(conv_out_channels, conv_out_channels, 3, 2, 1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU()
        )

        self.rnn_input_size = conv_out_channels * (n_mels // 4)
        self.rnn = nn.GRU(self.rnn_input_size, rnn_hidden_size,
                          num_rnn_layers, batch_first=True, bidirectional=bidirectional)

        self.fc = nn.Linear(rnn_hidden_size * 2, output_dim)

    def forward(self, waveform):
        mel = self.melspec(waveform).unsqueeze(1)
        conv_out = self.conv(mel)

        b, c, f, t = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)
        conv_out = conv_out.reshape(b, t, -1)

        rnn_out, _ = self.rnn(conv_out)
        pooled = rnn_out.mean(dim=1)
        return self.fc(pooled)

# ------------------------------
# 2. Multimodal Classifier
# ------------------------------
class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, image_model, audio_model,
                 text_feat_dim, image_feat_dim, audio_feat_dim,
                 hidden_dim, num_classes):
        super().__init__()

        self.text_model = text_model
        self.image_model = image_model
        self.audio_model = audio_model

        self.text_fc = nn.Linear(text_feat_dim, hidden_dim)
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)
        self.audio_fc = nn.Linear(audio_feat_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_input=None, image_input=None, audio_input=None):
        features, count = None, 0

        if text_input:
            t_out = self.text_model(**text_input).last_hidden_state.mean(dim=1)
            t_feat = self.text_fc(t_out)
            features = t_feat if features is None else features + t_feat
            count += 1

        if image_input is not None:
            i_feat = self.image_fc(self.image_model(image_input))
            features = i_feat if features is None else features + i_feat
            count += 1

        if audio_input is not None:
            a_feat = self.audio_fc(self.audio_model(audio_input))
            features = a_feat if features is None else features + a_feat
            count += 1

        features = features / count
        return self.classifier(features)

# ------------------------------
# 3. Inference
# ------------------------------
def inference_all(model, tokenizer, text, image_path, audio_path, transform, device):
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.mean(dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(text_input=encoding, image_input=image, audio_input=waveform)

    return id2label[torch.argmax(logits, dim=1).item()]

# ------------------------------
# 4. Load Models
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
text_encoder = BartModel.from_pretrained("facebook/bart-base").to(device)

image_encoder = models.resnet18(pretrained=True)
image_feat_dim = image_encoder.fc.in_features
image_encoder.fc = nn.Identity()
image_encoder.to(device)

audio_encoder = DeepSpeech2AudioModel().to(device)

label_list = ["Alzheimer", "Cervical Cancer", "Covid", "Kidney Cancer",
              "Malaria", "Monkeypox", "Pneumonia", "Tuberculosis"]
id2label = dict(enumerate(label_list))

model = MultiModalClassifier(text_encoder, image_encoder, audio_encoder,
                              text_encoder.config.d_model, image_feat_dim, 128,
                              512, len(label_list)).to(device)

model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
model.eval()

# ------------------------------
# 5. Transforms
# ------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------------------
# ✅ GEMINI REPORT GENERATION
# ------------------------------
def generate_suggestion_report(disease):
    prompt = f"""
Provide a detailed step-by-step medical suggestion report for "{disease}" in:

1. What it is
2. Why it occurs
3. How to overcome (step-by-step)
4. Fertilizer / Treatment Recommendation

Give output in:
✅ English
✅ Tamil
"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini Error:", e)
        return f"Basic treatment required for {disease}. Consult doctor."

# ------------------------------
# 6. Flask Routes
# ------------------------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/download_report")
def download_report():
    return send_from_directory(app.config["UPLOAD_FOLDER"], "suggestion_report.txt", as_attachment=True)

@app.route("/result", methods=["POST"])
def result():
    text_input = request.form.get("text_input")
    image_file = request.files.get("image_file")
    audio_file = request.files.get("audio_file")

    if not text_input or not image_file or not audio_file:
        flash("All inputs required")
        return redirect(url_for("index"))

    img_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(image_file.filename))
    aud_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(audio_file.filename))

    image_file.save(img_path)
    audio_file.save(aud_path)

    prediction = inference_all(model, tokenizer, text_input, img_path, aud_path, image_transform, device)

    report = generate_suggestion_report(prediction)

    with open(os.path.join(app.config["UPLOAD_FOLDER"], "suggestion_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    return render_template("result.html",
                           predicted_disease=prediction,
                           suggestion_report=report,
                           audio_filename=os.path.basename(aud_path))

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
