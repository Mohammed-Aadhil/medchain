import os
import time
import asyncio
import sqlite3
from asyncio import WindowsSelectorEventLoopPolicy
from functools import wraps
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
    session,
    jsonify,
)
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import BartModel, BartTokenizer
import torchvision.models as models
import torchaudio
from PIL import Image
import scipy.io.wavfile as wavfile

# ✅ GEMINI IMPORT
import google.generativeai as genai

# ------------------------------
# Configuration and Flask Setup
# ------------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"  # 🔐 change in production

BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DB_PATH = os.path.join(BASE_DIR, "app.db")

asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

# ------------------------------
# ✅ GEMINI API CONFIGURATION
# ------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # ✅ set in .env, do NOT hardcode
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not set. Gemini features may fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ------------------------------
# 🔹 SQLite Helper
# ------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # Users table (patients)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        );
        """
    )

    # Queries table (user → doctor)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            answered_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )

    conn.commit()
    conn.close()


# Initialize DB at startup
init_db()

# Default doctor credentials
DOCTOR_EMAIL = "doctor@gmail.com"
DOCTOR_PASSWORD = "doctor123"

# ------------------------------
# 🔹 Login Required Decorators
# ------------------------------
def user_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login as user first.", "warning")
            return redirect(url_for("user_login"))
        return f(*args, **kwargs)

    return decorated


def doctor_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_doctor"):
            flash("Please login as doctor first.", "warning")
            return redirect(url_for("doctor_login"))
        return f(*args, **kwargs)

    return decorated


# ------------------------------
# 1. Define Audio Model
# ------------------------------
class DeepSpeech2AudioModel(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mels=128,
        conv_out_channels=32,
        rnn_hidden_size=256,
        num_rnn_layers=3,
        bidirectional=True,
        output_dim=128,
    ):
        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_out_channels, 3, 2, 1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(),
            nn.Conv2d(conv_out_channels, conv_out_channels, 3, 2, 1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(),
        )

        self.rnn_input_size = conv_out_channels * (n_mels // 4)
        self.rnn = nn.GRU(
            self.rnn_input_size,
            rnn_hidden_size,
            num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

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
    def __init__(
        self,
        text_model,
        image_model,
        audio_model,
        text_feat_dim,
        image_feat_dim,
        audio_feat_dim,
        hidden_dim,
        num_classes,
    ):
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
    encoding = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    try:
        waveform, sr = torchaudio.load(audio_path)
    except ImportError:
        # Fallback if torchcodec is not installed
        try:
            import scipy.io.wavfile as wavfile
            sr, waveform_np = wavfile.read(audio_path)
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
        except Exception as e:
            print(f"Error loading audio with scipy: {e}")
            # Create a dummy waveform if all else fails
            waveform = torch.randn(1, 16000)
            sr = 16000
    
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

label_list = [
    "Alzheimer",
    "Cervical Cancer",
    "Covid",
    "Kidney Cancer",
    "Malaria",
    "Monkeypox",
    "Pneumonia",
    "Tuberculosis",
]
id2label = dict(enumerate(label_list))

model = MultiModalClassifier(
    text_encoder,
    image_encoder,
    audio_encoder,
    text_encoder.config.d_model,
    image_feat_dim,
    128,
    512,
    len(label_list),
).to(device)

model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
model.eval()

# ------------------------------
# 5. Transforms
# ------------------------------
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# ------------------------------
# ✅ GEMINI REPORT GENERATION
# ------------------------------
def generate_suggestion_report(disease):
    prompt = f"""
Provide a detailed step-by-step medical suggestion report for "{disease}" in:

1. What it is
2. Why it occurs
3. How to overcome (step-by-step)
4. Treatment Recommendation

Give output in:
✅ English
✅ Tamil

Also remind user to consult a real doctor.
"""

    try:
        if not GEMINI_API_KEY:
            return f"Predicted disease: {disease}. Please consult a doctor. (Gemini key not configured)"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini Error:", e)
        return f"Basic treatment required for {disease}. Consult doctor."


# ------------------------------
# ✅ GEMINI CHATBOT HELPER
# ------------------------------
def generate_chatbot_reply(message):
    prompt = f"""
You are a helpful medical assistant. Answer in simple language.
User message: {message}

Rules:
- Give general health education.
- Do NOT give final diagnosis.
- Always tell them to consult a doctor for confirmation.
- Reply in short paragraphs.
"""

    try:
        if not GEMINI_API_KEY:
            return "AI chatbot is not configured (missing GEMINI_API_KEY). Please contact admin."
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini Chatbot Error:", e)
        return "Sorry, I couldn't process your request. Please try again later."


# ------------------------------
# 6. Auth & Landing Routes
# ------------------------------
@app.route("/")
def landing():
    # Simple landing page with options: User / Doctor
    return render_template("landing.html")


@app.route("/forgot-password")
def forgot_password_redirect():
    return redirect(url_for("forgot_password"))


# ---------- User Register ----------
@app.route("/user/register", methods=["GET", "POST"])
def user_register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        if not name or not email or not password:
            flash("All fields are required", "danger")
            return redirect(url_for("user_register"))

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password),  # ⚠️ plain text for demo only
            )
            conn.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("user_login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")
            return redirect(url_for("user_register"))
        finally:
            conn.close()

    return render_template("auth/user_register.html")


# ---------- User Login ----------
@app.route("/user/login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE email = ? AND password = ?",
            (email, password),
        )
        user = cur.fetchone()
        conn.close()

        if user:
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("user_dashboard"))
        else:
            flash("Invalid credentials.", "danger")
            return redirect(url_for("user_login"))

    return render_template("auth/user_login.html")

# ---------- Forget Password ----------
@app.route("/user/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        new_password = request.form.get("new_password")

        if not email or not new_password:
            flash("All fields are required.", "danger")
            return redirect(url_for("forgot_password"))

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cur.fetchone()

        if not user:
            conn.close()
            flash("No account found with this email.", "danger")
            return redirect(url_for("forgot_password"))

        # ✅ Update new password directly
        cur.execute(
            "UPDATE users SET password = ? WHERE email = ?",
            (new_password, email)
        )
        conn.commit()
        conn.close()

        flash("Password reset successful. Please login.", "success")
        return redirect(url_for("user_login"))

    return render_template("auth/forgot_password.html")

# ---------- Doctor Login ----------
@app.route("/doctor/login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email == DOCTOR_EMAIL and password == DOCTOR_PASSWORD:
            session.clear()
            session["is_doctor"] = True
            session["doctor_email"] = email
            flash("Doctor login successful.", "success")
            return redirect(url_for("doctor_dashboard"))
        else:
            flash("Invalid doctor credentials.", "danger")
            return redirect(url_for("doctor_login"))

    return render_template("auth/doctor_login.html")


# ---------- Logout ----------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("landing"))


# ------------------------------
# 7. User Dashboard & Features
# ------------------------------
@app.route("/user/dashboard")
@user_login_required
def user_dashboard():
    return render_template("user/user_dashboard.html", user_name=session.get("user_name"))


# ---------- Prediction Page (uses your existing index.html) ----------
@app.route("/index")
@user_login_required
def index():
    # index.html = multimodal input page (symptoms, image, audio)
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download_report")
def download_report():
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], "suggestion_report.txt", as_attachment=True
    )


# ---------- Result Route (Prediction + Report) ----------
@app.route("/result", methods=["POST"])
@user_login_required
def result():
    text_input = request.form.get("text_input")
    image_file = request.files.get("image_file")
    audio_file = request.files.get("audio_file")

    if not text_input or not image_file or not audio_file:
        flash("All inputs required", "danger")
        return redirect(url_for("index"))

    img_path = os.path.join(
        app.config["UPLOAD_FOLDER"], secure_filename(image_file.filename)
    )
    aud_path = os.path.join(
        app.config["UPLOAD_FOLDER"], secure_filename(audio_file.filename)
    )

    image_file.save(img_path)
    audio_file.save(aud_path)

    prediction = inference_all(
        model, tokenizer, text_input, img_path, aud_path, image_transform, device
    )

    report = generate_suggestion_report(prediction)

    with open(
        os.path.join(app.config["UPLOAD_FOLDER"], "suggestion_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    return render_template(
        "result.html",
        predicted_disease=prediction,
        suggestion_report=report,
        audio_filename=os.path.basename(aud_path),
    )


# ---------- User → Doctor Query ----------
@app.route("/user/query", methods=["GET", "POST"])
@user_login_required
def user_query():
    if request.method == "POST":
        question = request.form.get("question")
        if not question:
            flash("Question cannot be empty.", "danger")
            return redirect(url_for("user_query"))

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO queries (user_id, question, status, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                session["user_id"],
                question,
                "pending",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        conn.close()

        flash("Your query has been sent to the doctor.", "success")
        return redirect(url_for("user_dashboard"))

    return render_template("user/ask_doctor.html")


# ---------- View Doctor Solutions ----------
@app.route("/user/solutions")
@user_login_required
def user_solutions():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT q.id, q.question, q.answer, q.status, q.created_at, q.answered_at
        FROM queries q
        WHERE q.user_id = ?
        ORDER BY q.created_at DESC
        """,
        (session["user_id"],),
    )
    queries = cur.fetchall()
    conn.close()

    return render_template("user/view_solution.html", queries=queries)


# ---------- AI Chatbot ----------
@app.route("/user/chatbot", methods=["GET"])
@user_login_required
def chatbot_page():
    return render_template("user/chatbot.html")


@app.route("/user/chatbot/ask", methods=["POST"])
@user_login_required
def chatbot_ask():
    data = request.get_json() or {}
    message = data.get("message") or request.form.get("message")

    if not message:
        return jsonify({"reply": "Please type a message."})

    reply = generate_chatbot_reply(message)
    return jsonify({"reply": reply})


# ------------------------------
# 8. Doctor Dashboard & Features
# ------------------------------
@app.route("/doctor/dashboard")
@doctor_login_required
def doctor_dashboard():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT q.id, q.question, q.answer, q.status, q.created_at, q.answered_at,
               u.name as user_name, u.email as user_email
        FROM queries q
        JOIN users u ON q.user_id = u.id
        ORDER BY q.created_at DESC
        """
    )
    queries = cur.fetchall()
    conn.close()

    return render_template("doctor/doctor_dashboard.html", queries=queries)


# ---------- Doctor Answer a Query ----------
@app.route("/doctor/reply/<int:query_id>", methods=["GET", "POST"])
@doctor_login_required
def doctor_reply(query_id):
    conn = get_db_connection()
    cur = conn.cursor()

    if request.method == "POST":
        answer = request.form.get("answer")
        if not answer:
            flash("Answer cannot be empty.", "danger")
            return redirect(url_for("doctor_reply", query_id=query_id))

        cur.execute(
            """
            UPDATE queries
            SET answer = ?, status = ?, answered_at = ?
            WHERE id = ?
            """,
            (answer, "answered", datetime.now().isoformat(timespec="seconds"), query_id),
        )
        conn.commit()
        conn.close()

        flash("Answer submitted successfully.", "success")
        return redirect(url_for("doctor_dashboard"))

    # GET - show question details
    cur.execute(
        """
        SELECT q.id, q.question, q.answer, q.status, q.created_at,
               u.name as user_name, u.email as user_email
        FROM queries q
        JOIN users u ON q.user_id = u.id
        WHERE q.id = ?
        """,
        (query_id,),
    )
    query = cur.fetchone()
    conn.close()

    if not query:
        flash("Query not found.", "danger")
        return redirect(url_for("doctor_dashboard"))

    return render_template("doctor/answer_query.html", query=query)


# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    # debug=True only for development
    app.run(debug=True)
