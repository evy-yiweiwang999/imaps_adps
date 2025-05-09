# app.py
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
import joblib
import torch
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights
import torch.nn as nn
import numpy as np
import random

# === Set seed for reproducibility ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'png', 'jpg', 'jpeg'}

DIAGNOSIS_MARKERS = ["CXCL9","IL4","IL17F","CCL8","IL17A","CSF2","VEGFA","IL13","CCL11","CCL19"] 
MARKERS_AD = ["CXCL11", "IL4", "CXCL10", "IL17F", "IL27", "CSF3", "IL13", "IL7", "CCL19", "IL33"]
MARKERS_PS = ["IL17A", "IL23", "TNF", "IL1B", "CXCL8", "IL36G", "CCL20", "IL6", "IFNG", "IL22"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load protein models
rf_diagnosis = joblib.load('models/model_dia_rf.pkl')
rf_ad = joblib.load('models/model_ad_rf.pkl')
rf_ps = joblib.load('models/model_ps_rf.pkl')

def extract_marker_features(protein_file, markers):
    try:
        df = pd.read_csv(protein_file)
    except Exception:
        return {}
    df.columns = [col.strip().upper() for col in df.columns]
    features = {}
    for marker in markers:
        if marker.upper() in df.columns:
            try:
                features[marker.upper()] = float(df[marker.upper()].iloc[0])
            except:
                continue
    return features

def predict_protein(csv_path, task, iga=None, bsa=None):
    import pandas as pd

    # 读取并统一列名
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().upper() for col in df.columns]

    # 根据任务选择模型与特征
    if task == 'diagnosis':
        features = extract_marker_features(csv_path, DIAGNOSIS_MARKERS)
        model = rf_diagnosis
        label_map = ["Atopic Dermatitis (AD)", "Psoriasis (PS)"]
    elif task == 'adtreat':
        features = extract_marker_features(csv_path, MARKERS_AD)
        model = rf_ad
        label_map = ["IL4_IL13i_R", "IL4_IL13i_NR", "JAKi_R"]
        if iga:
            features["IGA"] = float(iga)
        if bsa:
            features["BSA"] = float(bsa)
    elif task == 'pstreat':
        features = extract_marker_features(csv_path, MARKERS_PS)
        model = rf_ps
        label_map = ["IL23i_R", "PDE4i_R", "IL12_IL23i_R", "IL17i_R"]
    else:
        print("[ERROR] Unknown task.")
        return {}

    # 获取模型要求的特征顺序
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(features.keys())

    # 构建带列名的 DataFrame 输入（推荐方式）
    input_vec = [features.get(f, 0) for f in model_features]
    input_df = pd.DataFrame([input_vec], columns=model_features)

    # === Debug 输出 ===
    print("======= [DEBUG] Predict Protein =======")
    print(f"Task: {task}")
    print(f"Extracted Features: {features}")
    print(f"Model Expected Features: {model_features}")
    print(f"Input DataFrame:\n{input_df}")
    print("=======================================")

    try:
        proba = model.predict_proba(input_df)[0]
        return {label: round(float(p), 3) for label, p in zip(label_map, proba)}
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return {}


def load_efficientnet_model(path):
    state_dict = torch.load(path, map_location='cpu')
    try:
        out_features = state_dict['classifier.3.weight'].shape[0]
    except KeyError:
        out_features = 2
    base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    base.classifier = nn.Identity()
    model = nn.Sequential(
        base,
        nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_features)
        )
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

cnn_diagnosis = load_efficientnet_model('models/eff_dia_full_state_dict.pt')
cnn_ad = load_efficientnet_model('models/effnetb0_best_adtr_state_dict.pt')
cnn_ps = load_efficientnet_model('models/effnetb0_best_pstr_state_dict.pt')

def predict_image(img_path, task):
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        if task == 'diagnosis':
            output = cnn_diagnosis(input_tensor)
        elif task == 'adtreat':
            output = cnn_ad(input_tensor)
        elif task == 'pstreat':
            output = cnn_ps(input_tensor)
        else:
            return {}
        proba = torch.softmax(output, dim=1).numpy()[0]

    if task == 'diagnosis' and len(proba) == 2:
        return {
            "Atopic Dermatitis (AD)": round(float(proba[0]), 3),
            "Psoriasis (PS)": round(float(proba[1]), 3)
        }
    elif task == 'adtreat':
        labels = ["IL4_IL13i_R", "IL4_IL13i_NR", "JAKi_R"]
    elif task == 'pstreat':
        labels = ["IL23i_R", "PDE4i_R", "IL12_IL23i_R", "IL17i_R"]
    else:
        labels = [f"Class {i}" for i in range(len(proba))]

    return {label: round(float(p), 3) for label, p in zip(labels, proba)}
@app.route('/submit', methods=['POST'])
def submit():
    modality = request.form.get('modality')
    task = request.form.get('task')

    protein_file = request.files.get('protein_file')
    image_file = request.files.get('image_file')
    iga = request.form.get('iga')
    bsa = request.form.get('bsa')

    image_filename = None
    image_diag_result = {}
    image_treat_result = {}
    protein_diag_result = {}
    protein_treat_result = {}

    if modality in ['image', 'multimodal']:
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_filename = filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

            image_diag_result = predict_image(image_path, 'diagnosis')
            ad_score = image_diag_result.get("Atopic Dermatitis (AD)", 0)
            ps_score = image_diag_result.get("Psoriasis (PS)", 0)

            if ad_score > ps_score:
                image_treat_result = predict_image(image_path, 'adtreat')
            else:
                image_treat_result = predict_image(image_path, 'pstreat')

    if modality in ['protein', 'multimodal']:
        if protein_file and allowed_file(protein_file.filename):
            filename = secure_filename(protein_file.filename)
            protein_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            protein_file.save(protein_path)

            protein_diag_result = predict_protein(protein_path, 'diagnosis')
            ad_score = protein_diag_result.get("Atopic Dermatitis (AD)", 0)
            ps_score = protein_diag_result.get("Psoriasis (PS)", 0)

            if ad_score > ps_score:
                protein_treat_result = predict_protein(protein_path, 'adtreat', iga, bsa)
            else:
                protein_treat_result = predict_protein(protein_path, 'pstreat')

    return render_template(
        'result.html',
        image_filename=image_filename,
        image_diag=image_diag_result,
        image_treat=image_treat_result,
        protein_diag=protein_diag_result,
        protein_treat=protein_treat_result,
        recommendation="* We recommend using the Olink (protein-based) result; the image-based result is for reference only."
    )

@app.route('/')
def index():
    return render_template('index.html')

    return render_template('result.html', prediction=prediction_result + additional_result, image_filename=image_filename)

@app.route('/demo')
def run_demo():
    protein_path = os.path.join('static', 'demo', 'demo_protein.csv')
    image_path = os.path.join('static', 'demo', 'demo_image.jpg')

    image_diag_result = predict_image(image_path, 'diagnosis')
    ad_score_img = image_diag_result.get("Atopic Dermatitis (AD)", 0)
    ps_score_img = image_diag_result.get("Psoriasis (PS)", 0)
    image_treat_result = predict_image(image_path, 'adtreat') if ad_score_img > ps_score_img else predict_image(image_path, 'pstreat')

    protein_diag_result = predict_protein(protein_path, 'diagnosis')
    ad_score_pro = protein_diag_result.get("Atopic Dermatitis (AD)", 0)
    ps_score_pro = protein_diag_result.get("Psoriasis (PS)", 0)
    protein_treat_result = predict_protein(protein_path, 'adtreat') if ad_score_pro > ps_score_pro else predict_protein(protein_path, 'pstreat')

    return render_template(
        'result.html',
        image_filename='demo/demo_image.jpg',
        image_diag=image_diag_result,
        image_treat=image_treat_result,
        protein_diag=protein_diag_result,
        protein_treat=protein_treat_result,
        recommendation="* We recommend using the Olink (protein-based) result; the image-based result is for reference only."
    )

if __name__ == '__main__':
    app.run(debug=True)
