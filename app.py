"""
RetinaScan AI — Flask SPA
FIXED VERSION: Added correct preprocessing pipeline matching training
Run:  python app.py
"""
import os
import io
import base64
import json
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template_string
from PIL import Image
from torchvision import transforms
import torch

from model import build_model
from utils import load_checkpoint

# ── Config ───────────────────────────────────────────────────────────
MODEL_PATH = "saved_model/best_model.pth"
GOOGLE_DRIVE_FILE_ID = "11vq3--8cdxDyc2rJBOctQpRF6-q7EpAO"

PORT = int(os.environ.get("PORT", 8501))

CLASS_NAMES  = ['0_No_DR', '1_Mild', '2_Moderate', '3_Severe', '4_PDR']
CLASS_LABELS = {
    '0_No_DR'   : 'No DR Detected',
    '1_Mild'    : 'Mild NPDR',
    '2_Moderate': 'Moderate NPDR',
    '3_Severe'  : 'Severe NPDR',
    '4_PDR'     : 'Proliferative DR',
}
CLASS_COLORS = {
    '0_No_DR'   : '#22c55e',
    '1_Mild'    : '#eab308',
    '2_Moderate': '#f97316',
    '3_Severe'  : '#ef4444',
    '4_PDR'     : '#a855f7',
}
CLASS_DESC = {
    '0_No_DR'   : 'No signs detected. Routine annual screening recommended.',
    '1_Mild'    : 'Microaneurysms present. Annual follow-up advised.',
    '2_Moderate': 'Bleeds and leakage visible. Closer monitoring needed.',
    '3_Severe'  : 'Extensive vessel damage. Urgent specialist referral.',
    '4_PDR'     : 'New vessel growth detected. Immediate referral required.',
}
CLASS_EMOJI  = {'0_No_DR':'✅','1_Mild':'🟡','2_Moderate':'🟠','3_Severe':'🔴','4_PDR':'🚨'}
ALERTS = {
    '4_PDR'     : ('#a855f7', '🚨', 'URGENT — Immediate ophthalmologist referral required.'),
    '3_Severe'  : ('#ef4444', '🔴', 'HIGH RISK — Urgent specialist referral needed.'),
    '2_Moderate': ('#f97316', '⚠️', 'FOLLOW-UP — Schedule specialist appointment soon.'),
    '1_Mild'    : ('#0ea5e9', 'ℹ️', 'MONITOR — Annual eye exam recommended.'),
    '0_No_DR'   : ('#22c55e', '✅', 'CLEAR — Continue routine annual screening.'),
}



# ── Inference transform (applied AFTER preprocessing) ─────────────────
# ══════════════════════════════════════════════════════════════════════
# PREPROCESSING — matches preprocess_and_balance.py exactly
# ══════════════════════════════════════════════════════════════════════

def remove_black_border(img_pil):
    """Remove dark circular border. Falls back to original on any error."""
    try:
        arr  = np.array(img_pil)
        h0, w0 = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return img_pil
        x, y, w, h = cv2.boundingRect(coords)
        if w < 50 or h < 50:
            return img_pil
        if (h * w) < (h0 * w0 * 0.6):
            return img_pil
        return Image.fromarray(arr[y:y+h, x:x+w])
    except Exception:
        return img_pil

def apply_clahe(img_pil):
    """CLAHE on LAB L-channel. clipLimit=2.0, tileGrid=8x8."""
    try:
        arr     = np.array(img_pil)
        lab     = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        arr     = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        return Image.fromarray(arr)
    except Exception:
        return img_pil

def ben_graham(img_pil, size=224):
    """Ben Graham illumination normalisation. result = 4*img - 4*blur + 128."""
    try:
        arr     = np.array(img_pil.resize((size, size), Image.BILINEAR))
        arr     = arr.astype(np.float32)
        sigma   = max(5, size // 30)
        blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
        result  = np.clip(arr * 4 - blurred * 4 + 128, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    except Exception:
        try:
            return img_pil.resize((size, size), Image.BILINEAR)
        except Exception:
            return img_pil

def preprocess_fundus(img_pil):
    """
    Full preprocessing pipeline:
      1. Convert to RGB
      2. Black border removal
      3. CLAHE contrast enhancement
      4. Ben Graham illumination normalisation
    Matches training pipeline exactly — required for correct predictions.
    """
    img = img_pil.convert('RGB')
    img = remove_black_border(img)
    img = apply_clahe(img)
    img = ben_graham(img, size=224)
    return img

# ── Inference transforms + TTA ─────────────────────────────────────────
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def to_tensor_norm(img):
    return NORMALIZE(transforms.ToTensor()(img))

def get_tta_tensors(processed_img):
    """
    4-variant TTA: original + H-flip + V-flip + H+V flip.
    Averaging these 4 predictions gives +1-2% F1 improvement.
    Kept to 4 variants for speed (vs 8 which is slower).
    """
    img = processed_img.resize((224, 224), Image.BILINEAR)
    variants = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
    ]
    return torch.stack([to_tensor_norm(v) for v in variants])  # [4, 3, 224, 224]

# ── Model ─────────────────────────────────────────────────────────────
app    = Flask(__name__)
_model = None

def download_model_if_missing():
    if os.path.exists(MODEL_PATH):
        return

    import gdown

    os.makedirs("saved_model", exist_ok=True)

    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model download failed.")

def get_model():
    global _model

    if _model is None:
        download_model_if_missing()

        m = build_model(pretrained=False)
        m = load_checkpoint(m, MODEL_PATH)
        m.eval()
        _model = m

    return _model

def predict_pil(image: Image.Image):
    """
    Full inference pipeline:
      1. Preprocess (border removal + CLAHE + Ben Graham)
      2. 4-variant TTA (original + 3 flips)
      3. Average probabilities across all 4 variants
      4. Return argmax prediction
    """
    m = get_model()

    # Step 1: Preprocess — matches training pipeline exactly
    processed = preprocess_fundus(image.convert('RGB'))

    # Step 2: Generate 4 TTA tensors
    tta_tensors = get_tta_tensors(processed)          # [4, 3, 224, 224]

    # Step 3: Run all 4 variants through model in one forward pass
    with torch.no_grad():
        probs = torch.softmax(m(tta_tensors), dim=1)  # [4, 5]

    # Step 4: Average probabilities across all 4 TTA variants
    avg_probs = probs.mean(dim=0).numpy()             # [5]

    pred = int(np.argmax(avg_probs))
    return CLASS_NAMES[pred], avg_probs.tolist()

def img_to_b64(image: Image.Image, max_size=600) -> str:
    thumb = image.copy()
    thumb.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    thumb.save(buf, format='WEBP', quality=85)
    return base64.b64encode(buf.getvalue()).decode()

# ── API routes ────────────────────────────────────────────────────────
@app.route('/api/status')
def api_status():
    try:
        download_model_if_missing()
        return jsonify({'model_ready': os.path.exists(MODEL_PATH)})
    except Exception as e:
        return jsonify({'model_ready': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not found — run python train.py first'}), 503
    f     = request.files['file']
    image = Image.open(f.stream).convert('RGB')
    pred_class, probs = predict_pil(image)
    return jsonify({
        'predicted_class': pred_class,
        'label'          : CLASS_LABELS[pred_class],
        'color'          : CLASS_COLORS[pred_class],
        'desc'           : CLASS_DESC[pred_class],
        'emoji'          : CLASS_EMOJI[pred_class],
        'alert_color'    : ALERTS[pred_class][0],
        'alert_icon'     : ALERTS[pred_class][1],
        'alert_msg'      : ALERTS[pred_class][2],
        'confidence'     : float(max(probs)) * 100,
        'probs'          : {cn: float(p) * 100 for cn, p in zip(CLASS_NAMES, probs)},
        'image'          : img_to_b64(image),
    })

@app.route('/api/batch', methods=['POST'])
def api_batch():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not found — run python train.py first'}), 503
    files   = request.files.getlist('files')
    results = []
    for f in files:
        image      = Image.open(f.stream).convert('RGB')
        pred_class, probs = predict_pil(image)
        conf       = float(max(probs)) * 100
        results.append({
            'filename'       : f.filename,
            'predicted_class': pred_class,
            'diagnosis'      : CLASS_LABELS[pred_class],
            'color'          : CLASS_COLORS[pred_class],
            'confidence'     : round(conf, 2),
            'needs_referral' : pred_class in ['3_Severe', '4_PDR'],
            'timestamp'      : datetime.now().strftime('%Y-%m-%d %H:%M'),
        })
    return jsonify({'results': results})

@app.route('/api/model_img/<fname>')
def api_model_img(fname):
    path = os.path.join('saved_model', os.path.basename(fname))
    if os.path.exists(path):
        return send_file(path)
    return '', 404

# ── Main page ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    model_ready  = os.path.exists(MODEL_PATH)
    class_json   = json.dumps({
        'names' : CLASS_NAMES,
        'labels': CLASS_LABELS,
        'colors': CLASS_COLORS,
        'descs' : CLASS_DESC,
        'emojis': CLASS_EMOJI,
    })
    return render_template_string(HTML, model_ready=model_ready, class_json=class_json)

# ═══════════════════════════════════════════════════════════════════════
#  EMBEDDED HTML / CSS / JS  (single-file SPA)
# ═══════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RetinaScan AI — DR Screening</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Exo+2:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
/* ── Reset & vars ─────────────────────────────────────────────────── */
:root{
  --blue:#0ea5e9;--blue2:#3b82f6;--cyan:#22d3ee;
  --dark:#020817;--dark2:#0a1628;--dark3:#0f2044;
  --border:rgba(14,165,233,.15);--border2:rgba(14,165,233,.30);
  --text:#94a3b8;--textbr:#e2e8f0;
  --mono:'Share Tech Mono',monospace;
  --head:'Exo 2',sans-serif;
  --body:'Rajdhani',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{height:100%;font-size:17px}
body{
  font-family:var(--body);
  background:var(--dark);
  color:var(--text);
  min-height:100%;
  overflow-x:hidden;
  font-size:1rem;
}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:var(--dark)}
::-webkit-scrollbar-thumb{background:var(--blue);border-radius:3px}

/* ── Background rain canvas ──────────────────────────────────────── */
#rain{position:fixed;inset:0;z-index:0;opacity:.06;pointer-events:none}
.scanlines{
  position:fixed;inset:0;z-index:1;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px);
}

/* ── Corners ─────────────────────────────────────────────────────── */
.corner{position:fixed;width:56px;height:56px;z-index:5;pointer-events:none}
.c-tl{top:60px;left:10px;border-top:1px solid var(--border2);border-left:1px solid var(--border2)}
.c-tr{top:60px;right:10px;border-top:1px solid var(--border2);border-right:1px solid var(--border2)}
.c-bl{bottom:34px;left:10px;border-bottom:1px solid var(--border2);border-left:1px solid var(--border2)}
.c-br{bottom:34px;right:10px;border-bottom:1px solid var(--border2);border-right:1px solid var(--border2)}

/* ── Navbar ──────────────────────────────────────────────────────── */
nav{
  position:fixed;top:0;left:0;right:0;height:58px;
  background:rgba(2,8,23,.94);
  border-bottom:1px solid var(--border);
  backdrop-filter:blur(18px);
  z-index:200;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 24px;
}
.nav-logo{display:flex;align-items:center;gap:12px;cursor:pointer}
.nav-logo-box{
  width:36px;height:36px;
  background:linear-gradient(135deg,var(--blue2),var(--cyan));
  border-radius:8px;display:flex;align-items:center;justify-content:center;
  font-size:17px;box-shadow:0 0 16px rgba(14,165,233,.4);
  flex-shrink:0;
}
.nav-logo-name{font-family:var(--mono);font-size:1.05rem;color:var(--cyan);letter-spacing:2px;text-shadow:0 0 12px rgba(34,211,238,.5)}
.nav-logo-sub{font-family:var(--mono);font-size:.75rem;color:#1d4ed8;letter-spacing:3px;text-transform:uppercase}
.nav-center{font-family:var(--mono);font-size:.82rem;letter-spacing:4px;color:#1e3a5f;text-transform:uppercase}
.nav-right{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:.88rem}
.ndot{width:7px;height:7px;border-radius:50%;animation:ndot 2s infinite}
@keyframes ndot{0%,100%{opacity:1;box-shadow:0 0 6px currentColor}50%{opacity:.3;box-shadow:none}}

/* ── Ticker ──────────────────────────────────────────────────────── */
footer{
  position:fixed;bottom:0;left:0;right:0;height:32px;
  background:rgba(2,8,23,.96);border-top:1px solid var(--border);
  z-index:200;overflow:hidden;display:flex;align-items:center;
}
.tick-inner{
  display:flex;animation:tick 32s linear infinite;white-space:nowrap;
  font-family:var(--mono);font-size:.82rem;color:#1e3a5f;letter-spacing:2px;
}
@keyframes tick{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}

/* ── Main wrapper ────────────────────────────────────────────────── */
main{
  position:relative;z-index:10;
  padding-top:58px;padding-bottom:32px;
  min-height:100vh;
}

/* ── Pages ───────────────────────────────────────────────────────── */
.page{display:none}
.page.active{display:block}

/* ── Hero ────────────────────────────────────────────────────────── */
.hero{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:calc(100vh - 90px);
  padding:40px 20px;text-align:center;
}
.hero-eyebrow{
  font-family:var(--mono);font-size:.88rem;letter-spacing:5px;
  color:var(--blue);text-transform:uppercase;margin-bottom:18px;
  animation:fadeUp .6s ease both;
}
.hero-title{
  font-family:var(--head);font-size:clamp(3rem,8vw,5.8rem);
  font-weight:800;line-height:1;letter-spacing:-2px;color:#fff;
  text-shadow:0 0 40px rgba(14,165,233,.3);margin-bottom:10px;
  animation:fadeUp .7s ease .1s both;
}
.hero-title span{
  background:linear-gradient(135deg,var(--blue),var(--cyan));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 20px rgba(34,211,238,.45));
}
.hero-sub{
  font-family:var(--head);font-size:1.2rem;font-weight:400;
  color:#334155;margin-bottom:10px;letter-spacing:1px;
  animation:fadeUp .7s ease .2s both;
}
.hero-desc{
  font-size:1.02rem;color:#1e3a5f;max-width:520px;
  line-height:1.65;margin-bottom:44px;
  animation:fadeUp .7s ease .3s both;
}
.hero-btns{
  display:flex;gap:12px;flex-wrap:wrap;justify-content:center;
  margin-bottom:44px;animation:fadeUp .7s ease .4s both;
}

/* ── Pill buttons ────────────────────────────────────────────────── */
.pill{
  padding:11px 28px;border-radius:50px;font-family:var(--mono);
  font-size:.93rem;letter-spacing:2px;text-transform:uppercase;
  cursor:pointer;transition:all .22s;border:1px solid var(--border2);
  color:var(--blue);background:rgba(14,165,233,.06);display:inline-flex;
  align-items:center;gap:8px;
}
.pill:hover{background:rgba(14,165,233,.14);border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 20px rgba(14,165,233,.2);transform:translateY(-1px)}
.pill.primary{background:linear-gradient(135deg,var(--blue2),var(--cyan));border:none;color:#fff;box-shadow:0 0 24px rgba(14,165,233,.35)}
.pill.primary:hover{box-shadow:0 0 36px rgba(14,165,233,.55);transform:translateY(-2px)}

/* ── Status card ─────────────────────────────────────────────────── */
.status-card{
  background:rgba(10,22,40,.85);border:1px solid var(--border);
  border-radius:14px;padding:16px 24px;display:inline-flex;
  align-items:center;gap:14px;max-width:380px;width:100%;
  animation:fadeUp .7s ease .5s both;backdrop-filter:blur(10px);
}
.status-icon{font-size:1.6rem}
.status-title{font-family:var(--mono);font-size:1rem;color:var(--cyan);letter-spacing:1px}
.sbadge{font-family:var(--mono);font-size:.76rem;padding:3px 8px;border-radius:4px;letter-spacing:2px;margin-left:8px}
.status-desc{font-size:.93rem;color:#1e3a5f;margin-top:3px}

/* ── Inner pages ─────────────────────────────────────────────────── */
.inner{padding:20px 28px 24px}
.pg-tag{font-family:var(--mono);font-size:.8rem;letter-spacing:4px;color:var(--blue2);text-transform:uppercase;margin-bottom:4px}
.pg-title{font-family:var(--head);font-size:2.2rem;font-weight:800;color:#f1f5f9;letter-spacing:-.5px;margin-bottom:4px}
.pg-title b{background:linear-gradient(135deg,var(--blue),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.pg-sub{font-size:1rem;color:#1e3a5f;margin-bottom:20px;font-family:var(--body)}

/* ── Inner nav bar ───────────────────────────────────────────────── */
.inner-nav{display:flex;gap:8px;margin-bottom:24px;flex-wrap:wrap}
.inav-btn{
  padding:7px 16px;border-radius:7px;font-family:var(--mono);font-size:.88rem;
  letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;
  border:1px solid var(--border);color:#334155;background:rgba(10,22,40,.8);
  transition:all .2s;display:inline-flex;align-items:center;gap:6px;
}
.inav-btn:hover{background:rgba(14,165,233,.1);border-color:var(--border2);color:var(--cyan);box-shadow:0 0 16px rgba(14,165,233,.15)}

/* ── Cards ───────────────────────────────────────────────────────── */
.card{
  background:rgba(10,22,40,.78);border:1px solid var(--border);
  border-radius:16px;padding:20px;margin-bottom:16px;
  backdrop-filter:blur(12px);transition:border-color .3s,box-shadow .3s;
}
.card:hover{border-color:var(--border2);box-shadow:0 0 24px rgba(14,165,233,.06)}
.ctag{font-family:var(--mono);font-size:.8rem;letter-spacing:3px;text-transform:uppercase;color:#1e40af;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,.04)}

/* ── Two-col grid ────────────────────────────────────────────────── */
.two-col{display:grid;grid-template-columns:1fr 1.3fr;gap:20px}
.two-col-eq{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:800px){.two-col,.two-col-eq{grid-template-columns:1fr}}

/* ── Upload zone ─────────────────────────────────────────────────── */
.upload-zone{
  border:1.5px dashed rgba(14,165,233,.22);border-radius:12px;
  padding:32px 20px;text-align:center;cursor:pointer;
  transition:all .2s;background:rgba(14,165,233,.03);
  position:relative;
}
.upload-zone:hover,.upload-zone.drag{border-color:rgba(14,165,233,.5);background:rgba(14,165,233,.07)}
.upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.upload-icon{font-size:2.2rem;opacity:.35;margin-bottom:10px}
.upload-label{font-family:var(--mono);font-size:.93rem;color:#1e3a5f;letter-spacing:2px}
.upload-hint{font-size:.88rem;color:#0f172a;margin-top:6px}

/* ── Preview image ───────────────────────────────────────────────── */
.img-preview{width:100%;border-radius:10px;object-fit:contain;max-height:320px;display:none}
.img-meta{margin-top:8px;background:rgba(14,165,233,.06);border:1px solid rgba(14,165,233,.12);border-radius:8px;padding:8px 12px;font-family:var(--mono);font-size:.88rem;color:#1e3a5f;display:none}

/* ── Severity list ───────────────────────────────────────────────── */
.svitem{display:flex;align-items:flex-start;gap:12px;padding:9px 0;border-bottom:1px solid rgba(255,255,255,.03)}
.svdot{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:5px}
.svname{font-size:1rem;font-weight:600;color:#cbd5e1;min-width:120px;font-family:var(--head)}
.svdesc{font-size:.9rem;color:#1e3a5f;line-height:1.4}

/* ── Result hero ─────────────────────────────────────────────────── */
.rhero{
  border-radius:16px;padding:28px 20px;text-align:center;
  margin-bottom:14px;position:relative;overflow:hidden;
}
.rhero::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at center top,var(--glow,#0ea5e9) 0%,transparent 65%);
  opacity:.12;pointer-events:none;
}
.rhero-emoji{font-size:3rem;display:block;margin-bottom:10px}
.rhero-grade{font-size:1.65rem;font-weight:700;margin-bottom:8px;font-family:var(--head)}
.rhero-desc{font-size:.97rem;max-width:280px;margin:0 auto 16px;line-height:1.55;color:#334155}
.cpill{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(0,0,0,.4);border-radius:50px;padding:7px 18px;
  font-family:var(--mono);font-size:.95rem;border:1px solid rgba(255,255,255,.06);
}

/* ── Alert ───────────────────────────────────────────────────────── */
.alrt{
  display:flex;align-items:flex-start;gap:10px;
  padding:11px 15px;border-radius:10px;border-left:2px solid;
  font-size:.97rem;margin-bottom:12px;line-height:1.5;font-family:var(--body);
}

/* ── Prob bars ───────────────────────────────────────────────────── */
.prow{margin-bottom:10px}
.phdr{display:flex;justify-content:space-between;margin-bottom:4px}
.plbl{font-size:.95rem;color:#334155;font-family:var(--body)}
.pval{font-size:.95rem;font-weight:700;font-family:var(--mono)}
.ptrack{height:4px;background:rgba(255,255,255,.04);border-radius:50px;overflow:hidden}
.pfill{height:100%;border-radius:50px;transition:width .8s ease}

/* ── Metric boxes ────────────────────────────────────────────────── */
.mboxes{display:grid;gap:12px;margin-bottom:16px}
.mboxes-5{grid-template-columns:repeat(5,1fr)}
.mboxes-6{grid-template-columns:repeat(6,1fr)}
@media(max-width:700px){.mboxes-5,.mboxes-6{grid-template-columns:repeat(3,1fr)}}
.mbox{
  background:rgba(10,22,40,.9);border:1px solid var(--border);
  border-radius:14px;padding:16px 12px;text-align:center;transition:all .25s;
}
.mbox:hover{border-color:var(--border2);transform:translateY(-2px);box-shadow:0 8px 24px rgba(14,165,233,.1)}
.mico{font-size:1.3rem;margin-bottom:7px}
.mval{font-family:var(--mono);font-size:1.5rem;font-weight:700;color:var(--cyan);display:block;line-height:1;text-shadow:0 0 10px rgba(34,211,238,.4)}
.mlbl{font-size:.8rem;color:#1e3a5f;text-transform:uppercase;letter-spacing:1.5px;margin-top:5px;font-family:var(--mono)}

/* ── Results table ───────────────────────────────────────────────── */
.rtable{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.9rem}
.rtable th{color:#1e3a5f;letter-spacing:2px;text-transform:uppercase;font-size:.78rem;padding:9px 12px;border-bottom:1px solid var(--border);text-align:left}
.rtable td{padding:9px 12px;border-bottom:1px solid rgba(255,255,255,.03)}
.rtable tr:hover td{background:rgba(14,165,233,.04)}

/* ── Progress bar ────────────────────────────────────────────────── */
.prog-wrap{margin-bottom:16px;display:none}
.prog-label{font-family:var(--mono);font-size:.88rem;color:#1e3a5f;letter-spacing:2px;margin-bottom:6px}
.prog-bar{height:3px;background:rgba(255,255,255,.06);border-radius:50px;overflow:hidden}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--blue2),var(--cyan));transition:width .3s ease;width:0%}

/* ── Chip ────────────────────────────────────────────────────────── */
.chip{
  display:inline-block;padding:5px 13px;border-radius:4px;font-size:.88rem;
  font-weight:600;font-family:var(--mono);margin:3px;border:1px solid;letter-spacing:1px;
}

/* ── Download btn ────────────────────────────────────────────────── */
.dl-btn{
  display:block;width:100%;padding:11px;border-radius:8px;text-align:center;
  background:linear-gradient(135deg,var(--blue2),#0891b2);color:#fff;
  font-family:var(--mono);font-size:.92rem;letter-spacing:1.5px;cursor:pointer;
  border:none;transition:all .2s;text-transform:uppercase;
}
.dl-btn:hover{box-shadow:0 0 20px rgba(14,165,233,.4);transform:translateY(-1px)}

/* ── Spinner ─────────────────────────────────────────────────────── */
.spin{display:none;text-align:center;padding:60px 20px}
.spin-ring{
  width:48px;height:48px;border-radius:50%;
  border:2px solid rgba(14,165,233,.15);
  border-top-color:var(--cyan);
  animation:spin .8s linear infinite;margin:0 auto 14px;
}
@keyframes spin{to{transform:rotate(360deg)}}
.spin-lbl{font-family:var(--mono);font-size:.92rem;color:#1e3a5f;letter-spacing:3px}

/* ── Strategy boxes ──────────────────────────────────────────────── */
.strat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px}
@media(max-width:700px){.strat-grid{grid-template-columns:1fr}}
.strat-box{background:rgba(14,165,233,.04);border:1px solid rgba(14,165,233,.1);border-radius:12px;padding:16px}
.strat-title{color:var(--blue);font-weight:600;font-size:.95rem;font-family:var(--mono);letter-spacing:2px;margin-bottom:12px}
.strat-row{padding:6px 0;border-bottom:1px solid rgba(255,255,255,.03);font-size:.93rem;color:#334155;font-family:var(--body)}

/* ── Animations ──────────────────────────────────────────────────── */
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.anim{animation:fadeUp .45s ease both}

/* ── Empty state ─────────────────────────────────────────────────── */
.empty{text-align:center;padding:60px 20px}
.empty-icon{font-size:3rem;opacity:.12;margin-bottom:14px}
.empty-label{font-family:var(--mono);color:var(--blue);letter-spacing:3px;font-size:1rem;margin-bottom:6px}
.empty-hint{font-size:.93rem;color:#1e3a5f}

/* ── Model images ────────────────────────────────────────────────── */
.model-img{width:100%;border-radius:10px;border:1px solid var(--border)}
</style>
</head>
<body>

<!-- rain -->
<canvas id="rain"></canvas>
<div class="scanlines"></div>
<div class="corner c-tl"></div>
<div class="corner c-tr"></div>
<div class="corner c-bl"></div>
<div class="corner c-br"></div>

<!-- ── NAVBAR ─────────────────────────────────────────────────────── -->
<nav>
  <div class="nav-logo" onclick="showPage('home')">
    <div class="nav-logo-box">👁️</div>
    <div>
      <div class="nav-logo-name">RETINASCAN</div>
      <div class="nav-logo-sub">AI · DR SCREENING</div>
    </div>
  </div>
  <div class="nav-center">NATIONAL AI COMPETITION 2026</div>
  <div class="nav-right" id="nav-status">
    <div class="ndot" id="status-dot" style="background:#1e3a5f;color:#1e3a5f"></div>
    <span id="status-txt" style="letter-spacing:2px;color:#1e3a5f">CHECKING…</span>
  </div>
</nav>

<!-- ── MAIN ───────────────────────────────────────────────────────── -->
<main>

  <!-- ════ HOME PAGE ════ -->
  <div class="page active" id="page-home">
    <div class="hero">
      <div class="hero-eyebrow">// Smart City Healthcare · Computing Track</div>
      <div class="hero-title">RETINA<span>SCAN</span></div>
      <div class="hero-sub">Diabetic Retinopathy AI Screening System</div>
      <div class="hero-desc">
        Advanced DenseNet121 model trained to detect and grade<br>
        diabetic retinopathy across 5 severity levels from retinal fundus images.
      </div>
      <div class="hero-btns">
        <div class="pill primary" onclick="showPage('scan')">🔍&nbsp; SCAN IMAGE</div>
        <div class="pill" onclick="showPage('batch')">📋&nbsp; BATCH MODE</div>
        <div class="pill" onclick="showPage('dashboard')">📊&nbsp; DASHBOARD</div>
        <div class="pill" onclick="showPage('about')">ℹ️&nbsp; ABOUT</div>
      </div>
      <div class="status-card" id="home-status-card">
        <div class="status-icon">🧠</div>
        <div>
          <div class="status-title" id="home-status-title">DenseNet121 Model</div>
          <div class="status-desc" id="home-status-desc">Checking model status…</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ════ SCAN PAGE ════ -->
  <div class="page" id="page-scan">
    <div class="inner">
      <div class="inner-nav">
        <div class="inav-btn" onclick="showPage('home')">⬅&nbsp; HOME</div>
        <div class="inav-btn" onclick="showPage('scan')">🔍 SCAN</div>
        <div class="inav-btn" onclick="showPage('batch')">📋 BATCH</div>
        <div class="inav-btn" onclick="showPage('dashboard')">📊 DASH</div>
        <div class="inav-btn" onclick="showPage('about')">ℹ️ ABOUT</div>
      </div>
      <div class="pg-tag">// Diagnostic Tool</div>
      <div class="pg-title">Retinal <b>Scan Analysis</b></div>
      <div class="pg-sub">Upload a fundus image for instant AI-powered DR grading</div>
      <div class="two-col">
        <div>
          <div class="card">
            <div class="ctag">// Upload Image</div>
            <div class="upload-zone" id="scan-zone">
              <input type="file" id="scan-input" accept=".jpg,.jpeg,.png" onchange="handleScan(this)">
              <div class="upload-icon">🖼️</div>
              <div class="upload-label">DRAG & DROP OR CLICK TO UPLOAD</div>
              <div class="upload-hint">JPG / JPEG / PNG · Retinal fundus image</div>
            </div>
            <div style="margin-top:12px">
              <img id="scan-preview" class="img-preview">
              <div id="scan-meta" class="img-meta"></div>
            </div>
          </div>
          <div class="card">
            <div class="ctag">// Severity Scale</div>
            <div id="severity-list-scan"></div>
          </div>
        </div>
        <div id="scan-result-area">
          <div class="card">
            <div class="empty">
              <div class="empty-icon">👁️</div>
              <div class="empty-label">READY TO ANALYSE</div>
              <div class="empty-hint">Upload a retinal fundus image to begin</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ════ BATCH PAGE ════ -->
  <div class="page" id="page-batch">
    <div class="inner">
      <div class="inner-nav">
        <div class="inav-btn" onclick="showPage('home')">⬅&nbsp; HOME</div>
        <div class="inav-btn" onclick="showPage('scan')">🔍 SCAN</div>
        <div class="inav-btn" onclick="showPage('batch')">📋 BATCH</div>
        <div class="inav-btn" onclick="showPage('dashboard')">📊 DASH</div>
        <div class="inav-btn" onclick="showPage('about')">ℹ️ ABOUT</div>
      </div>
      <div class="pg-tag">// Multi-Patient</div>
      <div class="pg-title">Batch <b>Screening</b></div>
      <div class="pg-sub">Process multiple patient images simultaneously</div>
      <div class="card">
        <div class="ctag">// Upload Multiple Images</div>
        <div class="upload-zone" id="batch-zone">
          <input type="file" id="batch-input" accept=".jpg,.jpeg,.png" multiple onchange="handleBatch(this)">
          <div class="upload-icon">📁</div>
          <div class="upload-label">SELECT MULTIPLE IMAGES</div>
          <div class="upload-hint">Hold Ctrl / Cmd to select multiple files</div>
        </div>
      </div>
      <div class="prog-wrap" id="batch-prog">
        <div class="prog-label" id="batch-prog-label">PROCESSING…</div>
        <div class="prog-bar"><div class="prog-fill" id="batch-prog-fill"></div></div>
      </div>
      <div id="batch-result-area"></div>
    </div>
  </div>

  <!-- ════ DASHBOARD PAGE ════ -->
  <div class="page" id="page-dashboard">
    <div class="inner">
      <div class="inner-nav">
        <div class="inav-btn" onclick="showPage('home')">⬅&nbsp; HOME</div>
        <div class="inav-btn" onclick="showPage('scan')">🔍 SCAN</div>
        <div class="inav-btn" onclick="showPage('batch')">📋 BATCH</div>
        <div class="inav-btn" onclick="showPage('dashboard')">📊 DASH</div>
        <div class="inav-btn" onclick="showPage('about')">ℹ️ ABOUT</div>
      </div>
      <div class="pg-tag">// Performance</div>
      <div class="pg-title">Model <b>Dashboard</b></div>
      <div class="pg-sub">DenseNet121 architecture overview &amp; training metrics</div>
      <div class="mboxes mboxes-6" id="dash-specs"></div>
      <div class="two-col-eq">
        <div class="card">
          <div class="ctag">// Training Curves</div>
          <img src="/api/model_img/overall_training_curves.png" class="model-img"
               onerror="this.style.display='none';document.getElementById('tc-empty').style.display='block'">
          <div id="tc-empty" style="display:none" class="empty" style="padding:40px">
            <div class="empty-label">TRAIN MODEL FIRST</div>
          </div>
        </div>
        <div class="card">
          <div class="ctag">// Confusion Matrix</div>
          <img src="/api/model_img/overall_confusion_matrix.png" class="model-img"
               onerror="this.style.display='none';document.getElementById('cm-empty').style.display='block'">
          <div id="cm-empty" style="display:none" class="empty" style="padding:40px">
            <div class="empty-label">TRAIN MODEL FIRST</div>
          </div>
        </div>
      </div>
      <div class="card" style="margin-top:0">
        <div class="ctag">// Training Strategy</div>
        <div class="strat-grid">
          <div class="strat-box">
            <div class="strat-title">🏗️ &nbsp;ARCHITECTURE</div>
            <div class="strat-row">DenseNet121 — ImageNet pretrained</div>
            <div class="strat-row">Frozen: stem + denseblock1</div>
            <div class="strat-row">Trainable: denseblock2-4 + head</div>
            <div class="strat-row">Head: 1024→512→256→5 classes</div>
            <div class="strat-row">Progressive dropout: 0.4→0.15</div>
          </div>
          <div class="strat-box">
            <div class="strat-title">⚙️ &nbsp;TRAINING SETUP</div>
            <div class="strat-row">Optimizer: AdamW (lr=5e-5)</div>
            <div class="strat-row">Schedulers: Warmup + Cosine + Plateau</div>
            <div class="strat-row">Batch: 8 · Grad accum: 4 (eff. 32)</div>
            <div class="strat-row">Epochs: 100 · Early stop: 18</div>
            <div class="strat-row">Class weights: [1,2.5,2.5,3,4]</div>
          </div>
          <div class="strat-box">
            <div class="strat-title">🛡️ &nbsp;DATA STRATEGY</div>
            <div class="strat-row">CLAHE + Ben Graham preprocessing</div>
            <div class="strat-row">Offline augmentation (Albumentations)</div>
            <div class="strat-row">Mixup augmentation (alpha=0.2)</div>
            <div class="strat-row">Stratified 3-Fold Cross-Validation</div>
            <div class="strat-row">WeightedRandomSampler · Seed: 42</div>
          </div>
        </div>
      </div>
      <div class="card" style="margin-top:0">
        <div class="ctag">// Cross-Validation Results</div>
        <div class="mboxes mboxes-5">
          <div class="mbox"><div class="mico">📊</div><span class="mval">66.6%</span><div class="mlbl">Mean F1</div></div>
          <div class="mbox"><div class="mico">🎯</div><span class="mval">69.1%</span><div class="mlbl">Best F1</div></div>
          <div class="mbox"><div class="mico">✅</div><span class="mval">66.1%</span><div class="mlbl">Mean Acc</div></div>
          <div class="mbox"><div class="mico">📐</div><span class="mval">±1.99%</span><div class="mlbl">Std Dev</div></div>
          <div class="mbox"><div class="mico">🔁</div><span class="mval">3</span><div class="mlbl">Folds</div></div>
        </div>
        <table class="rtable">
          <thead><tr><th>FOLD</th><th>ACCURACY</th><th>F1 SCORE</th><th>STATUS</th></tr></thead>
          <tbody>
            <tr><td style="color:#94a3b8">Fold 1</td><td style="color:var(--cyan)">63.96%</td><td style="color:var(--cyan)">64.27%</td><td><span style="color:#22c55e">✓ Complete</span></td></tr>
            <tr><td style="color:#94a3b8">Fold 2</td><td style="color:var(--cyan)">68.66%</td><td style="color:#a855f7;font-weight:700">69.12% ★</td><td><span style="color:#22c55e">✓ Best Model</span></td></tr>
            <tr><td style="color:#94a3b8">Fold 3</td><td style="color:var(--cyan)">65.71%</td><td style="color:var(--cyan)">66.42%</td><td><span style="color:#22c55e">✓ Complete</span></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ════ ABOUT PAGE ════ -->
  <div class="page" id="page-about">
    <div class="inner">
      <div class="inner-nav">
        <div class="inav-btn" onclick="showPage('home')">⬅&nbsp; HOME</div>
        <div class="inav-btn" onclick="showPage('scan')">🔍 SCAN</div>
        <div class="inav-btn" onclick="showPage('batch')">📋 BATCH</div>
        <div class="inav-btn" onclick="showPage('dashboard')">📊 DASH</div>
        <div class="inav-btn" onclick="showPage('about')">ℹ️ ABOUT</div>
      </div>
      <div class="pg-tag">// Info</div>
      <div class="pg-title">About <b>RetinaScan AI</b></div>
      <div class="pg-sub">Competition details, model information &amp; medical disclaimer</div>
      <div class="two-col-eq">
        <div>
          <div class="card">
            <div class="ctag">// Competition Info</div>
            <div style="font-family:var(--mono);font-size:1rem;color:#22d3ee;letter-spacing:2px;margin-bottom:16px;text-shadow:0 0 10px rgba(34,211,238,.4)">NATIONAL AI COMPETITION 2026</div>
            <div style="display:flex;flex-direction:column;gap:10px">
              <div style="font-size:.97rem;color:#334155">🎯 &nbsp;Smart City Healthcare — DR Screening</div>
              <div style="font-size:.97rem;color:#334155">🏛️ &nbsp;Sunway University × Rakan Tutor</div>
              <div style="font-size:.97rem;color:#334155">📅 &nbsp;Grand Finals: 13 June 2026</div>
              <div style="font-size:.97rem;color:#334155">👥 &nbsp;Top 6 teams compete in finals</div>
            </div>
            <hr style="border-color:var(--border);margin:16px 0">
            <div style="font-family:var(--mono);font-size:.6rem;letter-spacing:3px;color:#1e3a5f;margin-bottom:12px">PRIZE POOL</div>
            <div style="display:flex;flex-direction:column;gap:8px">
              <div style="display:flex;justify-content:space-between;font-size:.97rem">
                <span style="color:#eab308;font-family:var(--mono)">🥇 1ST PLACE</span>
                <span style="color:#f1f5f9;font-family:var(--mono)">RM 16,000</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:.82rem">
                <span style="color:#94a3b8;font-family:var(--mono)">🥈 2ND PLACE</span>
                <span style="color:#f1f5f9;font-family:var(--mono)">RM 12,000</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:.82rem">
                <span style="color:#cd7f32;font-family:var(--mono)">🥉 3RD PLACE</span>
                <span style="color:#f1f5f9;font-family:var(--mono)">RM 8,000</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:.82rem">
                <span style="color:#334155;font-family:var(--mono)">4TH–6TH</span>
                <span style="color:#f1f5f9;font-family:var(--mono)">RM 5,000</span>
              </div>
            </div>
          </div>
          <div class="card">
            <div class="ctag">// Tech Stack</div>
            <div id="tech-chips"></div>
          </div>
        </div>
        <div>
          <div class="card">
            <div class="ctag">// DR Classification Guide</div>
            <div id="severity-list-about"></div>
          </div>
          <div class="card" style="background:rgba(239,68,68,.03);border-color:rgba(239,68,68,.12)">
            <div class="ctag" style="color:#991b1b">// ⚠ Medical Disclaimer</div>
            <div style="font-size:.97rem;color:#334155;line-height:1.7">
              Built for <span style="color:#f87171">National AI Competition 2026</span>
              — educational &amp; research use only.<br><br>
              <b style="color:#fca5a5">NOT</b> a replacement for professional
              medical diagnosis. Always consult a qualified ophthalmologist
              for clinical decisions.<br><br>
              <span style="font-family:var(--mono);font-size:.7rem;color:#7f1d1d;letter-spacing:1px">
              ALL RESULTS REQUIRE MEDICAL VERIFICATION.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

</main>

<!-- ── TICKER ─────────────────────────────────────────────────────── -->
<footer>
  <div class="tick-inner">
    &nbsp;&nbsp;[ RETINASCAN AI v2.1 ] &nbsp;·&nbsp;
    NATIONAL AI COMPETITION 2026 &nbsp;·&nbsp;
    SUNWAY UNIVERSITY × RAKAN TUTOR &nbsp;·&nbsp;
    DIABETIC RETINOPATHY SCREENING SYSTEM &nbsp;·&nbsp;
    DENSENET121 · PYTORCH · TRANSFER LEARNING &nbsp;·&nbsp;
    SMART CITY HEALTHCARE TRACK &nbsp;·&nbsp;
    [ RETINASCAN AI v2.1 ] &nbsp;·&nbsp;
    NATIONAL AI COMPETITION 2026 &nbsp;·&nbsp;
    SUNWAY UNIVERSITY × RAKAN TUTOR &nbsp;·&nbsp;
    DIABETIC RETINOPATHY SCREENING SYSTEM &nbsp;·&nbsp;
    DENSENET121 · PYTORCH · TRANSFER LEARNING &nbsp;·&nbsp;
    SMART CITY HEALTHCARE TRACK &nbsp;·&nbsp;
  </div>
</footer>

<script>
const CLS = {{ class_json|safe }};
const MODEL_READY_INIT = {{ 'true' if model_ready else 'false' }};

// ── Rain ──────────────────────────────────────────────────────────────
(function(){
  const c=document.getElementById('rain');const cx=c.getContext('2d');
  function resize(){c.width=innerWidth;c.height=innerHeight;}
  resize();addEventListener('resize',resize);
  const chars='01アイウエオカキクケコ0101ABCDEF<>[]{}();'.split('');
  let drops=[];
  function init(){drops=Array(Math.floor(c.width/18)).fill(1);}
  init();addEventListener('resize',init);
  setInterval(()=>{
    cx.fillStyle='rgba(2,8,23,.05)';cx.fillRect(0,0,c.width,c.height);
    cx.fillStyle='#0ea5e9';cx.font='13px Share Tech Mono,monospace';
    drops.forEach((y,i)=>{
      cx.fillText(chars[Math.floor(Math.random()*chars.length)],i*18,y*18);
      if(y*18>c.height&&Math.random()>.975)drops[i]=0;drops[i]++;
    });
  },50);
})();

// ── Page routing ──────────────────────────────────────────────────────
function showPage(name){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  window.scrollTo(0,0);
}

// ── Status ────────────────────────────────────────────────────────────
function updateStatus(ready){
  const dot=document.getElementById('status-dot');
  const txt=document.getElementById('status-txt');
  const desc=document.getElementById('home-status-desc');
  const title=document.getElementById('home-status-title');
  if(ready){
    dot.style.cssText='background:#22c55e;color:#22c55e;box-shadow:0 0 6px #22c55e';
    txt.style.color='#22c55e';txt.textContent='MODEL READY';
    title.innerHTML='DenseNet121 Model <span class="sbadge" style="background:#22c55e22;color:#22c55e;border:1px solid #22c55e44">ACTIVE</span>';
    desc.textContent='Transfer learning · 7.9M params · Preprocessed inference ready';
  }else{
    dot.style.cssText='background:#ef4444;color:#ef4444;box-shadow:0 0 6px #ef4444';
    txt.style.color='#ef4444';txt.textContent='NO MODEL';
    title.innerHTML='DenseNet121 Model <span class="sbadge" style="background:#ef444422;color:#ef4444;border:1px solid #ef444444">OFFLINE</span>';
    desc.textContent='Run python train_kfold.py to train the model';
  }
}
fetch('/api/status').then(r=>r.json()).then(d=>updateStatus(d.model_ready));

// ── Static UI ─────────────────────────────────────────────────────────
function buildSeverityList(id){
  const el=document.getElementById(id);if(!el)return;
  el.innerHTML=CLS.names.map(cn=>`
    <div class="svitem">
      <div class="svdot" style="background:${CLS.colors[cn]};box-shadow:0 0 6px ${CLS.colors[cn]}88"></div>
      <div class="svname">${CLS.labels[cn]}</div>
      <div class="svdesc">${CLS.descs[cn]}</div>
    </div>`).join('');
}
buildSeverityList('severity-list-scan');
buildSeverityList('severity-list-about');

document.getElementById('tech-chips').innerHTML=[
  ['#0ea5e9','PyTorch'],['#22d3ee','DenseNet121'],['#22c55e','Flask'],
  ['#a855f7','Transfer Learning'],['#f97316','OpenCV'],
  ['#eab308','CLAHE'],['#3b82f6','Ben Graham'],['#64748b','Albumentations'],
].map(([c,l])=>`<span class="chip" style="color:${c};background:${c}11;border-color:${c}33">${l}</span>`).join('');

document.getElementById('dash-specs').innerHTML=[
  ['🧠','DenseNet121','ARCH'],['⚙️','7.9M','PARAMS'],
  ['🎯','6.1M','TRAINABLE'],['📐','224×224','INPUT'],
  ['🏷️','5','CLASSES'],['🔥','PyTorch','FRAMEWORK'],
].map(([i,v,l])=>`
  <div class="mbox">
    <div class="mico">${i}</div>
    <span class="mval" style="font-size:1.1rem">${v}</span>
    <div class="mlbl">${l}</div>
  </div>`).join('');

// ── Scan ──────────────────────────────────────────────────────────────
const scanZone=document.getElementById('scan-zone');
['dragover','dragenter'].forEach(e=>scanZone.addEventListener(e,ev=>{ev.preventDefault();scanZone.classList.add('drag')}));
['dragleave','drop'].forEach(e=>scanZone.addEventListener(e,ev=>{ev.preventDefault();scanZone.classList.remove('drag')}));
scanZone.addEventListener('drop',ev=>{
  const files=ev.dataTransfer.files;
  if(files.length){const dt=new DataTransfer();dt.items.add(files[0]);document.getElementById('scan-input').files=dt.files;handleScan(document.getElementById('scan-input'));}
});

function handleScan(input){
  const file=input.files[0];if(!file)return;
  const preview=document.getElementById('scan-preview');
  const meta=document.getElementById('scan-meta');
  const reader=new FileReader();
  reader.onload=e=>{
    preview.src=e.target.result;preview.style.display='block';
    meta.style.display='block';
    meta.innerHTML=`✓ LOADED → <span style="color:#22d3ee">${file.name}</span> &nbsp;·&nbsp; ${(file.size/1024).toFixed(0)} KB`;
  };
  reader.readAsDataURL(file);
  const area=document.getElementById('scan-result-area');
  area.innerHTML=`<div class="card"><div class="spin" style="display:block"><div class="spin-ring"></div><div class="spin-lbl">ANALYSING…</div></div></div>`;
  const fd=new FormData();fd.append('file',file);
  fetch('/api/predict',{method:'POST',body:fd})
    .then(r=>r.json()).then(d=>renderScanResult(d,area))
    .catch(()=>{area.innerHTML=`<div class="card"><div class="empty"><div class="empty-icon">⚠️</div><div class="empty-label" style="color:#ef4444">REQUEST FAILED</div></div></div>`;});
}

function renderScanResult(d,area){
  if(d.error){area.innerHTML=`<div class="card"><div class="empty"><div class="empty-icon">⚠️</div><div class="empty-label" style="color:#ef4444">${d.error}</div></div></div>`;return;}
  const probBars=CLS.names.map(cn=>{
    const p=d.probs[cn];const c=CLS.colors[cn];
    return `<div class="prow">
      <div class="phdr"><span class="plbl">${CLS.labels[cn]}</span><span class="pval" style="color:${c};text-shadow:0 0 8px ${c}66">${p.toFixed(1)}%</span></div>
      <div class="ptrack"><div class="pfill" style="width:${p}%;background:linear-gradient(90deg,${c}77,${c});box-shadow:0 0 6px ${c}55"></div></div>
    </div>`;
  }).join('');
  area.innerHTML=`
    <div class="rhero anim" style="background:linear-gradient(145deg,${d.color}12,${d.color}05,transparent);border:1px solid ${d.color}30;--glow:${d.color}">
      <span class="rhero-emoji">${d.emoji}</span>
      <div class="rhero-grade" style="color:${d.color};text-shadow:0 0 20px ${d.color}66">${d.label}</div>
      <div class="rhero-desc">${d.desc}</div>
      <div class="cpill"><span style="color:#1e3a5f;font-size:.68rem">CONFIDENCE</span><span style="color:${d.color};font-weight:700;text-shadow:0 0 10px ${d.color}88">${d.confidence.toFixed(1)}%</span></div>
    </div>
    <div class="alrt anim" style="background:${d.alert_color}11;border-left-color:${d.alert_color};color:${d.alert_color}cc">
      <span style="font-size:1rem">${d.alert_icon}</span>
      <span style="font-family:var(--mono);letter-spacing:.5px">${d.alert_msg}</span>
    </div>
    <div class="card anim"><div class="ctag">// Class Probabilities</div>${probBars}</div>`;
}

// ── Batch ─────────────────────────────────────────────────────────────
const batchZone=document.getElementById('batch-zone');
['dragover','dragenter'].forEach(e=>batchZone.addEventListener(e,ev=>{ev.preventDefault();batchZone.classList.add('drag')}));
['dragleave','drop'].forEach(e=>batchZone.addEventListener(e,ev=>{ev.preventDefault();batchZone.classList.remove('drag')}));
batchZone.addEventListener('drop',ev=>{
  ev.preventDefault();batchZone.classList.remove('drag');
  const inp=document.getElementById('batch-input');inp.files=ev.dataTransfer.files;handleBatch(inp);
});

function handleBatch(input){
  const files=Array.from(input.files);if(!files.length)return;
  const prog=document.getElementById('batch-prog');
  const progFill=document.getElementById('batch-prog-fill');
  const progLabel=document.getElementById('batch-prog-label');
  const area=document.getElementById('batch-result-area');
  prog.style.display='block';progLabel.textContent=`PROCESSING 0 / ${files.length}…`;progFill.style.width='0%';area.innerHTML='';
  const fd=new FormData();files.forEach(f=>fd.append('files',f));
  let fake=0;
  const ticker=setInterval(()=>{fake=Math.min(fake+2,85);progFill.style.width=fake+'%';progLabel.textContent=`PROCESSING… ${fake}%`;},120);
  fetch('/api/batch',{method:'POST',body:fd})
    .then(r=>r.json())
    .then(d=>{
      clearInterval(ticker);progFill.style.width='100%';progLabel.textContent='COMPLETE';
      setTimeout(()=>{prog.style.display='none';},600);
      if(d.error){area.innerHTML=`<div class="card"><div class="empty"><div class="empty-icon">⚠️</div><div class="empty-label" style="color:#ef4444">${d.error}</div></div></div>`;return;}
      renderBatchResult(d.results,area);
    })
    .catch(()=>{clearInterval(ticker);prog.style.display='none';area.innerHTML=`<div class="card"><div class="empty"><div class="empty-icon">⚠️</div><div class="empty-label" style="color:#ef4444">REQUEST FAILED</div></div></div>`;});
}

function renderBatchResult(results,area){
  const total   = results.length;
  const avgConf = (results.reduce((s,r)=>s+r.confidence,0)/total).toFixed(1)+'%';

  // ── Per-class counts ──────────────────────────────────────────────
  const classDef = [
    { key:'0_No_DR',    label:'No DR',        color:'#22c55e', emoji:'✅' },
    { key:'1_Mild',     label:'Mild NPDR',    color:'#eab308', emoji:'🟡' },
    { key:'2_Moderate', label:'Moderate NPDR',color:'#f97316', emoji:'🟠' },
    { key:'3_Severe',   label:'Severe NPDR',  color:'#ef4444', emoji:'🔴' },
    { key:'4_PDR',      label:'PDR',          color:'#a855f7', emoji:'🚨' },
  ];
  const counts = {};
  classDef.forEach(c=>{ counts[c.key]=results.filter(r=>r.predicted_class===c.key).length; });

  const clear   = counts['0_No_DR'];
  const monitor = counts['1_Mild'] + counts['2_Moderate'];
  const urgent  = counts['3_Severe'] + counts['4_PDR'];

  // ── Summary metric boxes ──────────────────────────────────────────
  const metrics = [
    ['🔍', total,    'TOTAL'],
    ['✅', clear,    'CLEAR'],
    ['⚠️', monitor,  'MONITOR'],
    ['🚨', urgent,   'URGENT'],
    ['📊', avgConf,  'AVG CONF'],
  ].map(([i,v,l])=>`
    <div class="mbox">
      <div class="mico">${i}</div>
      <span class="mval">${v}</span>
      <div class="mlbl">${l}</div>
    </div>`).join('');

  // ── Per-class breakdown bars ──────────────────────────────────────
  const breakdownBars = classDef.map(c=>{
    const count = counts[c.key];
    const pct   = total > 0 ? (count/total*100).toFixed(1) : '0.0';
    const width  = total > 0 ? (count/total*100) : 0;
    return `
      <div style="margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:1rem">${c.emoji}</span>
            <span style="font-family:var(--mono);font-size:.88rem;color:#94a3b8">${c.label}</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px">
            <span style="font-family:var(--mono);font-size:.95rem;font-weight:700;color:${c.color};text-shadow:0 0 8px ${c.color}66">${count}</span>
            <span style="font-family:var(--mono);font-size:.78rem;color:#334155;min-width:42px;text-align:right">${pct}%</span>
          </div>
        </div>
        <div style="height:5px;background:rgba(255,255,255,.04);border-radius:50px;overflow:hidden">
          <div style="height:100%;width:${width}%;background:linear-gradient(90deg,${c.color}77,${c.color});border-radius:50px;box-shadow:0 0 6px ${c.color}55;transition:width .8s ease"></div>
        </div>
      </div>`;
  }).join('');

  // ── Results table rows ────────────────────────────────────────────
  const rows = results.map(r=>`
    <tr>
      <td style="color:#94a3b8">${r.filename}</td>
      <td><span style="color:${r.color};font-weight:700">${r.diagnosis}</span></td>
      <td style="color:var(--cyan)">${r.confidence}%</td>
      <td>${r.needs_referral?'<span style="color:#ef4444">🚨 YES</span>':'<span style="color:#22c55e">✅ NO</span>'}</td>
      <td style="color:#1e3a5f">${r.timestamp}</td>
    </tr>`).join('');

  // ── CSV export ────────────────────────────────────────────────────
  const csvRows = [['filename','predicted_class','diagnosis','confidence_%','needs_referral','timestamp']].concat(
    results.map(r=>[r.filename,r.predicted_class,r.diagnosis,r.confidence,r.needs_referral?'YES':'NO',r.timestamp]));
  const csv    = csvRows.map(r=>r.join(',')).join('\n');
  const csvUrl = URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  const ts     = new Date().toISOString().replace(/[:.]/g,'-').slice(0,16);

  // ── Render all ────────────────────────────────────────────────────
  area.innerHTML=`
    <div class="mboxes mboxes-5 anim">${metrics}</div>

    <div class="card anim">
      <div class="ctag">// DR Grade Distribution — ${total} Images Scanned</div>
      ${breakdownBars}
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,.04);
                  display:flex;justify-content:space-between;font-family:var(--mono);font-size:.82rem">
        <span style="color:#1e3a5f">CLEAR (No DR)</span>
        <span style="color:#1e3a5f">MONITOR (Mild + Moderate)</span>
        <span style="color:#1e3a5f">URGENT (Severe + PDR)</span>
      </div>
      <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:1rem;font-weight:700;margin-top:4px">
        <span style="color:#22c55e">${clear} / ${total}</span>
        <span style="color:#f97316">${monitor} / ${total}</span>
        <span style="color:#ef4444">${urgent} / ${total}</span>
      </div>
    </div>

    <div class="card anim">
      <div class="ctag">// Individual Results</div>
      <div style="overflow-x:auto">
        <table class="rtable">
          <thead><tr>
            <th>FILENAME</th><th>DIAGNOSIS</th>
            <th>CONFIDENCE</th><th>REFERRAL</th><th>TIMESTAMP</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div style="margin-top:14px">
        <a href="${csvUrl}" download="dr_report_${ts}.csv" class="dl-btn">
          💾 &nbsp;DOWNLOAD REPORT (CSV)
        </a>
      </div>
    </div>`;
}

updateStatus(MODEL_READY_INIT);
</script>
</body>
</html>"""

# ── Entry ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    get_model()
    print(f'\n  RetinaScan AI v2.3 — Flask  →  http://localhost:{PORT}')
    print(f'  Preprocessing : Border removal + CLAHE + Ben Graham ✓')
    print(f'  TTA           : 4-variant flip averaging ✓\n')
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
