from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch, joblib
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


DET_REPO = "jueedeshmukh/01_toxic_detection_daniel"

det_tokenizer = AutoTokenizer.from_pretrained(DET_REPO)
det_model = AutoModelForSequenceClassification.from_pretrained(DET_REPO).eval()

@torch.no_grad()
def detect(query: str) -> int: #detect turns the query into tokens, runs the model, and gets 2 scores
    enc = det_tokenizer([query], truncation=True, max_length=512, return_tensors="pt")
    logits = det_model(**enc).logits
    return int(logits.argmax(dim=-1)[0])   # 0 = safe, 1 = toxic


CLASSIFICATION_REPO = "jueedeshmukh/toxic_prompt_classification_riya"
MAX_LENGTH = 128
W_B = 0.5
LABELS = ["Disinformation", "Economic harm", "Expert advice", "Fraud/Deception",
          "Government decision-making", "Harassment/Discrimination", "Malware/Hacking",
          "Physical harm", "Privacy", "Sexual/Adult content"]

# Load once when the app starts
tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_REPO, subfolder="finetuned")
encoder_b = AutoModel.from_pretrained(CLASSIFICATION_REPO, subfolder="contrastive").eval()
model_c = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_REPO, subfolder="finetuned").eval()
clf_b = joblib.load(hf_hub_download(CLASSIFICATION_REPO, "clfB.joblib"))

@torch.no_grad()
def classify(query: str) -> dict:
    enc = tokenizer([query], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    # Approach B: CLS embedding -> normalize -> logistic regression
    cls = encoder_b(**enc).last_hidden_state[:, 0]
    emb = F.normalize(cls, dim=-1).numpy()
    p_b = clf_b.predict_proba(emb)[0]

    # Approach C: direct class probabilities
    p_c = torch.softmax(model_c(**enc).logits, dim=-1)[0].numpy()

    # Ensemble: 50/50 average, pick the top category
    p = W_B * p_b + (1 - W_B) * p_c
    i = int(p.argmax())
    return {"category": LABELS[i], "confidence": round(float(p[i]), 3)}

def run_pipeline(query: str) -> dict:
    if detect(query) == 0:
        return {"label": "safe"}
    return {"label": "unsafe", **classify(query)}