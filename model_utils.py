# model_utils.py — v0.7
# Treino e inferência leves a partir de rótulos enviados pelo relatório/painel.
from pathlib import Path
import json, csv, math
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    import joblib
except Exception:  # fallback para ambientes sem sklearn
    LogisticRegression = None
    joblib = None

FEATURES = ["vari","ngrdi","ifv","zmin","roi_ratio","near_veg_ratio","area_px","aspect"]

def _occ_aspect(occ:Dict[str,Any])->float:
    try:
        x,y,w,h = occ.get("bbox")[:4]
        return float(w)/float(h) if h else 1.0
    except Exception:
        return 1.0

def occ_features(occ:Dict[str,Any])->List[float]:
    ev = occ.get("evidence",{}) or {}
    vals = [
        ev.get("vari",0.0),
        ev.get("ngrdi",0.0),
        ev.get("ifv",0.0),
        ev.get("zmin",0.0),
        ev.get("roi_ratio",1.0),
        ev.get("near_veg_ratio",0.0),
        float(occ.get("area_px",0)),
        _occ_aspect(occ),
    ]
    # normalização simples de área (log1p) para escala comparável
    vals[6] = float(np.log1p(max(0.0, vals[6])))
    return [float(v) for v in vals]

def collect_labels_and_features(runs_dir:Path)->Tuple[np.ndarray,np.ndarray,int]:
    """Lê todos labels.csv em runs/*/, casa com occurrences_v2.json pelo frame (mais simples), 
    extrai features e retorna X,y. y=1 para 'confirm', 0 para 'fp'."""
    X=[]; y=[]; n_rows=0
    for labels_path in runs_dir.glob("*/labels.csv"):
        run_id = labels_path.parent.name
        occ_path = labels_path.parent/"occurrences_v2.json"
        if not occ_path.exists():
            # tentar v1
            occ_path = labels_path.parent/"occurrences.json"
            if not occ_path.exists(): 
                continue
        try:
            occs = json.loads(occ_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_frame = {}
        for i,occ in enumerate(occs):
            by_frame.setdefault(int(occ.get("frame", occ.get("idx", occ.get("id", i)))), []).append(occ)

        with labels_path.open(newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                n_rows += 1
                try:
                    frame = int(float(row.get("frame", 0)))
                except Exception:
                    frame = 0
                label = (row.get("label") or "").strip().lower()
                y_val = 1 if label in ("confirm","confirmed","positivo","pos") else 0
                cands = by_frame.get(frame, [])
                if not cands:
                    continue
                # pega a maior área do frame como candidata (simples e robusto)
                occ = max(cands, key=lambda o: float(o.get("area_px",0)))
                X.append(occ_features(occ))
                y.append(y_val)
    if not X:
        return np.zeros((0,len(FEATURES))), np.zeros((0,)), n_rows
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), n_rows

def fit_and_save(runs_dir:Path, out_path:Path)->Dict[str,Any]:
    X,y,n_rows = collect_labels_and_features(runs_dir)
    if X.shape[0] < 5 or LogisticRegression is None or joblib is None:
        return {"ok": False, "reason": "dados_insuficientes_ou_dependencias", "n_rows": int(n_rows), "n": int(X.shape[0])}
    # modelo simples, regularizado
    clf = LogisticRegression(max_iter=500, n_jobs=None)
    clf.fit(X,y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURES}, out_path)
    return {"ok": True, "n": int(X.shape[0]), "pos_frac": float(y.mean())}

def load_model(model_path:Path):
    if joblib is None: 
        return None
    try:
        obj = joblib.load(model_path)
        return obj.get("model")
    except Exception:
        return None

def apply_model(model, occs:List[Dict[str,Any]], boost:bool=True)->List[Dict[str,Any]]:
    """Aplica o modelo e, se boost=True, ajusta confiança e tipo final por cima das regras."""
    if model is None or not occs:
        return occs
    import numpy as np
    X = np.array([occ_features(o) for o in occs], dtype=np.float32)
    proba = model.predict_proba(X)[:,1]  # prob de 'confirmado'
    for i,o in enumerate(occs):
        score = float(proba[i])
        o.setdefault("ml", {})["score"] = round(score,3)
        if not boost:
            continue
        # Ajuste de confiança/tipo:
        orig_conf = int(o.get("confidence", 70))
        # confiança final = max(orig_conf, prob*100 arredondado)
        new_conf = max(orig_conf, int(round(score*100)))
        o["confidence"] = new_conf
        # Se modelo contraria (score baixo), marcar como possível FP
        if score < 0.35 and orig_conf < 90:
            o["type"] = o.get("type","ocorrencia") + "|possivel_fp"
        # Se modelo reforça (alto), promover a tipo mais crítico (se aplicável)
        if score > 0.75 and "baixo_sinal" in o.get("type",""):
            o["type"] = "sinal_critico"
    return occs
