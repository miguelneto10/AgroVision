#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
veg_product_cli.py — v0.7c2
Compat:
- aceita flags com hífen OU sublinhado:
  --min-area / --min_area
  --agree-k / --agree_k
  --min-severity / --min_severity
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# Template inline de fallback (mantém funcionamento do relatório)
REPORT_INLINE = "<!doctype html><meta charset='utf-8'><p>Relatório gerado (use report_backend_auto.html para layout completo).</p>"

def read_report_template():
    here = Path(__file__).resolve().parent
    tpl = here / "report_backend_auto.html"
    if tpl.exists():
        try: return tpl.read_text(encoding="utf-8")
        except Exception: pass
    return REPORT_INLINE

def indices_from_bgr(frame):
    b,g,r = cv2.split(frame.astype(np.float32)+1e-6)
    vari = (g - r) / (g + r - b + 1e-6)
    ngrdi = (g - r) / (g + r + 1e-6)
    ifv = g / (r + g + b + 1e-6)
    return np.clip(vari,-1,1), np.clip(ngrdi,-1,1), np.clip(ifv,0,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=30, help="processar a cada N frames (padrão 30)")
    # aliases hífen/sublinhado
    ap.add_argument("--min-area","--min_area", dest="min_area", type=int, default=6000, help="área mínima (px)")
    ap.add_argument("--agree-k","--agree_k", dest="agree_k", type=int, default=2, choices=[2,3], help="concordância mínima entre índices (2 ou 3)")
    ap.add_argument("--min-severity","--min_severity", dest="min_severity", type=float, default=0.9, help="severidade mínima")
    ap.add_argument("--disable-soil-guard", action="store_true", help="desativar guard de solo (por padrão está ATIVO)")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir); ensure_dir(out_dir/"thumbs")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print("ERRO: não abriu vídeo", file=sys.stderr); sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_occs = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % args.every != 0:
            idx += 1; continue

        work = cv2.resize(frame, (W//2, H//2)) if max(W,H) > 1280 else frame
        vari, ngrdi, ifv = indices_from_bgr(work)

        # Soil-guard (ativado por padrão): se o frame é majoritariamente solo, não reporta
        if not args.disable_soil_guard:
            if float(ifv.mean()) < 0.28 and float(ngrdi.mean()) < 0.02:
                overlay = frame.copy()
                cv2.imwrite(str(out_dir/"thumbs"/f"frame{idx}.png"), frame)
                cv2.imwrite(str(out_dir/"thumbs"/f"frame{idx}_overlay.png"), overlay)
                idx += 1; continue

        # Máscara: consenso entre índices com thresholds conservadores
        m1 = vari < 0.02        # VARI baixo
        m2 = ngrdi < 0.02       # NGRDI baixo
        m3 = ifv   < 0.30       # IFV baixo
        agree = (m1.astype(np.uint8)+m2.astype(np.uint8)+m3.astype(np.uint8)) >= args.agree_k

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.morphologyEx(agree.astype(np.uint8)*255, cv2.MORPH_OPEN, k, iterations=1)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = frame.copy()
        sx = frame.shape[1]/work.shape[1]; sy = frame.shape[0]/work.shape[0]

        for c in cnts:
            area = cv2.contourArea(c)
            if area < args.min_area: 
                continue
            x,y,w,h = cv2.boundingRect(c)
            wx,wy,ww,wh = int(x*work.shape[1]/frame.shape[1]), int(y*work.shape[0]/frame.shape[0]), int(w*work.shape[1]/frame.shape[1]), int(h*work.shape[0]/frame.shape[0])
            crop_v = vari[wy:wy+wh, wx:wx+ww]; crop_n=ngrdi[wy:wy+wh, wx:wx+ww]; crop_f=ifv[wy:wy+wh, wx:wx+ww]
            if crop_v.size < 25: 
                continue
            mV=float(crop_v.mean()); mN=float(crop_n.mean()); mF=float(crop_f.mean())
            sev = max(0.0,(0.15-mV)*4)+max(0.0,(0.12-mN)*3)+max(0.0,(0.40-mF)*2)
            if sev < args.min_severity:
                continue

            cc = (c.astype(np.float32) * [sx, sy]).astype(np.int32)
            cv2.drawContours(overlay,[cc],-1,(0,0,255),2)

            all_occs.append({
                "frame": idx, "time_s": round(idx/float(fps),3),
                "bbox": [int(x*sx), int(y*sy), int(w*sx), int(h*sy)],
                "area_px": int(area*sx*sy),
                "type": "baixo_sinal",
                "confidence": 80 if sev < 1.3 else 92,
                "recommendation": "Atenção moderada: monitorar; checar irrigação/manejo." if sev < 1.3 else "Prioridade alta: vistoriar imediatamente; verificar irrigação/solo/pragas.",
                "evidence": {"vari": round(mV,3), "ngrdi": round(mN,3), "ifv": round(mF,3), "severity": round(float(sev),2)}
            })

        cv2.imwrite(str(out_dir/"thumbs"/f"frame{idx}.png"), frame)
        cv2.imwrite(str(out_dir/"thumbs"/f"frame{idx}_overlay.png"), overlay)
        idx += 1

    cap.release()

    (out_dir/"occurrences_v2.json").write_text(json.dumps(all_occs, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir/"report.html").write_text(read_report_template(), encoding="utf-8")
    print(f"[OK] Ocorrências: {len(all_occs)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
