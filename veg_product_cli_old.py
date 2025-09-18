#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
veg_product_cli.py — v0.6d
Processa um vídeo RGB e produz:
- occurrences_v2.json (ou occurrences.json)
- pasta thumbs/ com frame####.png e frame####_overlay.png
- report.html (template auto v2/v1 + miniatura + popup)
- grids e resumos básicos

Uso:
  python veg_product_cli.py --input /caminho/video.mp4 --out /pasta/out [--every 30]
"""
import argparse, json, os, sys, math, csv, time, uuid
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def indices_from_bgr(frame):
    """Calcula três índices simulados (escala 0..1 aprox)."""
    b,g,r = cv2.split(frame.astype(np.float32)+1e-6)
    vari = (g - r) / (g + r - b + 1e-6)
    ngrdi = (g - r) / (g + r + 1e-6)
    ifv = g / (r + g + b + 1e-6)
    return np.clip(vari, -1, 1), np.clip(ngrdi, -1, 1), np.clip(ifv, 0, 1)

def detect_low_veg(vari, ngrdi, ifv):
    """Máscara simples de 'baixo verdor' combinando limiares heurísticos."""
    m1 = vari < 0.03
    m2 = ngrdi < 0.03
    m3 = ifv   < 0.35
    agree = (m1.astype(np.uint8) + m2.astype(np.uint8) + m3.astype(np.uint8))
    mask = agree >= 2  # exige concordância de pelo menos 2 índices
    return mask, agree

def contours_from_mask(mask: np.ndarray, min_area=1500):
    """Extrai contornos e bboxes a partir da máscara (área mínima em px)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_OPEN, k, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: 
            continue
        x,y,w,h = cv2.boundingRect(c)
        bboxes.append((x,y,w,h, area, c))
    return bboxes

def overlay_polygons(img, contours, color=(0,0,255), thickness=2):
    out = img.copy()
    if contours:
        cv2.drawContours(out, [c for *_rest, c in contours], -1, color, thickness)
    return out

def save_thumb_pair(out_dir: Path, idx: int, frame_bgr: np.ndarray, overlay_bgr: np.ndarray):
    tdir = out_dir / "thumbs"
    ensure_dir(tdir)
    cv2.imwrite(str(tdir / f"frame{idx}.png"), frame_bgr)
    cv2.imwrite(str(tdir / f"frame{idx}_overlay.png"), overlay_bgr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="vídeo de entrada (mp4/mov)")
    ap.add_argument("--out", required=True, help="pasta de saída (run)")
    ap.add_argument("--every", type=int, default=30, help="processar a cada N frames (padrão 30)")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)

    # gravar o report.html (template auto) desde já
    (out_dir / "report.html").write_text("""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8"/>
<title>Relatório de Ocorrências — (auto v2/v1 + miniatura + popup + rótulo)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root{ --muted:#6b7280; --border:#e5e7eb; }
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:20px;color:#111}
  h1{margin:0 0 12px}
  .occ{display:flex;gap:10px;align-items:flex-start;border:1px solid var(--border);border-radius:10px;padding:8px;margin-bottom:10px;background:#fff}
  .thumb{width:220px;height:110px;object-fit:cover;border-radius:8px;border:1px solid #eee;cursor:pointer}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid var(--border);background:#f8f9fa}
  .pill-high{background:#fee2e2;border-color:#fecaca}.pill-mid{background:#fef3c7;border-color:#fde68a}.pill-low{background:#dcfce7;border-color:#bbf7d0}
  .link{background:none;border:none;color:#2563eb;cursor:pointer;padding:0;font:inherit}
  .muted{color:var(--muted)}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .hr{height:1px;background:#eee;margin:10px 0}
  /* Modal */
  .backdrop{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;padding:20px;z-index:1000}
  .modal{background:#fff;border-radius:12px;max-width:96vw;max-height:92vh;overflow:auto;box-shadow:0 10px 40px rgba(0,0,0,.2)}
  .modal header{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid #eee;position:sticky;top:0;background:#fff}
  .tabs{display:flex;gap:8px;border-bottom:1px solid #eee;margin:10px}
  .tab{padding:6px 10px;border:1px solid #eee;border-bottom:none;border-radius:8px 8px 0 0;background:#f8fafc;cursor:pointer}
  .tab.on{background:#fff;font-weight:600}
</style>
</head>
<body>
<h1>Relatório de Ocorrências</h1>
<p class="muted">Este arquivo procura <code>occurrences_v2.json</code> e, se não achar, usa <code>occurrences.json</code>. As miniaturas devem estar em <code>./thumbs/</code>. Para rotular, a API precisa estar acessível (CORS) e a URL pode ser definida com <code>localStorage.setItem('API_URL','http://localhost:8000')</code>.</p>
<div id="wrap"></div>

<div class="backdrop" id="bd" onclick="if(event.target.id==='bd') closeM()">
  <div class="modal">
    <header><div id="mt">Visualização</div><button onclick="closeM()">×</button></header>
    <div class="tabs">
      <div id="to" class="tab on" onclick="tab('o')">Overlay</div>
      <div id="tf" class="tab" onclick="tab('f')">Frame</div>
      <div id="tc" class="tab" onclick="tab('c')">Comparar</div>
    </div>
    <div id="vo"><img id="io" style="max-width:96vw;max-height:80vh"/></div>
    <div id="vf" style="display:none"><img id="if" style="max-width:96vw;max-height:80vh"/></div>
    <div id="vc" style="display:none;display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <img id="ic1" style="max-width:48vw;max-height:80vh"/>
      <img id="ic2" style="max-width:48vw;max-height:80vh"/>
    </div>
  </div>
</div>

<script>
function tab(w){ document.getElementById('to').classList.toggle('on',w==='o'); document.getElementById('tf').classList.toggle('on',w==='f'); document.getElementById('tc').classList.toggle('on',w==='c'); document.getElementById('vo').style.display=(w==='o')?'block':'none'; document.getElementById('vf').style.display=(w==='f')?'block':'none'; document.getElementById('vc').style.display=(w==='c')?'grid':'none'; }
function openM(frame, overlay, title){ document.getElementById('io').src=overlay||''; document.getElementById('if').src=frame||''; document.getElementById('ic1').src=frame||''; document.getElementById('ic2').src=overlay||''; document.getElementById('mt').textContent=title||'Visualização'; document.getElementById('bd').style.display='flex'; tab('o'); }
function closeM(){ document.getElementById('bd').style.display='none'; }
function sevClass(s){ if(s>=1.3) return 'pill-high'; if(s>=0.8) return 'pill-mid'; return 'pill-low'; }

async function fetchFirst(paths){
  for (const p of paths){
    try { const r = await fetch(p); if (r.ok) { const j = await r.json(); return {data:j, path:p}; } } catch(e){}
  }
  throw new Error('Nenhum arquivo de ocorrências encontrado.');
}

async function main(){
  const base = location.href.substring(0, location.href.lastIndexOf('/')+1);
  const {data:occs} = await fetchFirst([base+'occurrences_v2.json', base+'occurrences.json']);
  const frameName = o=> 'frame'+(o.frame ?? o.idx ?? o.id)+'.png';
  const overlayName = o=> 'frame'+(o.frame ?? o.idx ?? o.id)+'_overlay.png';
  const thumb = name => base + 'thumbs/' + name;

  const wrap = document.getElementById('wrap');
  occs.forEach((o,i)=>{
    const frame = thumb(frameName(o));
    const overlay = thumb(overlayName(o));
    const sev = (o.evidence && (o.evidence.severity ?? o.severity)) ?? (o.severity ?? 0);
    const conf = o.confidence ?? o.conf ?? 0;
    const type = o.type ?? o.label ?? 'ocorrencia';
    const time = o.time_s ?? o.time ?? 0;
    const area = o.area_px ?? o.area ?? 0;
    const ev = o.evidence || {zmin:o.zmin, roi_ratio:o.roi_ratio, near_veg_ratio:o.near_veg_ratio};

    const d = document.createElement('div'); d.className='occ';
    d.innerHTML = `
      <img class="thumb" src="${overlay}" onclick="openM('${frame}','${overlay}','~${(time.toFixed?time.toFixed(1):time)}s (Frame ~${o.frame||o.idx||o.id})')"/>
      <div class="b">
        <div><strong>~${(time.toFixed?time.toFixed(1):time)}s</strong> (Frame ~${o.frame||o.idx||o.id}) — <span class="pill">${type}</span> · <span class="pill ${sevClass(+sev)}">sev ${(+sev).toFixed(2)}</span> — conf ${conf}%</div>
        <div>${o.recommendation||o.rec||''}</div>
        <div class="muted">zmin:${ev?.zmin ?? ''} | ROI:${ev?.roi_ratio ?? ''} | nearVeg:${ev?.near_veg_ratio ?? ''} | área:${area}</div>
        <div>
          <button class="link" onclick="openM('${frame}','${overlay}','Frame ~${o.frame||o.idx||o.id}')">ver frame</button> ·
          <button class="link" onclick="openM('${frame}','${overlay}','Overlay ~${o.frame||o.idx||o.id}')">ver overlay</button> ·
          <button class="link" onclick="sendLabel(${i}, 'confirm')">confirmar ocorrência</button> ·
          <button class="link" onclick="sendLabel(${i}, 'fp')">marcar como falso positivo</button>
          <span id="lab-${i}" class="muted"></span>
        </div>
      </div>`;
    wrap.appendChild(d);
  });

  window.sendLabel = async function(idx, label){
    const API = localStorage.getItem('API_URL') || 'http://localhost:8000';
    const o = occs[idx]; const msg = document.getElementById('lab-'+idx);
    msg.textContent = ' — enviando...';
    try {
      const payload = { run_id: o.run_id || 'report', frame:o.frame, time_s:o.time_s, type:o.type, bbox:o.bbox, label, evidence:o.evidence };
      const r = await fetch(API + '/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      msg.textContent = r.ok ? (label==='confirm'?' — confirmado':' — falso positivo') : ' — erro ao enviar';
    } catch (e) {
      msg.textContent = ' — falha de rede';
    }
  }
}
main();
</script>
</body>
</html>
""", encoding="utf-8")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print("ERRO: não abriu vídeo", file=sys.stderr); sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    occurrences = []
    grid_samples = []
    idx = 0
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.every != 0:
            idx += 1
            continue
        # reduz escala para acelerar (mantém thumbs em full)
        small = cv2.resize(frame, (w//2, h//2)) if max(w,h) > 1280 else frame
        vari, ngrdi, ifv = indices_from_bgr(small)
        mask, agree = detect_low_veg(vari, ngrdi, ifv)

        # stats simples para severidade
        zmin = float(np.quantile(vari, 0.1))
        roi_ratio = 1.0
        near_veg_ratio = float((ifv > 0.4).mean())

        # contornos em resolução small
        bboxes = contours_from_mask(mask, min_area=max(1200, (small.shape[0]*small.shape[1])//800))
        # reescala bboxes/contornos p/ fullres se necessário
        scale_x = frame.shape[1] / small.shape[1]
        scale_y = frame.shape[0] / small.shape[0]
        contours_full = []
        for x,y,wc,hc,area,c in bboxes:
            c2 = (c.astype(np.float32) * [scale_x, scale_y]).astype(np.int32)
            contours_full.append((int(x*scale_x), int(y*scale_y), int(wc*scale_x), int(hc*scale_y), area*scale_x*scale_y, c2))

        # overlay
        overlay = overlay_polygons(frame, contours_full)
        save_thumb_pair(out_dir, idx, frame, overlay)

        # registrar ocorrências
        t = idx / fps
        for (x,y,wc,hc,area,c2) in contours_full:
            # valores médios na bbox (em small para custo menor)
            xs, ys, ws, hs = int(x/scale_x), int(y/scale_y), int(wc/scale_x), int(hc/scale_y)
            xs = max(0, xs); ys = max(0, ys)
            crop_vari = vari[ys:ys+hs, xs:xs+ws]
            crop_ngr = ngrdi[ys:ys+hs, xs:xs+ws]
            crop_ifv = ifv[ys:ys+hs, xs:xs+ws]
            if crop_vari.size < 50: 
                continue
            mVARI = float(np.mean(crop_vari))
            mNGRDI = float(np.mean(crop_ngr))
            mIFV = float(np.mean(crop_ifv))
            agree_local = int(((crop_vari<0.03).mean() + (crop_ngr<0.03).mean() + (crop_ifv<0.35).mean()) >= 2)

            # severidade heurística
            severity = max(0.0, (0.15 - mVARI)*4) + max(0.0, (0.12 - mNGRDI)*3) + max(0.0, (0.40 - mIFV)*2)
            severity = float(severity)

            # tipo simplificado
            typ = "buraco_dossel" if mIFV<0.28 and mVARI<0.02 else ("solo_exposto" if mVARI<0.01 and mIFV<0.22 else "baixo_sinal")
            rec = "Prioridade alta: vistoriar imediatamente; investigar irrigação/solo/pragas no entorno." if severity>=1.3 else \
                  ("Atenção moderada: monitorar evolução; checar manejo/irrigação/piso." if severity>=0.8 else \
                   "Sinal baixo/ambíguo; se possível, revisar no campo ou repetir voo para confirmar.")

            occ = {
                "run_id": out_dir.name,
                "frame": idx,
                "time_s": round(t,3),
                "bbox": [x,y,wc,hc],
                "area_px": int(area),
                "type": typ,
                "confidence": 90 if severity>=0.8 else 70,
                "recommendation": rec,
                "evidence": {
                    "vari": round(mVARI,3),
                    "ngrdi": round(mNGRDI,3),
                    "ifv": round(mIFV,3),
                    "zmin": round(zmin,3),
                    "roi_ratio": round(roi_ratio,3),
                    "near_veg_ratio": round(near_veg_ratio,3),
                    "severity": round(severity,2)
                }
            }
            occurrences.append(occ)

        processed += 1
        idx += 1

    cap.release()

    # salvar JSON (v2)
    out_json = out_dir / "occurrences_v2.json"
    out_json.write_text(json.dumps(occurrences, ensure_ascii=False, indent=2), encoding="utf-8")

    # salvar resumos simples
    (out_dir / "resumo.txt").write_text(f"frames_processados={processed}\nfps={fps}\nocc={len(occurrences)}\n", encoding="utf-8")

    # grids ilustrativos (opcional simples: pegar último frame)
    if processed>0:
        cv2.imwrite(str(out_dir / "grid_vari.png"), np.full((120,320,3), (180,230,180), np.uint8))
        cv2.imwrite(str(out_dir / "grid_ngrdi.png"), np.full((120,320,3), (180,220,240), np.uint8))
        cv2.imwrite(str(out_dir / "grid_ifv.png"),   np.full((120,320,3), (180,240,200), np.uint8))

    print(f"[OK] Processado. Ocorrências: {len(occurrences)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
