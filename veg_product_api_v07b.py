# -*- coding: utf-8 -*-
# veg_product_api_v07b.py — v0.7b
# Lê options_json do formulário e repassa flags ao CLI; mantém /train com treino de modelo.
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, subprocess, os, csv, time, json, sys, uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from model_utils import fit_and_save, load_model, apply_model

APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = RUNS_DIR / "model.joblib"

app = FastAPI(title="AgroVision API v0.7b", version="0.7b")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def add_artifact(arts:List[Dict[str,str]], out_dir:Path, name:str):
    p = out_dir / name
    if p.exists():
        arts.append({"name": name, "url": f"/download/{out_dir.name}/{name}"})

@app.get("/status")
def status():
    return {"ok": True, "version": "0.7b", "has_model": MODEL_PATH.exists()}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    options_json: Optional[str] = Form(None),
):
    run_id = uuid.uuid4().hex[:8]
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = out_dir / file.filename
    with in_path.open("wb") as f:
        f.write(await file.read())

    # Parse options (all optional)
    opts = {}
    if options_json:
        try:
            opts = json.loads(options_json)
        except Exception:
            opts = {}

    # Build CLI command
    cli = APP_DIR / "veg_product_cli.py"
    py = sys.executable
    cmd = [py, str(cli), "--input", str(in_path), "--out", str(out_dir)]
    # Map options -> flags (if provided)
    if isinstance(opts.get("every"), (int, float)) and int(opts["every"])>0:
        cmd += ["--every", str(int(opts["every"]))]
    if isinstance(opts.get("min_area"), (int, float)) and int(opts["min_area"])>=0:
        cmd += ["--min-area", str(int(opts["min_area"]))]
    if int(opts.get("agree_k", 0)) in (2,3):
        cmd += ["--agree-k", str(int(opts["agree_k"]))]
    if isinstance(opts.get("min_severity"), (int, float)):
        cmd += ["--min-severity", str(float(opts["min_severity"]))]
    # soil_guard True => default; if false, add --disable-soil-guard
    if opts.get("soil_guard") is False:
        cmd += ["--disable-soil-guard"]

    ok=True; err=""; out_text=""

    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out_text = cp.stdout or ""
    except subprocess.CalledProcessError as e:
        ok=False; err=e.stderr or e.stdout or str(e)

    # Pós-processamento com modelo (se existir)
    occ_path = out_dir / "occurrences_v2.json"
    if ok and occ_path.exists() and MODEL_PATH.exists():
        try:
            occs = json.loads(occ_path.read_text(encoding="utf-8"))
            model = load_model(MODEL_PATH)
            occs2 = apply_model(model, occs, boost=True)
            occ_path.write_text(json.dumps(occs2, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            err += f" | model_apply_error: {e}"

    # Artefatos
    arts=[]
    for name in ["report.html","report_v2.html","occurrences_v2.json","occurrences.json","resumo.txt","grid_vari.png","grid_ngrdi.png","grid_ifv.png"]:
        add_artifact(arts, out_dir, name)
    if (out_dir/"thumbs").exists():
        arts.append({"name":"thumbs/","url":f"/download/{run_id}/thumbs"})

    return JSONResponse({"ok": ok, "run_id": run_id, "artifacts": arts, "stderr": err, "stdout": out_text, "cmd": cmd}, status_code=200 if ok else 500)

@app.post("/train")
async def train(payload: Dict[str,Any]):
    run_id = (payload.get("run_id") or "unknown")
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "labels.csv"
    new = not path.exists()
    hdr = ["ts","frame","time_s","type","label","bbox","evidence_json"]
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(hdr)
        w.writerow([time.time(), payload.get("frame"), payload.get("time_s"),
                    payload.get("type"), payload.get("label"),
                    json.dumps(payload.get("bbox")), json.dumps(payload.get("evidence"))])
    # treino leve
    info = fit_and_save(RUNS_DIR, MODEL_PATH)
    return {"ok": True, "model": info, "model_path": str(MODEL_PATH)}

@app.get("/download/{run_id}/{path:path}")
async def download(run_id:str, path:str):
    base = RUNS_DIR / run_id
    target = (base / path).resolve()
    if base not in target.parents and base != target:
        return JSONResponse({"detail":"Invalid path"}, status_code=400)
    if target.is_dir():
        items = [f"<li><a href='/download/{run_id}/{p.relative_to(base).as_posix()}'>{p.name}</a></li>" for p in sorted(target.iterdir())]
        return HTMLResponse("<ul>"+ "\n".join(items) + "</ul>")
    if not target.exists():
        return JSONResponse({"detail":"Not Found"}, status_code=404)
    return FileResponse(str(target))

if __name__ == "__main__":
    uvicorn.run("veg_product_api_v07b:app", host="0.0.0.0", port=8000, reload=False)
