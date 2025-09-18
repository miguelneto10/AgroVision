# AgroVision
# AgroVision — v0.7c

Protótipo para **análise de vídeo RGB** (drone) com detecção de **manchas de baixo vigor** via índices visuais (VARI, NGRDI, IFV), visualização web, **rótulos** (aprimoramento supervisionado leve) e **API FastAPI**.

> Esta versão inclui: painel com abas (Overlay/Frame/Comparar), barra de progresso, rótulos → `/train`, API que lê `options_json` e repassa flags ao CLI, e um CLI **conservador** para reduzir falsos positivos. O modelo leve é treinado com os rótulos e salva em `runs/model.joblib`.

---

## ✨ Funcionalidades

- **Painel Web (estático)**: upload do vídeo, controles de sensibilidade, miniaturas, abas Overlay/Frame/Comparar, **rótulos** (“confirmar ocorrência” / “falso positivo”), botão “Abrir relatório”.
- **API (FastAPI)**: `/analyze` (processa vídeo), `/train` (recebe rótulos e re‑treina), `/download/<run_id>/...` (serve artefatos), `/status` (status).
- **CLI**: processamento com índices visuais e heurísticas conservadoras, geração de thumbs, relatório HTML e `occurrences_v2.json`.
- **Modelo leve** (opcional): Logistic Regression treinada com rótulos; ajusta confiança e tipo nas próximas análises.

---

## 📁 Estrutura do projeto

```
AgroVision/
├─ veg_panel_index.html            # Painel Web (v0.7c)
├─ veg_product_api_v07c.py         # API FastAPI (retorno 200 + ok/erro + error.txt)
├─ veg_product_cli.py              # CLI (v0.7c2, aceita hífen e sublinhado nas flags)
├─ model_utils.py                  # Treino/inferência do modelo leve
├─ report_backend_auto.html        # Template de relatório
└─ runs/
   └─ <run_id>/
      ├─ <video>.mp4|mov
      ├─ occurrences_v2.json
      ├─ report.html
      ├─ thumbs/ (frame####.png, frame####_overlay.png)
      ├─ labels.csv                # (aparece após /train)
      └─ error.txt                 # logs do processamento
```

---

## 🚀 Como rodar (dev)

### 1) Dependências
```bash
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn opencv-python-headless numpy joblib scikit-learn
```

### 2) API
```bash
uvicorn veg_product_api_v07c:app --host 0.0.0.0 --port 8000
```

### 3) Painel
Abra **`veg_panel_index.html`** no navegador. Em “Diagnóstico” você pode ajustar a URL da API.

---

## 🧪 Fluxo de uso

1. **Upload + Analisar** no painel. A barra mostra o progresso do upload.
2. Após o processamento, veja **thumbs** e clique em **ver frame** / **ver overlay** (abre popup com abas **Overlay | Frame | Comparar**).
3. Abra o **relatório** (botão “Abrir relatório”) → `report.html` do run.
4. **Rotule** ocorrências (confirmar / falso positivo) → o painel faz `POST /train` e o modelo é salvo em `runs/model.joblib`.
5. Em novas análises, a API aplica o modelo para **ajustar confiança/tipo**.

---

## 🎚️ Controles (enviados à API via `options_json`)

- **Step (every)**: processar a cada **N** frames (padrão 30).
- **Área mínima (px)**: filtra manchas pequenas (padrão 6000).
- **Concordância (2/3)**: quantos índices precisam concordar (padrão 2).
- **Severidade mínima**: 0.6 (sensível) / 0.9 (padrão) / 1.3 (alta).
- **Soil‑guard**: ligado por padrão (ignora frames dominados por solo).

> Se vier “concluído sem ocorrências”, diminua **severidade** e **área mínima**, reduza **every** e desative o soil‑guard temporariamente.

---

## 🔌 API — Endpoints

### `GET /status`
Retorna versão e se o modelo existe.

### `POST /analyze`
**FormData**: `file` + `options_json` (opcional, JSON com os controles).
**Resposta**: sempre **200** com `{ ok: true|false, run_id, artifacts[], stderr, stdout, cmd[] }`.
Artefatos incluem `report.html`, `occurrences_v2.json`, `thumbs/` e `error.txt` (quando houver).

### `POST /train`
Recebe rótulo `{run_id, frame, time_s, type, label}`; re‑treina modelo leve e salva em `runs/model.joblib`.

### `GET /download/{run_id}/{path}`
Serve arquivos e diretórios do run.

---

## 🧰 CLI — Referência (v0.7c2)

Compatível com **hífen** e **sublinhado**:
```
python veg_product_cli.py \
  --input /caminho/video.mp4 \
  --out   runs/teste \
  --every 30 \
  --min-area 6000   (ou --min_area) \
  --agree-k 2       (ou --agree_k) \
  --min-severity 0.9 (ou --min_severity) \
  --disable-soil-guard   # opcional
```

**Saídas do CLI**: `thumbs/`, `occurrences_v2.json`, `report.html`.

---

## 🧠 Modelo leve (aprendizado com rótulos)

- **Features**: `vari, ngrdi, ifv, zmin (se houver), roi_ratio, near_veg_ratio, log1p(area_px), aspect`
- **Treino**: Logistic Regression (scikit-learn); lê `runs/*/labels.csv`, cruza por *frame* com `occurrences_v2.json`.
- **Aplicação**: adiciona `ml.score`, faz **boost** de confiança, marca `|possivel_fp` (score baixo) e promove para `sinal_critico` (score alto + baixo_sinal).

Se não houver `scikit-learn` ou `n<5`, segue só com as regras.

---

## 🧩 Formatos de dados

- **`occurrences_v2.json`**: lista de ocorrências (frame, time_s, bbox, area_px, type, confidence, recommendation, evidence, ml?).
- **`labels.csv`**: `ts,frame,time_s,type,label,bbox,evidence_json` (1 linha por rótulo).

---

## 🩺 Diagnóstico rápido

- **Falhou**: veja `stderr` na resposta e baixe `/download/<run_id>/error.txt`.
- **Sem ocorrências**: ajuste sensibilidade (seção **Controles**).
- **Args não reconhecidos (CLI)**: use o **v0.7c2** (hífen/sublinhado).
- **Codecs**: se vídeo não abrir, reencode para MP4/H.264 + AAC.

---

## 🛣️ Roadmap sugerido

- Classificador por textura/forma (RF/XGBoost) além de médias.
- ROI por talhão (GeoJSON/KML) e métricas por parcela.
- Agregação temporal (persistência entre frames).
- Exportações (GeoTIFF/Shapefile/CSV detalhado).
- Docker + compose; otimizações de performance.

---

## 🧾 Histórico

- **v0.7c2 (CLI)**: flags com hífen **e** sublinhado.
- **v0.7c (API/painel)**: retorno 200 com `ok:false` + `error.txt`; painel com abas e barra de progresso; rótulos reintroduzidos.
- **v0.7b**: API lê `options_json` e repassa flags ao CLI.
- **v0.7**: `/train` treina modelo leve; aplicação no `/analyze`.
- **v0.6d**: “Abrir relatório” prioriza `report.html`; thumbs robustas.
- **v0.6**: base do projeto.

---

## 📜 Licença

Defina sua licença (ex.: MIT).

---

_Atualizado em 2025-09-18 20:23:02_
