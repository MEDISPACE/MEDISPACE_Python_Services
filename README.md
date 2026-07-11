# MediSpace Python Services

Python microservices for MediSpace AI and machine-learning workloads. This monorepo contains the prescription OCR service, product recommendation service, and AI pharmacy assistant service used by the MediSpace e-commerce backend.

These services keep compute-heavy OCR, LLM, RAG, guardrail, and recommendation logic outside the Node.js API while exposing simple FastAPI contracts to the rest of the platform.

## Services

```text
MEDISPACE_Python_Services/
├── MEDISPACE_OCR_Service/       # Port 8001 - Prescription OCR and extraction
├── MEDISPACE_ML_Service/        # Port 8002 - Product recommendation engine
└── MEDISPACE_Chat_AI_Service/   # Port 8003 - AI pharmacy assistant and article AI
```

## Architecture

```text
MediSpace Backend
      |
      +-- POST /api/ocr/extract-prescription  ---> OCR Service
      |        PaddleOCR + VietOCR + Regex + Vision LLM + quality merge
      |
      +-- /recommend/*                         ---> ML Service
      |        TF-IDF/MMR + FP-Growth + NMF + SVD + hybrid fallback
      |
      +-- /chat, /chat/stream, /article/*       ---> Chat AI Service
               Gemma via custom LLM API + Typesense RAG + AI guardrails
```

## Authors

- Tran Nguyen Quoc Bao
- Nguyen Huu Thong

## 1. OCR Service

Path: `MEDISPACE_OCR_Service`

Default port: `8001`

Main endpoint: `POST /api/ocr/extract-prescription`

The OCR service extracts structured prescription information from JPG, PNG, and WEBP images.

Current pipeline:

1. Validate uploaded image format.
2. Analyze image quality signals such as blur, brightness, contrast, resolution, and aspect ratio.
3. Traditional branch: PaddleOCR detects text regions, VietOCR reads Vietnamese text, then regex/LLM extraction converts raw text into structured JSON.
4. Vision branch: sends image input to the configured custom LLM API for direct prescription extraction.
5. Parallel mode: runs traditional and vision extraction together, scores both candidates, applies early-return safety rules, and merges the best result.
6. Returns extracted prescription data, raw text, quality score, flags, timing, and optional candidate details.

Supported modes:

| Mode | Purpose |
| --- | --- |
| `traditional` | PaddleOCR + VietOCR + regex/LLM extraction |
| `vision` | Vision LLM extraction from image |
| `parallel` | Production-oriented merge of traditional and vision candidates |
| `parallel_benchmark` | Debug/benchmark mode with candidate detail |

Important OCR configuration:

| Variable | Purpose |
| --- | --- |
| `EXTRACTOR_BACKEND` | Extractor backend, currently expected to use `custom` |
| `CUSTOM_LLM_BASE_URL` | Self-hosted/OpenAI-compatible LLM endpoint |
| `CUSTOM_LLM_MODEL` | Model name, for example `gemma-4-e4b-it.gguf` |
| `VISION_EXTRACTION_STRATEGY` | Vision strategy such as `two_stage` |
| `PRESCRIPTION_OCR_MODE` | Backend-selected mode, commonly `parallel` |
| `TRADITIONAL_EARLY_RETURN_MIN_SCORE` | Minimum score before traditional branch can return early |
| `TRADITIONAL_EARLY_RETURN_MIN_MEDICATIONS` | Minimum medication count before traditional branch can return early |
| `PADDLE_OCR_MIN_CONFIDENCE` | Minimum OCR confidence threshold |
| `OCR_INCLUDE_CANDIDATES` | Include candidate branch details for debugging |

## 2. ML Recommendation Service

Path: `MEDISPACE_ML_Service`

Default port: `8002`

Main endpoints: `/recommend/*`

The ML service trains and serves product recommendations for the storefront, checkout, pharmacist tools, and account experiences.

Algorithms currently implemented:

| Algorithm | Role in MediSpace |
| --- | --- |
| TF-IDF + cosine similarity | Finds related products by product name, ingredients, indications, category, brand, dosage form, and description |
| MMR diversification | Reduces repetitive recommendations by balancing relevance and category/content diversity |
| FP-Growth | Learns products frequently bought together from delivered order transactions |
| NMF trending | Ranks trending/popular products using interaction and rating signals |
| SVD personalization | Predicts products a specific user may prefer from historical user-product behavior |
| Hybrid engine | Trains all models, chooses the best available algorithm, handles fallback, and exposes model metrics |

Key endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health and model readiness |
| `GET /metrics` | Model and serving metrics, protected by service token |
| `POST /train` | Trigger retraining in background |
| `GET /recommend/related/{product_id}` | Related products via TF-IDF/MMR |
| `GET /recommend/bought-together/{product_id}` | Bought-together products via FP-Growth with TF-IDF fallback |
| `GET /recommend/trending` | Trending products via NMF |
| `GET /recommend/popular` | Popular products from transaction/interaction signals |
| `GET /recommend/for-you/{user_id}` | Personalized recommendations via SVD with fallback |
| `POST /recommend/post-purchase` | Hybrid recommendations after checkout/order |
| `POST /recommend/pharmacist` | Pharmacist support suggestions using medical context and allergy filtering |
| `GET /recommend/replenishment/{user_id}` | Repurchase/replenishment suggestions by purchase interval |

Important ML configuration:

| Variable | Purpose |
| --- | --- |
| `MONGODB_URI`, `MONGODB_DB_NAME` | Source data for products, orders, reviews, interactions |
| `ML_SERVICE_TOKEN` | Internal token required by backend and protected ML endpoints |
| `BE_SERVICE_URLS` | Backend webhook targets to flush recommendation cache after retraining |
| `ML_RETRAIN_INTERVAL_HOURS` | Scheduled retraining interval |
| `SVD_MIN_USERS` | Minimum user count before SVD personalization is useful |
| `FPGROWTH_MIN_TRANSACTIONS` | Minimum transaction count before FP-Growth rules are trained |
| `FPGROWTH_MIN_SUPPORT`, `FPGROWTH_MIN_CONFIDENCE` | Association-rule thresholds |

## 3. Chat AI Service

Path: `MEDISPACE_Chat_AI_Service`

Default port: `8003`

Main endpoints: `/chat`, `/chat/stream`, `/article/assist`, `/article/ask`

The Chat AI service provides an AI pharmacy assistant and article assistant. It uses a custom OpenAI-compatible LLM endpoint, currently configured for Gemma via a self-hosted LLM server, and enriches answers with Typesense search results when RAG is available.

Current capabilities:

- Intent-aware chat classification for general, product search, order tracking, loyalty, coupon, return, prescription status, prescription request, emergency, mental-health crisis, and too-long messages.
- RAG context from Typesense products and articles when the intent benefits from internal MediSpace knowledge.
- Medical guardrails through pre-filtering, prompt rules, and post-filter sanitization.
- Streaming responses through Server-Sent Events at `/chat/stream`.
- Optional image URL support for chat messages with prescription/product context.
- Article assistant actions for outline, SEO, excerpt, FAQ, quality check, and source suggestions.
- Article question answering for HealthHub article detail pages.

Key endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check, model info, RAG status, supported intents, vision contract status |
| `POST /chat` | Non-streaming AI pharmacy response |
| `POST /chat/stream` | SSE streaming AI pharmacy response |
| `POST /article/assist` | AI authoring support for health articles |
| `POST /article/ask` | Answer reader questions about a specific article |

Important Chat AI configuration:

| Variable | Purpose |
| --- | --- |
| `CUSTOM_LLM_BASE_URL` | Custom LLM API base URL, for example a Llama.cpp/OpenAI-compatible server |
| `CUSTOM_LLM_MODEL` | Model name, for example `gemma-4-e4b-it.gguf` |
| `CUSTOM_LLM_API_KEY` | Optional bearer token if the LLM endpoint requires authentication |
| `CUSTOM_LLM_MAX_TOKENS` | Output limit for LLM responses |
| `TYPESENSE_URL`, `TYPESENSE_API_KEY` | RAG search backend configuration |
| `TYPESENSE_PRODUCTS_COLLECTION` | Product collection used for product RAG |
| `DB_PATIENT_MEDICAL_COLLECTION` | Patient medical context collection when enabled |
| `LOG_LEVEL` | Service logging level |

## Self-Hosted Gemma / Custom LLM API

OCR and Chat AI are configured to call a custom LLM endpoint instead of relying directly on a public AI API from application code.

Typical configuration:

```env
EXTRACTOR_BACKEND=custom
CUSTOM_LLM_BASE_URL=https://llm.example.com
CUSTOM_LLM_MODEL=gemma-4-e4b-it.gguf
CUSTOM_LLM_API_KEY=
CUSTOM_LLM_MAX_TOKENS=4096
```

In the current architecture, the LLM server is expected to expose an OpenAI-compatible API. Llama.cpp can be used to host a `.gguf` Gemma model and serve inference behind this API. This approach helps MediSpace reduce dependency on third-party AI APIs and gives the team more control over data flow, cost, and infrastructure policy. It still requires normal production controls such as access restrictions, secret management, logging policy, monitoring, and medical guardrails.

## Local Development

Install dependencies per service:

```bash
cd MEDISPACE_OCR_Service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

```bash
cd MEDISPACE_ML_Service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8002
```

```bash
cd MEDISPACE_Chat_AI_Service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8003
```

The full Docker-based local stack can also be started from the backend repository with `docker-compose.dev.yml`.

## Docker

Each service has its own Dockerfile:

```text
MEDISPACE_OCR_Service/Dockerfile
MEDISPACE_ML_Service/Dockerfile
MEDISPACE_Chat_AI_Service/Dockerfile
```

In production, these images are normally orchestrated by the backend repository's `docker-compose.yml` together with the Node.js backend, frontend, Redis, and Typesense.

## Health Checks

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

The ML health endpoint returns `503` until models are trained and ready. The Chat AI health endpoint also reports Typesense/RAG availability and the current LLM model configuration.

## Current Integration Status

- OCR service supports traditional, vision, parallel, and benchmark extraction modes.
- OCR parallel mode includes candidate scoring, early-return safeguards, image-quality flags, and merged output.
- ML service trains TF-IDF, FP-Growth, NMF, and SVD models on startup, then retrains on a configured interval.
- ML service protects internal endpoints with `ML_SERVICE_TOKEN` and notifies backend instances to flush recommendation cache after retraining.
- Chat AI service supports non-streaming and streaming chat, RAG with Typesense, guardrails, image URL contract support, and article assistant endpoints.
- All three services are intended to run behind the Node.js backend rather than being called directly by the browser.

## Safety and Limitations

- AI and OCR outputs are assistive and must remain reviewable, especially for prescriptions, prescription-only medicines, urgent symptoms, and patient-specific dosing.
- OCR accuracy depends on image quality, handwriting/print clarity, angle, lighting, and model availability.
- RAG quality depends on Typesense index freshness and the quality of product/article data.
- Recommendation quality depends on product catalog quality and available behavioral/order history.
- Self-hosted LLM operation still requires infrastructure monitoring, timeout tuning, secret handling, access control, and fallback behavior.
