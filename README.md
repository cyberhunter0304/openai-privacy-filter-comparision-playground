# 🛡 PII Shield — Live Detection Comparison Tool

A single-file FastAPI demo that runs **3 PII detection tools side-by-side** and compares:
- Latency (ms)
- Entity detection accuracy
- Redacted output quality

---

## Tools Compared

| Tool | Type | Cost | Key Strength |
|------|------|------|-------------|
| **OpenAI Privacy Filter** | Local model (1.5B params) | Free (Apache 2.0) | Context-aware, 128K tokens, no data leaves machine |
| **Azure AI Language PII** | Cloud REST API | Paid (free tier available) | 30+ entity types, enterprise-grade, regex + NLP |
| **Microsoft Presidio** | Local SDK | Free (MIT) | Fully customizable, regex + SpaCy NER, extensible |

---

## Quick Start

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Download spaCy model (for Presidio)
python -m spacy download en_core_web_lg

# 3. Configure credentials
cp example.env .env
# Edit .env — add your Azure AI Language key + endpoint

# 4. Launch
python main.py

# 5. Open browser
open http://localhost:8000
```

> **First run note:** OpenAI Privacy Filter will download ~2.8GB from HuggingFace on first use. Click "Warm Up Local Models" to pre-load before your demo.

---

## File Structure

```
.
├── main.py          ← FastAPI backend (single file, all logic)
├── index.html       ← Frontend UI (served by FastAPI)
├── requirements.txt ← All Python dependencies
├── example.env      ← Template for credentials
└── README.md        ← This file
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the UI |
| `/api/detect` | POST | Run PII detection (JSON body: `{text, tools}`) |
| `/api/warmup` | GET | Pre-load local models |
| `/api/health` | GET | Check tool availability |

### Example curl
```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My name is John Smith and my SSN is 123-45-6789",
    "tools": ["openai_privacy_filter", "presidio", "azure_content_safety"]
  }'
```

---

## Azure AI Language Setup

1. Go to [Azure Portal](https://portal.azure.com) → Create → **Language Service**
2. Under **Keys and Endpoint**, copy Key 1 and the endpoint URL
3. Paste into your `.env`:
   ```
   AZURE_CONTENT_SAFETY_KEY=...
   AZURE_CONTENT_SAFETY_ENDPOINT=https://your-resource.cognitiveservices.azure.com
   ```
4. **Free tier (F0):** 5,000 transactions/month at no cost

> The app calls: `POST {endpoint}/text/analytics/v3.1/entities/recognition/pii`

---

## PII Categories Detected

### OpenAI Privacy Filter (8 categories)
`private_person` · `private_email` · `private_phone` · `private_address` · `private_url` · `private_date` · `account_number` · `secret`

### Azure AI Language (30+ categories)
Person · Email · Phone · Address · CreditCard · SSN · IPAddress · DateTime · Organization · URL · Age · and many more

### Presidio (50+ entity types)
PERSON · EMAIL_ADDRESS · PHONE_NUMBER · CREDIT_CARD · US_SSN · LOCATION · DATE_TIME · URL · IBAN_CODE · CRYPTO · US_PASSPORT · NRP · IP_ADDRESS · and 30+ more

---

## Latency Expectations

| Tool | Cold Start | Warm |
|------|-----------|------|
| OpenAI Privacy Filter | 10–30s (model load) | **50–300ms** |
| Microsoft Presidio | 5–15s (spaCy load) | **20–150ms** |
| Azure AI Language | None | **200–800ms** (network) |

---

## Notes

- OpenAI Privacy Filter and Presidio run **100% locally** — no data sent to any cloud
- Azure AI Language sends text to Microsoft's cloud (check your data residency requirements)
- For GPU acceleration of Privacy Filter, install `torch` with CUDA support and remove `--device cpu` from the pipeline call
- Presidio is highly customizable — add custom recognizers for domain-specific PII (employee IDs, custom formats, etc.)
