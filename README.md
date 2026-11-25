# nib-rs

A **from-scratch neural network in Rust**. It trains its own models — **no pre-trained weights, no GGUF, no external model files, no mock data**. You feed it real text or real images, it learns real patterns.

Smallest binary, fastest CPU training/inference, easy integration for any developer. Includes **TurboLM**: a 100M-shape language model in a **994,656-byte** file, and **DenseLM**: a true **90,004,008-weight** model stored in **45,117,066 bytes (45.12 MB)** at 4 bits — comfortably under 50 MB — plus a zero-dependency HTTP API and embeddable web UI.

**Live:** [Browser UI](https://vamshavardhan50.github.io/nib-ai/) · [Documentation](https://vamshavardhan50.github.io/nib-ai/docs/)

**Full documentation:** [docs/index.html](docs/index.html) · [Getting started](docs/getting-started.html) · [What you can do](docs/use-cases.html) · [HTTP API](docs/http-api.html) · [Models & training](docs/models.html) · [Deployment](docs/deployment.html)

## Introduction: what the model is about

Nib is a **from-scratch character-level language model**. Give it up to 64 characters of text and it predicts the next character. Feed that prediction back in and it writes Shakespeare-style dialogue, code-like text, or whatever pattern its training corpus contains — one character at a time.

It is deliberately **not a chatbot**. There is no attention, no instruction tuning, and no world knowledge. The goal is a complete language-model pipeline you can read end to end, train on a laptop CPU, compress into a small file, and embed in a CLI, a Rust program, an HTTP service, or any website.

Nib ships two text artifacts from the same architecture family:

- **TurboLM** — a low-rank model with a 100M-entry shape stored in **994,656 bytes**. Fast CPU demo.
- **DenseLM** — a true dense model with **90,004,008 independent weights** stored in **45,117,066 bytes (45.12 MB)** at 4 bits. Every weight is real.

| At a glance | |
|-------------|---|
| **Task** | Next-character prediction from a 64-character context |
| **Dense architecture** | Embedding 62×128 → flatten 8,192 → Linear 8,192×10,902 → ReLU → Linear 10,902×62 → softmax |
| **Turbo architecture** | Same MLP shape with low-rank `U·V` factors, rank 30 |
| **Vocabulary** | 62 characters (newline, space, punctuation, upper/lowercase) |
| **Training data** | First 200,000 bytes of Tiny Shakespeare (public domain) |
| **Training** | Hand-written forward/backward passes, cross-entropy, SGD |
| **Runtime** | CPU, single thread, no GPU, no Python |
| **Interfaces** | CLI, Rust crate, HTTP JSON API, browser UI |

**Good at:** continuing text in a learned local style, learning speaker labels and formatting from small corpora, running offline on a CPU, teaching the full training/quantization pipeline, and embedding a tiny model in an app.

**Not built for:** question answering, reasoning, instructions, factual knowledge, long context (64 characters only), chat, tool use, streaming, or GPU-scale training.

## What "from scratch" means here

| Component | Built from scratch? | Notes |
|-----------|---------------------|-------|
| Tensor / matrix math | ✅ Yes | `ndarray` used only for array storage/ops |
| Weight initialization | ✅ Yes | Xavier/Glorot, He, LeCun formulas |
| Linear / Embedding layers | ✅ Yes | Hand-written forward + backprop |
| Conv2d / MaxPool layers | ✅ Yes | Hand-written forward + backprop |
| ReLU / Softmax / Cross-entropy | ✅ Yes | Hand-written |
| Optimizer (SGD) | ✅ Yes | Hand-written update rule |
| Tokenizer | ✅ Yes | Character-level, built from your text |
| Training loop | ✅ Yes | Mini-batch with shuffling |
| Pre-trained model weights | ❌ No | None — starts from random weights |

No external `.gguf` or pre-trained `.bin`, no Hugging Face download. Models begin with random weights and learn from **your real data**. (`turbo.bin` is this project's own compact output format, produced by `turbo-train`.)

## Four models

| Model | Task | Data you provide | Command |
|-------|------|------------------|---------|
| **CharLM** | Text generation | Any real `.txt` file | `train` / `generate` |
| **VisionCNN** | Image classification | Labeled images + manifest | `vision-train` / `vision-predict` |
| **TurboLM** | Text generation, <1 MB | Any real `.txt` file | `turbo-train` / `turbo-generate` / `serve` |
| **DenseLM** | Text generation, true 90M weights, 45 MB | Any real `.txt` file | `dense-train` / `dense-generate` / `serve` |

## What you can do with it — and how

| Problem | Feature | How to use it |
|---------|---------|---------------|
| Continue text offline in a learned style | Turbo / Dense generation | `nib turbo-generate --model turbo.bin --tokenizer tokenizer.json --seed "ROMEO: "` |
| Use every one of 90M dense weights | Dense 4-bit model | `nib dense-generate --model nib.bin --tokenizer tokenizer.json --seed "First Citizen: ..."` |
| Train on your own notes, lyrics, code, or logs | CharLM, Turbo, or Dense training | `nib train --input mydata.txt --output model.json --tokenizer tokenizer.json` |
| Build a next-character predictor / autocomplete | Rust `predict` + `sample` | `dense::load_int4("nib.bin")`, then `model.predict(&ctx)` |
| Add text generation to a website | Static UI + CORS API | `nib web-export`, host `index.html`, open `?api=https://your-api-host` |
| Serve a JSON API to any client | Zero-dependency HTTP server | `nib serve --model turbo.bin --tokenizer tokenizer.json --port 8080` |
| Embed the model in a Rust app | `nib-rs` crate | `nib-rs = { git = "https://github.com/Vamshavardhan50/nib-ai.git" }` |
| Deploy with one artifact | Docker / single binary | `docker build --build-arg MODEL=nib.bin -t nib-dense .` |
| Classify small image sets | VisionCNN | `nib vision-train --manifest manifest.txt --output vision.json` |
| Choose a size before training | Instant size calculators | `nib turbo-info` / `nib dense-info --hidden 10902` |
| Learn how an LM works end to end | Hand-written layers and backprop | Read `src/model.rs`, `src/dense.rs`, `src/turbo.rs`; run `cargo test` |

**What it is not:** a chatbot, an instruction-following assistant, or a source of world knowledge. It is a character-level model trained on real text. It does not answer questions, use tools, stream tokens, or handle long contexts, and it runs on CPU only. Full recipes, real sample output, and caveats: [docs/use-cases.html](docs/use-cases.html).

## Quick start (developer)

### 1. Install Rust

```powershell
# Windows (PowerShell)
irm https://rustup.rs/rs | iex

# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Restart your terminal so `cargo` is on `PATH`.

### 2. Build

```powershell
git clone https://github.com/Vamshavardhan50/nib-ai.git
cd nib-ai
.\build.ps1 -Release
```

`build.ps1` automatically finds MinGW and uses the GNU toolchain (no Visual Studio needed). If you prefer plain cargo:

```powershell
cargo build --release
```

**Verified:** release binary = **3,041,792 bytes (3.04 MB / 2.90 MiB)**, 27 unit tests + 1 doc test pass, CharLM loss 2.35 → 1.46 (40 epochs), Turbo loss 2.32 → 1.78 (5 epochs), Turbo = **100.01M logical / 994,656-byte file**, DenseLM = **90,004,008 weights / 45,117,066-byte file**, 100M dense fine-tune loss **1.5755 → 1.5290** (2 epochs), pruned 90M fine-tune loss **1.5311 → 1.5154, best epoch 1.5079** (3 epochs), HTTP generate roundtrip + CORS verified, vision loss drops → 0.02, red/blue classification 98%+ confidence.

---

## Text model (real data)

### Get real-world data

```powershell
# Downloads Tiny Shakespeare (~1.1 MB of real text) into shakespeare.txt
.\get-data.ps1

# Or use your own file: code, logs, articles, books, chat logs, ...
```

### Train on any real text

```powershell
.\target\release\nib train --input shakespeare.txt --output model.json --tokenizer tokenizer.json --context 12 --hidden 128 --embed 24 --epochs 40
```

Loss should decrease each epoch as it learns your data's patterns.

### Generate

```powershell
.\target\release\nib generate --model model.json --tokenizer tokenizer.json --seed "ROMEO: " --length 500 --temperature 0.7
```

**Verified on real Shakespeare (200 KB, 40 epochs):** loss 2.35 → **1.46**, and generation reproduces play structure with character names (ROMEO, CORIOLANUS, SICINIUS).

---

## Turbo model — 100M-entry shape in under 1 MB

A dense 100M-parameter model needs **~400 MB** at f32, or **~12.5 MB** even at 1 bit. So 100M *independent* parameters in under 1 MB is physically impossible.

Cactus's own Needle 3 is a useful reference: [its docs](https://github.com/cactus-compute/cactus/blob/main/llms.txt) describe a **29–121M parameter** model with **35 MB weights**, and a **sub-1 MB engine**. The sub-1 MB number is the runtime, not the model.

| Precision | 100M weights |
|-----------|-------------:|
| f32 | ~400 MB |
| int8 | ~100 MB |
| 4-bit | ~50 MB |
| 2-bit | ~25 MB |
| 1-bit | ~12.5 MB |

Turbo gets under 1 MB by storing a 100M-entry **shape** in factored form:

```
fc1: (context·embed = 8192) × (hidden = 12114)  ≈ 99.2M logical weights
fc2: (hidden = 12114) × (vocab = 62)            ≈  0.75M logical weights
total logical (dense) parameters                ≈ 100M

stored as:  W = U · V
fc1: U(8192×30) · V(30×12114)
fc2: U(12114×30) · V(30×62)
then quantized to int8 on disk
```

The forward pass is still a normal 2-layer network:

```
flat = embed(context chars)                 # 8192 values
h    = ReLU(flat · U1 · V1 + b1)            # 12114 values
logits = h · U2 · V2 + b2                   # 62 values
```

The file is ~0.99 MB. The number of **independent** weights is ~1M (the stored factors); the 100M figure is the dense-equivalent shape. This is the same idea as LoRA-style low-rank factorization, not a free lunch. Implementation: `src/turbo.rs` (`LowRankLinear`, `TurboLM`, `TurboTrainer`). If you need every weight stored densely, use the **DenseLM** section below: 90M weights in 45.12 MB at 4 bits (the 100M configuration is 50.13 MB).

| `--rank` | stored params | file size |
|---------:|--------------:|----------:|
| 16 | 0.53M | ~0.53 MB |
| 24 | 0.79M | ~0.79 MB |
| **30** | **0.99M** | **~0.995 MB** |
| 32 | 1.05M | ~1.05 MB (over 1 MB) |

Higher rank means more capacity and a bigger file. Rank 30 is the largest tested rank that keeps the 100M shape under 1,000,000 bytes.

### Inspect the size first

```powershell
.\target\release\nib turbo-info --target 100000000
```

Prints logical params, stored params, estimated file size, and compression versus a dense f32 model.

### Train

```powershell
# Quick run: 50k windows, 5 epochs (a few minutes on a laptop)
.\target\release\nib turbo-train --input shakespeare.txt --output turbo.bin --context 64 --embed 128 --rank 30 --epochs 5 --max-samples 50000

# Full 200 KB local corpus, 5 epochs (about 15 minutes)
.\target\release\nib turbo-train --input shakespeare.txt --output turbo.bin --context 64 --embed 128 --rank 30 --epochs 5

# Larger corpus: cap windows per epoch to keep runtime predictable
.\target\release\nib turbo-train --input tinyshakespeare.txt --output turbo.bin --context 64 --embed 128 --rank 30 --epochs 10 --max-samples 200000
```

### Generate

```powershell
.\target\release\nib turbo-generate --model turbo.bin --tokenizer tokenizer.json --seed "ROMEO: " --length 500 --temperature 0.7
```

**Verified:** 199,936 windows per epoch, 5 epochs, loss **2.3209 → 1.7765**. A sample at temperature 0.6:

```text
ROMEO: ::
When with.
O, that's to that the rather the hoak agard.

LADY INNE:
What in the his the sall to the me, to the sees--
```

It learns speaker labels, line breaks, and common Elizabethan words. The tradeoff is capacity: a rank-30 2-layer factorization is much smaller than a dense 100M transformer, so expect simpler patterns, not ChatGPT-quality prose.

### Build it in Rust

```rust
use nib_rs::{TurboConfig, TurboLM, TurboTrainer};

let config = TurboConfig {
    vocab_size: 62,      // from your Tokenizer
    embed_dim: 128,
    hidden_dim: 64,      // placeholder; with_logical_target sets it
    context: 64,
    rank: 30,
};
let model = TurboLM::new(config).with_logical_target(100_000_000);
let mut trainer = TurboTrainer::new(model, 0.05);

// Each item is (last 64 character ids, next character id).
let batch = vec![(vec![0usize; 64], 1usize)];
trainer.train_step(&batch);
trainer.save("turbo.bin")?;
```

---

## Dense 90M model — 90M real weights, 4-bit, under 50 MB

Turbo stores a 100M **shape** in ~1 MB of factors. This variant stores **all of its weights**. The shipped artifact is pruned from a 100M dense model to fit comfortably under 50 MB:

```text
embedding:      62 × 128              =      7,936
fc1:         8,192 × 10,902           = 89,309,184
fc2:        10,902 × 62               =    675,924
biases:      10,902 + 62              =     10,964
total independent weights             = 90,004,008
```

At **4 bits per weight**, the file is **45,117,066 bytes (45.12 MB)**. It uses one f32 scale per row, so it stays more accurate than a single global scale.

| Variant | Independent weights | File | Use case |
|---------|--------------------:|-----:|----------|
| **TurboLM** | ~995K stored | 994,656 bytes | fastest CPU / tiny deploy |
| **DenseLM (shipped)** | **90,004,008** | **45,117,066 bytes (45.12 MB)** | true 90M-weight model, under 50 MB |
| **DenseLM (100M intermediate)** | 100,009,068 | 50,128,686 bytes (50.13 MB) | full-size model before pruning |

### Inspect the size

```powershell
# Shipped 90M configuration
.\target\release\nib dense-info --hidden 10902 --tokenizer tokenizer.json

# 100M configuration
.\target\release\nib dense-info --target 100000000 --tokenizer tokenizer.json
```

### Materialize, fine-tune, and prune

The practical training path is: train the compact model first, expand it, fine-tune all 100M weights, then prune to 90M and fine-tune the sub-network:

```powershell
# 1. Expand Turbo factors into full dense matrices
.\target\release\nib dense-build --turbo turbo.bin --output dense-100m.bin --tokenizer tokenizer.json

# 2. Dense backprop updates all 100M weights
.\target\release\nib dense-train --input shakespeare.txt --output dense-100m.bin --tokenizer tokenizer.json --init-turbo turbo.bin --epochs 2 --max-samples 20000 --batch-size 128 --lr 0.02

# 3. Prune 12,114 -> 10,902 hidden units and fine-tune the 90M sub-network
.\target\release\nib dense-train --input shakespeare.txt --output nib.bin --tokenizer tokenizer.json --init-dense dense-100m.bin --hidden 10902 --epochs 3 --max-samples 20000 --batch-size 128 --lr 0.01
```

**Verified runs:**
- **100M:** 2 epochs × 20,000 windows, batch 128, lr 0.02, SGD: loss **1.5755 → 1.5290**. Artifact: exactly **50,128,686 bytes**.
- **90M:** pruned hidden 12,114 → 10,902, then 3 epochs × 20,000 windows, batch 128, lr 0.01: loss **1.5311 → 1.5154** (best epoch **1.5079**). Artifact: exactly **45,117,066 bytes**, SHA-256 `59B690608138DDECD596752DA5665C7718D0E8F18B5FF8A9B706E035D7CBE165`.

On this CPU, batch 128 trains at ~0.035 s/sample (~12 minutes/epoch); batch 32 is about 3× slower. `--hidden` with `--init-dense` ranks hidden units by the L2 norm of their outgoing `fc2` row, keeps the strongest units, and fine-tunes the resulting fully dense sub-network.

You can also train from random weights, but this is much slower on CPU:

```powershell
.\target\release\nib dense-train --input shakespeare.txt --output nib.bin --tokenizer tokenizer.json --target 100000000 --epochs 1 --max-samples 2000 --batch-size 128 --lr 0.01
```

### Generate

```powershell
.\target\release\nib dense-generate --model nib.bin --tokenizer tokenizer.json --seed "First Citizen: Before we proceed any further, hear me speak." --length 300 --temperature 0.7
```

### Serve

`serve` auto-detects the model format, so the same API/UI works for both models:

```powershell
.\target\release\nib serve --model nib.bin --tokenizer tokenizer.json --port 8080
```

Dense inference is heavier than Turbo: 90M multiply-adds per generated character versus ~1M stored factors. On the development CPU, a 20-character HTTP request took about **1.7 s** and 120 characters about **11 s** in CLI sampling (~90 ms/char). Turbo generated 200 characters in **150 ms** through the same API. Dense fine-tuning ran at ~**0.035 s/sample** at batch 128. Dense inference needs roughly **400 MB RAM** for the f32 weights; training needs more. For the fastest web demo use `turbo.bin`; use `nib.bin` when you need the real 90M-weight artifact.

> **Prompt length:** both models use a fixed 64-character context. Prompts shorter than that are repeated to fill the context instead of being padded with spaces. Prompts of 64+ characters give the most natural results; the web UI ships with a natural Shakespeare line as its default prompt.

> **100B parameters under 100 MB is still impossible.** 100B at 1 bit needs ~12.5 GB, at 4 bits ~50 GB, and at int8 ~100 GB. 90M under 50 MB is the target this model meets.

---

## Use it anywhere — HTTP API + web UI

The same release binary can serve the model. No web framework, no Node, no Python, and no extra dependencies:

```powershell
.\target\release\nib serve --model turbo.bin --tokenizer tokenizer.json --host 0.0.0.0 --port 8080
```

Open <http://localhost:8080>. The page has a prompt box, length/temperature controls, copy button, and a ready-to-copy `<iframe>` embed. The server also exposes a JSON API with **CORS enabled**, so any website can call it.

> **No public API is hosted for you.** The live UI and docs on GitHub Pages are static; they cannot run Rust. Until you start `nib serve` locally or deploy it, the only endpoint is `http://127.0.0.1:8080` on your own machine.

### Where is the API endpoint?

| Where the server runs | Base URL | Who can reach it |
|-----------------------|----------|------------------|
| Your machine (local only) | `http://127.0.0.1:8080` | Only that machine |
| Your machine (LAN / Docker) | `http://192.168.1.20:8080` | Same network |
| VPS or cloud service | `https://api.example.com` | Internet |
| Temporary tunnel | `https://random-name.trycloudflare.com` | Internet while it runs |

Generation is always `POST {base}/api/generate`.

### Get a public endpoint

```powershell
# 1. Build once
.\build.ps1 -Release

# 2. Copy these three files to the host:
#    nib (or nib.exe), nib.bin or turbo.bin, tokenizer.json

# 3. Run on all interfaces
./nib serve --model nib.bin --tokenizer tokenizer.json --host 0.0.0.0 --port 8080

# 4. Put HTTPS in front (Caddy example) or use the platform's HTTPS URL
# api.example.com {
#     reverse_proxy 127.0.0.1:8080
# }
```

Test from outside your network, then connect the static UI:

```text
https://vamshavardhan50.github.io/nib-ai/?api=https://api.example.com
```

For a quick demo without deploying, expose your local server through a tunnel:

```powershell
# Terminal 1
.\target\release\nib serve --model turbo.bin --tokenizer tokenizer.json --host 127.0.0.1 --port 8080

# Terminal 2 (prints an HTTPS URL)
cloudflared tunnel --url http://127.0.0.1:8080
# or: ngrok http 8080
```

The laptop must stay awake and the URL changes each run. Do not use a tunnel for production. Full endpoint walkthrough, host comparison, UI resolution order, and troubleshooting: [docs/http-api.html#endpoint](docs/http-api.html#endpoint).

### API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Browser UI (single embedded HTML file) |
| `GET` | `/health` | Health check for Docker/Render/Railway/Fly |
| `GET` | `/api/info` | Parameter counts, config, model file size |
| `POST` | `/api/generate` | Generate text from a prompt |
| `GET` | `/openapi.json` | OpenAPI 3.1 schema for tooling |
| `OPTIONS` | any | CORS preflight |

```bash
# Bash / macOS / Linux
curl -s http://localhost:8080/health
curl -s http://localhost:8080/api/info

curl -s http://localhost:8080/api/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"ROMEO: ","length":300,"temperature":0.8}'
```

```powershell
# PowerShell
$body = @{ prompt = "ROMEO: "; length = 300; temperature = 0.8 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8080/api/generate `
  -ContentType "application/json" -Body $body
```

Response:

```json
{
  "text": "ROMEO: ...",
  "prompt": "ROMEO: ",
  "length": 300,
  "temperature": 0.8,
  "ms": 42
}
```

### Call it from any website

```js
const response = await fetch("https://your-nib-host/api/generate", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ prompt: "ROMEO: ", length: 300, temperature: 0.8 })
});
const { text } = await response.json();
console.log(text);
```

```python
import requests

r = requests.post(
    "https://your-nib-host/api/generate",
    json={"prompt": "ROMEO: ", "length": 300, "temperature": 0.8},
    timeout=60,
)
print(r.json()["text"])
```

Embed the whole demo on any site. If the API host also serves the UI:

```html
<iframe
  src="https://your-api-host/"
  width="100%" height="760"
  style="border:0;border-radius:16px"
  title="nib-rs language model"></iframe>
```

If the UI is the static GitHub Pages copy, pass the API host explicitly:

```html
<iframe
  src="https://vamshavardhan50.github.io/nib-ai/?api=https://your-api-host"
  width="100%" height="760"
  style="border:0;border-radius:16px"
  title="nib-rs language model"></iframe>
```

### Host the UI on GitHub Pages + the model anywhere

GitHub Pages serves static files only — it cannot execute Rust. Use the recommended split: **static UI on GitHub Pages, API on any Rust host**.

```powershell
# 1. Export the same UI that `serve` uses into a static index.html
.\target\release\nib web-export --output index.html

# 2. Commit the UI and the tracked tokenizer + models (turbo.bin, nib.bin)
git init
git add .
git commit -m "nib-rs: from-scratch language models with web API"
```

Create a repository on GitHub, then:

```bash
git remote add origin https://github.com/Vamshavardhan50/nib-ai.git
git branch -M main
git push -u origin main
```

Enable **Settings → Pages → Deploy from a branch → main → / (root)**. Your UI and docs are now at:

```
https://vamshavardhan50.github.io/nib-ai/
https://vamshavardhan50.github.io/nib-ai/docs/
```

Start the API on Render, Railway, Fly.io, a VPS, or a GitHub Codespace, then point the static UI at it:

```
https://vamshavardhan50.github.io/nib-ai/?api=https://your-api-host
```

The endpoint is also editable in the page and saved to `localStorage`, so the same static UI works with any backend.

### Docker (works on Render, Railway, Fly.io, Hugging Face Spaces, any VPS)

The included `Dockerfile` builds a small Debian image with the `nib` binary, a model, and `tokenizer.json`. Pick the model with the `MODEL` build arg:

```bash
# Turbo: ~1 MB model, tiny memory, fastest CPU generation
docker build -t nib-turbo .
docker run --rm -p 8080:8080 nib-turbo

# Dense: true 90M weights, 45 MB model, needs roughly 0.5-1 GB RAM
docker build --build-arg MODEL=nib.bin -t nib-dense .
docker run --rm -p 8080:8080 --memory=1g nib-dense
```

`nib.bin` is the trained 45.12 MB artifact and is committed alongside `turbo.bin`, so both images build from a fresh clone. If you would rather keep large files out of git, upload `nib.bin` as a release asset and adjust the `COPY` line.

| Platform | What to do |
|----------|------------|
| **Render** | New → Web Service → this repo → Runtime **Docker** → health check `/health` |
| **Railway** | New Project → Deploy from GitHub; the Dockerfile is detected automatically |
| **Fly.io** | `fly launch --now`; internal port `8080`; use a 1 GB+ VM for the dense model |
| **Hugging Face Spaces** | New Space → SDK **Docker**; set variable `PORT=7860` |
| **Any VPS** | Copy `nib`, the model, `tokenizer.json`; run `./nib serve --host 0.0.0.0 --port 8080`; put Caddy/nginx in front |

The Turbo model is only ~1 MB and needs a few MB of RAM. The dense model is 45.12 MB on disk and ~400 MB of f32 weights in memory; allow 1 GB in containers. If you prefer not to commit model weights, upload them as GitHub Release assets or mount them into the container and adjust the `COPY` line.

> **Public deployment:** the demo API has no authentication (so it can be embedded anywhere). Put it behind a reverse proxy with auth/rate limiting if you need a private endpoint. Request bodies are capped at 64 KB, prompts at 4,096 characters, and generation at 2,000 characters.

---

## Vision model (real images)

### 1. Prepare a manifest

Create `manifest.txt` with one `image_path label` pair per line:

```
data/cats/001.jpg 0
data/cats/002.jpg 0
data/dogs/001.jpg 1
data/dogs/002.jpg 1
```

- `label` is a class index starting at `0`.
- Paths may contain spaces (label is the last token).
- Lines starting with `#` are ignored.

### 2. Train

```powershell
.\target\release\nib vision-train --manifest manifest.txt --output vision.json --classes 2 --image-size 32 --epochs 20
```

### 3. Predict

```powershell
.\target\release\nib vision-predict --model vision.json --image data/cats/new.jpg --classes classes.txt
```

`classes.txt` is optional — one class name per line (line 0 = class 0, etc.).

### Vision architecture (all hand-written)

```
input image (C × H × W, normalized to [0,1])
   │
   ▼
Conv2d(C → 8, 3×3) → ReLU → MaxPool(2×2)
   │
   ▼
Conv2d(8 → 16, 3×3) → ReLU → MaxPool(2×2)
   │
   ▼
Flatten → Linear(features → num_classes) → Softmax
```

Backpropagation is implemented by hand through every conv, pool, relu, and linear layer.

---

## CLI reference

```
nib train          --input <file> --output <model.json> --tokenizer <tok.json>
                      [--context 8] [--hidden 64] [--embed 16] [--lr 0.1]
                      [--epochs 200] [--optimizer sgd]

nib generate       --model <model.json> --tokenizer <tok.json>
                      [--seed ""] [--length 200] [--temperature 1.0]

nib vision-train   --manifest <manifest.txt> --output <vision.json>
                      [--image-size 16] [--channels 1] [--classes 10]
                      [--lr 0.01] [--epochs 10] [--batch-size 16]

nib vision-predict --model <vision.json> --image <path> [--classes <names.txt>]

nib turbo-info     [--target 100000000] [--context 64] [--embed 128]
                      [--rank 30] [--vocab 62] [--tokenizer <tok.json>]

nib turbo-train    --input <file> --output <turbo.bin> --tokenizer <tok.json>
                      [--target 100000000] [--context 64] [--embed 128]
                      [--rank 30] [--lr 0.05] [--epochs 5] [--batch-size 32]
                      [--max-samples 0]

nib turbo-generate --model <turbo.bin> --tokenizer <tok.json>
                      [--seed "ROMEO: "] [--length 500] [--temperature 0.7]

nib dense-info     [--target 100000000] [--context 64] [--embed 128]
                      [--hidden 0] [--vocab 62] [--tokenizer <tok.json>]

nib dense-build    --turbo <turbo.bin> --output <nib.bin>
                      [--tokenizer <tok.json>]

nib dense-train    --input <file> --output <nib.bin> --tokenizer <tok.json>
                      [--init-turbo <turbo.bin>] [--init-dense <nib.bin>]
                      [--hidden 0] [--target 100000000] [--context 64]
                      [--embed 128] [--lr 0.02] [--epochs 1]
                      [--batch-size 128] [--max-samples 20000]

nib dense-generate --model <nib.bin> --tokenizer <tok.json>
                      [--seed "ROMEO: "] [--length 300] [--temperature 0.7]

nib serve          --model <turbo.bin|nib.bin> --tokenizer <tok.json>
                      [--host 0.0.0.0] [--port 8080]

nib web-export     [--output index.html]

nib version
```

---

## Use as a Rust library

```toml
[dependencies]
nib-rs = { path = "." }  # or publish to crates.io
```

### Text

```rust
use nib_rs::{Tokenizer, CharLM, Trainer};

fn main() -> anyhow::Result<()> {
    let text = std::fs::read_to_string("mydata.txt")?;
    let tok = Tokenizer::from_text(&text);
    let config = nib_rs::ModelConfig { vocab_size: tok.vocab_size(), ..Default::default() };
    let model = CharLM::new(config);
    let mut trainer = Trainer::new(model, "sgd".into(), 0.1);

    let ids = tok.encode(&text);
    let ctx = trainer.model.config.context;
    let batches: Vec<(Vec<usize>, usize)> = (0..ids.len() - ctx)
        .map(|i| (ids[i..i + ctx].to_vec(), ids[i + ctx]))
        .collect();

    for _ in 0..100 {
        println!("loss: {:.4}", trainer.train_step(&batches));
    }
    trainer.save("model.json")?;
    tok.save("tokenizer.json")?;
    Ok(())
}
```

### Dense 90M model

```rust
use nib_rs::{dense, Tokenizer};

fn main() -> anyhow::Result<()> {
    let mut model = dense::load_int4("nib.bin")?;
    let tok = Tokenizer::load("tokenizer.json")?;
    let text = dense::generate_text(&mut model, &tok, "ROMEO: ", 120, 0.7);
    println!("{text}");
    Ok(())
}
```

To train instead of load, wrap `model` in `dense::DenseTrainer::new(model, 0.02)` and call `train_step(&batch)`.

### Vision

```rust
use nib_rs::{load_image, load_manifest, VisionCNN, VisionTrainer};

fn main() -> anyhow::Result<()> {
    let entries = load_manifest("manifest.txt")?;
    let images: Vec<_> = entries.iter()
        .map(|(p, _)| load_image(p, 32, 32, 1))
        .collect::<Result<_, _>>()?;
    let labels: Vec<usize> = entries.iter().map(|(_, l)| *l).collect();

    let mut trainer = VisionTrainer::new(VisionCNN::new(1, 32, 32, 2), 0.01);
    for _ in 0..20 {
        println!("loss: {:.4}", trainer.train_step(&images, &labels));
    }
    trainer.save("vision.json")?;
    Ok(())
}
```

---

## Real-world data tips

| Goal | Recommendation |
|------|----------------|
| Faster text training | Smaller `--hidden 32`, `--embed 8`, `--context 4` |
| Better text quality | More data (MBs), more `--epochs`, larger `--context` |
| Less repetitive text | Lower `--temperature` (0.7–0.9) |
| More creative text | Higher `--temperature` (1.1–1.3) |
| Faster vision training | Smaller `--image-size` (16), grayscale (`--channels 1`) |
| Better vision accuracy | More images per class, more `--epochs`, RGB (`--channels 3`) |

## Release optimizations (already applied)

In `Cargo.toml` `[profile.release]`:

| Setting | Effect |
|---------|--------|
| `opt-level = 3` | fastest code |
| `lto = true` | cross-crate optimization (smaller + faster) |
| `codegen-units = 1` | better optimization |
| `panic = "abort"` | no unwind tables → smaller binary |
| `strip = true` | remove debug symbols |

## File layout (flat — no subfolders except `src/`)

```
nib-ai/
├── Cargo.toml
├── README.md
├── LICENSE               # Apache-2.0
├── Dockerfile            # one-image deploy (Render/Railway/Fly/VPS)
├── .dockerignore
├── .gitignore
├── build.ps1             # one-command build (GNU toolchain + MinGW)
├── get-data.ps1          # download real-world corpora
├── data.txt              # sample real text (train on this or your own)
├── shakespeare.txt       # real-world corpus (via get-data.ps1, ignored)
├── index.html            # static UI export (via `nib web-export`)
├── tokenizer.json        # 62-character vocabulary (tracked for deploys)
├── turbo.bin             # compact Turbo model, 994,656 bytes (tracked)
├── nib.bin               # dense 90M model, 45,117,066 bytes (tracked)
├── docs/                 # detailed static documentation site
│   ├── index.html        # overview
│   ├── getting-started.html
│   ├── cli.html
│   ├── http-api.html
│   ├── rust-api.html
│   ├── models.html
│   ├── deployment.html
│   ├── faq.html
│   ├── style.css
│   ├── docs.js
│   └── build.mjs         # regenerates the pages from one layout
└── src/
    ├── main.rs           # CLI (train/generate/turbo-*/dense-*/serve/vision-*)
    ├── lib.rs            # public API
    ├── tensor.rs         # from-scratch tensor math
    ├── layers.rs         # Linear + Embedding (forward/backward)
    ├── optimizer.rs      # SGD / Adam hyperparameters
    ├── model.rs          # CharLM, Trainer, Tokenizer (text)
    ├── turbo.rs          # LowRankLinear, TurboLM, TurboTrainer (compact)
    ├── dense.rs          # dense 90M CharLM + pruning + 4-bit save/load + trainer
    ├── web.rs            # zero-dependency HTTP API + embedded browser UI
    └── vision.rs         # Tensor4, Conv2d, MaxPool2d, VisionCNN (images)
```

## License

Apache-2.0.

## Contributing

PRs welcome. Keep the dependency tree small — every crate adds MB and compile time. Target: from-scratch, smallest MB, fastest CPU, real-world data.