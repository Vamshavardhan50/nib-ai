# cactus-rs

A **from-scratch neural network in Rust**. It trains its own models — **no pre-trained weights, no GGUF, no external model files, no mock data**. You feed it real text or real images, it learns real patterns.

Smallest binary, fastest CPU training/inference, easy integration for any developer. Includes **TurboLM**: a 100M-shape language model in a **994,656-byte** file, and **DenseLM**: a true **100,009,068-weight** model stored in **50,128,686 bytes (50.13 MB)** at 4 bits — plus a zero-dependency HTTP API and embeddable web UI.

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
| **DenseLM** | Text generation, true 100M weights, ~50 MB | Any real `.txt` file | `dense-train` / `dense-generate` / `serve` |

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
cd "D:\Smallest language Model"
.\build.ps1 -Release
```

`build.ps1` automatically finds MinGW and uses the GNU toolchain (no Visual Studio needed). If you prefer plain cargo:

```powershell
cargo build --release
```

**Verified:** release binary = **3,025,408 bytes (3.03 MB / 2.88 MiB)**, 25 unit tests + 1 doc test pass, CharLM loss 2.35 → 1.46 (40 epochs), Turbo loss 2.32 → 1.78 (5 epochs), Turbo = **100.01M logical / 994,656-byte file**, DenseLM = **100,009,068 weights / 50,128,686-byte file**, dense loss **1.5755 → 1.5290** (2 epochs), HTTP generate roundtrip + CORS verified, vision loss drops → 0.02, red/blue classification 98%+ confidence.

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
.\target\release\cactus train --input shakespeare.txt --output model.json --tokenizer tokenizer.json --context 12 --hidden 128 --embed 24 --epochs 40
```

Loss should decrease each epoch as it learns your data's patterns.

### Generate

```powershell
.\target\release\cactus generate --model model.json --tokenizer tokenizer.json --seed "ROMEO: " --length 500 --temperature 0.7
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

The file is ~0.99 MB. The number of **independent** weights is ~1M (the stored factors); the 100M figure is the dense-equivalent shape. This is the same idea as LoRA-style low-rank factorization, not a free lunch. Implementation: `src/turbo.rs` (`LowRankLinear`, `TurboLM`, `TurboTrainer`). If you need every one of the 100M weights stored, use the **DenseLM** section below: 4-bit weights in ~50 MB.

| `--rank` | stored params | file size |
|---------:|--------------:|----------:|
| 16 | 0.53M | ~0.53 MB |
| 24 | 0.79M | ~0.79 MB |
| **30** | **0.99M** | **~0.995 MB** |
| 32 | 1.05M | ~1.05 MB (over 1 MB) |

Higher rank means more capacity and a bigger file. Rank 30 is the largest tested rank that keeps the 100M shape under 1,000,000 bytes.

### Inspect the size first

```powershell
.\target\release\cactus turbo-info --target 100000000
```

Prints logical params, stored params, estimated file size, and compression versus a dense f32 model.

### Train

```powershell
# Quick run: 50k windows, 5 epochs (a few minutes on a laptop)
.\target\release\cactus turbo-train --input shakespeare.txt --output turbo.bin --context 64 --embed 128 --rank 30 --epochs 5 --max-samples 50000

# Full 200 KB local corpus, 5 epochs (about 15 minutes)
.\target\release\cactus turbo-train --input shakespeare.txt --output turbo.bin --context 64 --embed 128 --rank 30 --epochs 5

# Larger corpus: cap windows per epoch to keep runtime predictable
.\target\release\cactus turbo-train --input tinyshakespeare.txt --output turbo.bin --context 64 --embed 128 --rank 30 --epochs 10 --max-samples 200000
```

### Generate

```powershell
.\target\release\cactus turbo-generate --model turbo.bin --tokenizer tokenizer.json --seed "ROMEO: " --length 500 --temperature 0.7
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
use cactus_rs::{TurboConfig, TurboLM, TurboTrainer};

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

## Dense 100M model — 100M real weights, 4-bit, ~50 MB

Turbo stores a 100M **shape** in ~1 MB of factors. This variant stores **all 100M weights**:

```text
embedding:      62 × 128              =      7,936
fc1:         8,192 × 12,114           = 99,237,888
fc2:        12,114 × 62               =    751,068
biases:      12,114 + 62              =     12,176
total independent weights             = 100,009,068
```

At **4 bits per weight**, the file is about **50 MB** — well under the 100 MB target. It uses one scale per row, so it stays more accurate than a single global scale.

| Variant | Independent weights | File | Use case |
|---------|--------------------:|-----:|----------|
| **TurboLM** | ~995K | 994,656 bytes | fastest CPU / tiny deploy |
| **DenseLM** | **100,009,068** | **50,128,686 bytes (50.13 MB)** | true 100M-weight model |

### Inspect the size

```powershell
.\target\release\cactus dense-info --target 100000000
```

### Materialize a dense model

This expands a trained Turbo model's factors into the full `U · V` matrices and writes 4-bit weights:

```powershell
.\target\release\cactus dense-build --turbo turbo.bin --output dense.bin --tokenizer tokenizer.json
```

### Fine-tune the 100M dense model

The practical training path is: train the compact model first, expand it, then run **dense backprop** so every one of the 100M weights is updated:

```powershell
.\target\release\cactus dense-train --input shakespeare.txt --output dense.bin --tokenizer tokenizer.json --init-turbo turbo.bin --epochs 2 --max-samples 20000 --batch-size 128 --lr 0.02
```

**Verified run:** 2 epochs × 20,000 windows, batch 128, lr 0.02, SGD: loss **1.5755 → 1.5290**. The final 4-bit artifact is exactly **50,128,686 bytes**. On this CPU, batch 128 trains at ~0.035 s/sample (~12 minutes/epoch); batch 32 is about 3× slower.

You can also train from random weights, but this is much slower on CPU:

```powershell
.\target\release\cactus dense-train --input shakespeare.txt --output dense.bin --tokenizer tokenizer.json --target 100000000 --epochs 1 --max-samples 2000 --batch-size 128 --lr 0.01
```

### Generate

```powershell
.\target\release\cactus dense-generate --model dense.bin --tokenizer tokenizer.json --seed "ROMEO: " --length 300 --temperature 0.7
```

### Serve

`serve` auto-detects the model format, so the same API/UI works for both models:

```powershell
.\target\release\cactus serve --model dense.bin --tokenizer tokenizer.json --port 8080
```

Dense inference is heavier than Turbo: 100M multiply-adds per generated character versus ~1M stored factors. On the development CPU, 40 characters took **7.2 s** through the HTTP API (~180 ms/char), and dense fine-tuning ran at ~**0.035 s/sample** at batch 128. Dense inference needs roughly **400 MB RAM** for the f32 weights; training needs more. For the fastest web demo use `turbo.bin`; use `dense.bin` when you need the real 100M-weight artifact.

> **Prompt length:** both models use a fixed 64-character context. Prompts shorter than that are repeated to fill the context instead of being padded with spaces. Prompts of 64+ characters give the most natural results; the web UI ships with a natural Shakespeare line as its default prompt.

> **100B parameters under 100 MB is still impossible.** 100B at 1 bit needs ~12.5 GB, at 4 bits ~50 GB, and at int8 ~100 GB. 100M under 100 MB is the target this model meets.

---

## Use it anywhere — HTTP API + web UI

The same release binary can serve the model. No web framework, no Node, no Python, and no extra dependencies:

```powershell
.\target\release\cactus serve --model turbo.bin --tokenizer tokenizer.json --host 0.0.0.0 --port 8080
```

Open <http://localhost:8080>. The page has a prompt box, length/temperature controls, copy button, and a ready-to-copy `<iframe>` embed. The server also exposes a JSON API with **CORS enabled**, so any website can call it.

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
const response = await fetch("https://your-cactus-host/api/generate", {
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
    "https://your-cactus-host/api/generate",
    json={"prompt": "ROMEO: ", "length": 300, "temperature": 0.8},
    timeout=60,
)
print(r.json()["text"])
```

Embed the whole demo on any site:

```html
<iframe
  src="https://your-cactus-host/"
  width="100%" height="760"
  style="border:0;border-radius:16px"
  title="cactus-rs language model"></iframe>
```

### Host the UI on GitHub Pages + the model anywhere

GitHub Pages serves static files only — it cannot execute Rust. Use the recommended split: **static UI on GitHub Pages, API on any Rust host**.

```powershell
# 1. Export the same UI that `serve` uses into a static index.html
.\target\release\cactus web-export --output index.html

# 2. Commit the UI and the tracked tokenizer + models (turbo.bin, dense.bin)
git init
git add .
git commit -m "cactus-rs: from-scratch language models with web API"
```

Create a repository on GitHub, then:

```bash
git remote add origin https://github.com/<you>/cactus-rs.git
git branch -M main
git push -u origin main
```

Enable **Settings → Pages → Deploy from a branch → main → / (root)**. Your UI is now at:

```
https://<you>.github.io/cactus-rs/
```

Start the API on Render, Railway, Fly.io, a VPS, or a GitHub Codespace, then point the static UI at it:

```
https://<you>.github.io/cactus-rs/?api=https://your-api-host
```

The endpoint is also editable in the page and saved to `localStorage`, so the same static UI works with any backend.

### Docker (works on Render, Railway, Fly.io, Hugging Face Spaces, any VPS)

The included `Dockerfile` builds a small Debian image with the `cactus` binary, a model, and `tokenizer.json`. Pick the model with the `MODEL` build arg:

```bash
# Turbo: ~1 MB model, tiny memory, fastest CPU generation
docker build -t cactus-turbo .
docker run --rm -p 8080:8080 cactus-turbo

# Dense: true 100M weights, ~50 MB model, needs roughly 0.5 GB RAM
docker build --build-arg MODEL=dense.bin -t cactus-dense .
docker run --rm -p 8080:8080 cactus-dense
```

`dense.bin` is the trained 50.13 MB artifact and is committed alongside `turbo.bin`, so both images build from a fresh clone. If you would rather keep large files out of git, upload `dense.bin` as a release asset and adjust the `COPY` line.

| Platform | What to do |
|----------|------------|
| **Render** | New → Web Service → this repo → Runtime **Docker** → health check `/health` |
| **Railway** | New Project → Deploy from GitHub; the Dockerfile is detected automatically |
| **Fly.io** | `fly launch --now`; internal port `8080`; use a 1 GB+ VM for the dense model |
| **Hugging Face Spaces** | New Space → SDK **Docker**; set variable `PORT=7860` |
| **Any VPS** | Copy `cactus`, the model, `tokenizer.json`; run `./cactus serve --host 0.0.0.0 --port 8080`; put Caddy/nginx in front |

The Turbo model is only ~1 MB and needs a few MB of RAM. The dense model is ~50 MB on disk and ~400 MB of f32 weights in memory. If you prefer not to commit model weights, upload them as GitHub Release assets or mount them into the container and adjust the `COPY` line.

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
.\target\release\cactus vision-train --manifest manifest.txt --output vision.json --classes 2 --image-size 32 --epochs 20
```

### 3. Predict

```powershell
.\target\release\cactus vision-predict --model vision.json --image data/cats/new.jpg --classes classes.txt
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
cactus train          --input <file> --output <model.json> --tokenizer <tok.json>
                      [--context 8] [--hidden 64] [--embed 16] [--lr 0.1]
                      [--epochs 200] [--optimizer sgd]

cactus generate       --model <model.json> --tokenizer <tok.json>
                      [--seed ""] [--length 200] [--temperature 1.0]

cactus vision-train   --manifest <manifest.txt> --output <vision.json>
                      [--image-size 16] [--channels 1] [--classes 10]
                      [--lr 0.01] [--epochs 10] [--batch-size 16]

cactus vision-predict --model <vision.json> --image <path> [--classes <names.txt>]

cactus turbo-info     [--target 100000000] [--context 64] [--embed 128]
                      [--rank 30] [--vocab 62] [--tokenizer <tok.json>]

cactus turbo-train    --input <file> --output <turbo.bin> --tokenizer <tok.json>
                      [--target 100000000] [--context 64] [--embed 128]
                      [--rank 30] [--lr 0.05] [--epochs 5] [--batch-size 32]
                      [--max-samples 0]

cactus turbo-generate --model <turbo.bin> --tokenizer <tok.json>
                      [--seed "ROMEO: "] [--length 500] [--temperature 0.7]

cactus dense-info     [--target 100000000] [--context 64] [--embed 128]
                      [--hidden 0] [--vocab 62] [--tokenizer <tok.json>]

cactus dense-build    --turbo <turbo.bin> --output <dense.bin>
                      [--tokenizer <tok.json>]

cactus dense-train    --input <file> --output <dense.bin> --tokenizer <tok.json>
                      [--init-turbo <turbo.bin>] [--target 100000000]
                      [--context 64] [--embed 128] [--hidden 0] [--lr 0.02]
                      [--epochs 1] [--batch-size 128] [--max-samples 20000]

cactus dense-generate --model <dense.bin> --tokenizer <tok.json>
                      [--seed "ROMEO: "] [--length 300] [--temperature 0.7]

cactus serve          --model <turbo.bin|dense.bin> --tokenizer <tok.json>
                      [--host 0.0.0.0] [--port 8080]

cactus web-export     [--output index.html]

cactus version
```

---

## Use as a Rust library

```toml
[dependencies]
cactus-rs = { path = "." }  # or publish to crates.io
```

### Text

```rust
use cactus_rs::{Tokenizer, CharLM, Trainer};

fn main() -> anyhow::Result<()> {
    let text = std::fs::read_to_string("mydata.txt")?;
    let tok = Tokenizer::from_text(&text);
    let config = cactus_rs::ModelConfig { vocab_size: tok.vocab_size(), ..Default::default() };
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

### Dense 100M model

```rust
use cactus_rs::{dense, Tokenizer};

fn main() -> anyhow::Result<()> {
    let mut model = dense::load_int4("dense.bin")?;
    let tok = Tokenizer::load("tokenizer.json")?;
    let text = dense::generate_text(&mut model, &tok, "ROMEO: ", 120, 0.7);
    println!("{text}");
    Ok(())
}
```

To train instead of load, wrap `model` in `dense::DenseTrainer::new(model, 0.02)` and call `train_step(&batch)`.

### Vision

```rust
use cactus_rs::{load_image, load_manifest, VisionCNN, VisionTrainer};

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
D:\Smallest language Model\
├── Cargo.toml
├── README.md
├── LICENSE               # Apache-2.0
├── Dockerfile            # one-image deploy (Render/Railway/Fly/VPS)
├── .dockerignore
├── .gitignore
├── build.ps1             # one-command build (GNU toolchain + MinGW)
├── get-data.ps1          # download real-world corpora
├── data.txt              # sample real text (train on this or your own)
├── shakespeare.txt       # real-world corpus (via get-data.ps1)
├── index.html            # static UI export (via `cactus web-export`)
├── model.json            # generated after text training
├── tokenizer.json        # generated after text training (tracked for deploys)
├── vision.json           # generated after vision training
├── turbo.bin             # generated compact Turbo model (~1 MB, tracked)
├── dense.bin             # trained dense 100M model (~50 MB, tracked)
└── src/
    ├── main.rs           # CLI (train/generate/turbo-*/dense-*/serve/vision-*)
    ├── lib.rs            # public API
    ├── tensor.rs         # from-scratch tensor math
    ├── layers.rs         # Linear + Embedding (forward/backward)
    ├── optimizer.rs      # SGD / Adam hyperparameters
    ├── model.rs          # CharLM, Trainer, Tokenizer (text)
    ├── turbo.rs          # LowRankLinear, TurboLM, TurboTrainer (compact)
    ├── dense.rs          # dense 100M CharLM + 4-bit save/load + trainer
    ├── web.rs            # zero-dependency HTTP API + embedded browser UI
    └── vision.rs         # Tensor4, Conv2d, MaxPool2d, VisionCNN (images)
```

## License

Apache-2.0.

## Contributing

PRs welcome. Keep the dependency tree small — every crate adds MB and compile time. Target: from-scratch, smallest MB, fastest CPU, real-world data.