// Generate the static nib-rs documentation site.
// Usage: node docs/build.mjs
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const docsDir = dirname(fileURLToPath(import.meta.url));
const repo = "https://github.com/Vamshavardhan50/nib-ai";
const site = "https://vamshavardhan50.github.io/nib-ai/docs";
const version = "0.1.0";

const pages = [
  {
    slug: "index",
    title: "Overview",
    description:
      "Nib is a from-scratch Rust language-model toolkit: a real 90M-weight model in a 45 MB file, a compact 1 MB model, a zero-dependency HTTP API, and an embeddable UI.",
    body: `
<section class="hero">
  <p class="eyebrow">From-scratch language models in Rust</p>
  <h1>A real 90M-weight model in a 45&nbsp;MB file.</h1>
  <p class="lead">
    Nib trains character-level language models from random weights, stores every weight at
    4&nbsp;bits, and serves them through one tiny HTTP API. No pre-trained checkpoints, no GGUF,
    no Python stack, no mock data.
  </p>
  <div class="hero-actions">
    <a class="btn" href="getting-started.html">Get started in 5 minutes</a>
    <a class="btn secondary" href="http-api.html">HTTP API reference</a>
    <a class="btn secondary" href="${repo}" rel="noopener">View on GitHub</a>
  </div>
  <div class="stats">
    <div class="stat"><b>90,004,008</b><span>independent dense weights</span></div>
    <div class="stat"><b>45,117,066</b><span>bytes on disk · 4-bit</span></div>
    <div class="stat"><b>994,656</b><span>bytes for the compact Turbo model</span></div>
    <div class="stat"><b>~3 MB</b><span>stripped release binary</span></div>
  </div>
</section>

<h2 id="introduction">Introduction: what the model is about</h2>
<p>
  Nib is a <strong>from-scratch character-level language model</strong>. Give it up to 64
  characters of text and it predicts the next character. Feed that prediction back in and it
  writes Shakespeare-style dialogue, code-like text, or whatever pattern its training corpus
  contains — one character at a time.
</p>
<p>
  It is deliberately <strong>not a chatbot</strong> and not a general-purpose assistant. There
  is no attention, no instruction tuning, and no world knowledge. The point is a complete
  language-model pipeline you can read end to end, train on a laptop CPU, compress into a small
  file, and embed in a CLI, a Rust program, an HTTP service, or any website.
</p>
<p>
  Nib ships two text artifacts from the same architecture family. <strong>TurboLM</strong> is a
  low-rank model with a 100M-entry shape stored in <strong>994,656 bytes</strong>.
  <strong>DenseLM</strong> is a true dense model with <strong>90,004,008 independent
  weights</strong> stored in <strong>45,117,066 bytes</strong> at 4 bits. Both use a 62-character
  vocabulary and a 64-character context. Turbo is the fast interactive demo; Dense is the
  artifact that actually stores every weight.
</p>

<h3>Model at a glance</h3>
<div class="table-wrap">
<table>
  <thead><tr><th>Property</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Task</td><td>Next-character prediction (causal character language modeling)</td></tr>
    <tr><td>Input</td><td>64 character IDs</td></tr>
    <tr><td>Dense architecture</td><td>Embedding 62×128 → flatten 8,192 → Linear 8,192×10,902 → ReLU → Linear 10,902×62 → softmax</td></tr>
    <tr><td>Turbo architecture</td><td>Same MLP shape with low-rank <code>U·V</code> factors, rank 30</td></tr>
    <tr><td>Vocabulary</td><td>62 characters: newline, space, punctuation, uppercase, lowercase</td></tr>
    <tr><td>Context</td><td>64 characters; longer prompts are truncated from the left</td></tr>
    <tr><td>Training data</td><td>First 200,000 bytes of Tiny Shakespeare (public domain)</td></tr>
    <tr><td>Training</td><td>Hand-written forward/backward passes, cross-entropy loss, SGD</td></tr>
    <tr><td>Dense artifact</td><td>90,004,008 weights · 45,117,066 bytes · 4-bit with row scales</td></tr>
    <tr><td>Turbo artifact</td><td>100,009,068 logical / 994,572 stored · 994,656 bytes · int8</td></tr>
    <tr><td>Runtime</td><td>CPU, single thread, <code>ndarray</code>; no GPU or Python</td></tr>
    <tr><td>Interfaces</td><td>CLI, Rust crate, HTTP JSON API, embedded browser UI</td></tr>
  </tbody>
</table>
</div>

<div class="grid two">
  <div class="card">
    <h3>What it is good at</h3>
    <ul>
      <li>Continuing text in a learned local style and format.</li>
      <li>Learning speaker labels, line breaks, spelling, and small-corpus patterns.</li>
      <li>Running offline on a CPU with no GPU, Python, or model hub.</li>
      <li>Teaching the full training and quantization pipeline in readable Rust.</li>
      <li>Embedding a tiny text model in a website, CLI, or desktop app.</li>
    </ul>
  </div>
  <div class="card">
    <h3>What it is not built for</h3>
    <ul>
      <li>Answering questions, reasoning, or following instructions.</li>
      <li>General world knowledge or factual recall.</li>
      <li>Long documents; the context is only 64 characters.</li>
      <li>Chat, tool use, streaming, or multi-turn conversation.</li>
      <li>GPU-scale training or production multi-tenant serving without a proxy.</li>
    </ul>
  </div>
</div>

<h2>Design principles</h2>
<p>
  Nib is a deliberately small, readable implementation of the whole language-model pipeline.
  The tensor math, layers, backpropagation, optimizer, tokenizer, training loop, quantization,
  HTTP server, and browser UI are all in this repository. The only heavyweight dependency is
  <code>ndarray</code> for array storage and matrix multiplication.
</p>

<div class="grid three">
  <div class="card">
    <div class="feature-icon">λ</div>
    <h3>From scratch</h3>
    <p>Xavier/He initialization, hand-written forward and backward passes, SGD, and cross-entropy. Nothing is downloaded except real training text.</p>
  </div>
  <div class="card">
    <div class="feature-icon">◫</div>
    <h3>Two text models</h3>
    <p><strong>TurboLM</strong> keeps a 100M-entry shape in under 1 MB. <strong>DenseLM</strong> stores 90M independent weights in 45 MB at 4 bits.</p>
  </div>
  <div class="card">
    <div class="feature-icon">⇄</div>
    <h3>One API</h3>
    <p><code>nib serve</code> auto-detects either format and exposes the same JSON endpoints, CORS, OpenAPI document, and browser UI.</p>
  </div>
  <div class="card">
    <div class="feature-icon">⌁</div>
    <h3>Embeddable</h3>
    <p>Export the UI as one HTML file, host it on GitHub Pages, Netlify, Vercel, or Cloudflare Pages, and point it at your API with <code>?api=</code>.</p>
  </div>
  <div class="card">
    <div class="feature-icon">▦</div>
    <h3>Vision too</h3>
    <p>A from-scratch CNN with convolution, pooling, and backprop for labeled image classification, trained from a simple manifest.</p>
  </div>
  <div class="card">
    <div class="feature-icon">▣</div>
    <h3>Portable</h3>
    <p>A single static binary, two model files, and a tokenizer. Docker image included; no runtime, VM, or GPU required.</p>
  </div>
</div>

<h2>Two text models, one integration</h2>
<div class="table-wrap">
<table>
  <thead>
    <tr><th>Model</th><th>What is stored</th><th>File size</th><th>Best for</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>TurboLM</strong></td>
      <td>Low-rank factors for a 100M-entry shape, int8</td>
      <td><code>994,656 bytes</code></td>
      <td>Fast CPU demos, small containers, interactive generation</td>
    </tr>
    <tr>
      <td><strong>DenseLM</strong></td>
      <td>90,004,008 independent weights, 4-bit with row scales</td>
      <td><code>45,117,066 bytes</code></td>
      <td>A real dense model that still fits under 50 MB</td>
    </tr>
    <tr>
      <td><strong>CharLM</strong></td>
      <td>Plain JSON weights for a small configurable MLP</td>
      <td>Varies (KB–MB)</td>
      <td>Learning the pipeline, tiny experiments, custom text</td>
    </tr>
    <tr>
      <td><strong>VisionCNN</strong></td>
      <td>JSON convolution/pooling weights</td>
      <td>Varies (KB–MB)</td>
      <td>Image classification experiments</td>
    </tr>
  </tbody>
</table>
</div>
<p>
  Both text formats are selected automatically by magic bytes, so the command and API do not
  change when you swap models. See <a href="models.html">Models &amp; training</a> for the
  architecture and exact size math.
</p>

<h2>Quick start</h2>
<div class="steps">
  <div class="step">
    <div>
      <h4>Clone the repository</h4>
      <pre><code>git clone ${repo}.git
cd nib-ai</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Build the release binary</h4>
      <pre><code>cargo build --release

# Windows helper (finds MinGW automatically):
.\\build.ps1 -Release</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Serve the shipped dense model</h4>
      <pre><code>./target/release/nib serve --model nib.bin --tokenizer tokenizer.json --port 8080

# Windows:
.\\target\\release\\nib.exe serve --model nib.bin --tokenizer tokenizer.json --port 8080</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Open the UI or call the API</h4>
      <pre><code>curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/api/generate \\
  -H "content-type: application/json" \\
  -d '{"prompt":"ROMEO: ","length":120,"temperature":0.7}'</code></pre>
    </div>
  </div>
</div>
<p class="muted small">
  For the fastest interactive demo, serve <code>turbo.bin</code> instead. Dense inference is
  roughly one to two orders of magnitude slower on CPU because every generated character
  touches all 90M weights.
</p>
<div class="callout">
  <strong>No public API is attached to the hosted UI</strong>
  <p>
    GitHub Pages serves static files only. The live UI has no backend until you start
    <code>nib serve</code> locally or deploy it to a Rust host. Local base URL:
    <code>http://127.0.0.1:8080</code>. Public endpoint steps, tunnels, and UI connection:
    <a href="http-api.html#endpoint">HTTP API → What is the endpoint?</a>.
  </p>
</div>

<h2>How the pipeline fits together</h2>
<div class="pipeline">
  <div class="pipe"><b>Text</b><span>real .txt corpus</span></div>
  <div class="pipe"><b>Tokenizer</b><span>character vocabulary</span></div>
  <div class="pipe"><b>Model</b><span>MLP / low-rank MLP</span></div>
  <div class="pipe"><b>4-bit file</b><span>nib.bin / turbo.bin</span></div>
  <div class="pipe"><b>HTTP + UI</b><span>one static binary</span></div>
</div>
<p>
  Training and inference share the same code. The dense trainer materializes low-rank factors
  into full matrices, then updates every weight with dense gradients. Saving quantizes each row
  to 4 bits with one f32 scale, so the artifact is small but still a normal dense network at
  load time.
</p>

<h2>What is actually in the box</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Path</th><th>Purpose</th></tr></thead>
  <tbody>
    <tr><td><code>src/main.rs</code></td><td>CLI: train, generate, dense/Turbo commands, serve, vision</td></tr>
    <tr><td><code>src/model.rs</code></td><td>Tokenizer, CharLM, training loop</td></tr>
    <tr><td><code>src/turbo.rs</code></td><td>Low-rank TurboLM, int8 persistence, Turbo trainer</td></tr>
    <tr><td><code>src/dense.rs</code></td><td>DenseLM, pruning, 4-bit format, batched dense trainer</td></tr>
    <tr><td><code>src/web.rs</code></td><td>Zero-dependency HTTP server, JSON API, embedded UI</td></tr>
    <tr><td><code>src/vision.rs</code></td><td>Conv2d, pooling, VisionCNN, image loading</td></tr>
    <tr><td><code>nib.bin</code></td><td>Shipped 90M dense model (45.12 MB)</td></tr>
    <tr><td><code>turbo.bin</code></td><td>Shipped compact model (994,656 bytes)</td></tr>
    <tr><td><code>tokenizer.json</code></td><td>62-character vocabulary built from the corpus</td></tr>
    <tr><td><code>docs/</code></td><td>This static documentation site</td></tr>
  </tbody>
</table>
</div>

<h2>Documentation map</h2>
<div class="grid two">
  <a class="card" href="getting-started.html">
    <span class="badge">Start here</span>
    <h3>Getting started</h3>
    <p>Install, build, run the shipped model, train your first model, and fix common problems.</p>
    <div class="card-arrow">Read the guide →</div>
  </a>
  <a class="card" href="use-cases.html">
    <span class="badge plain">Recipes</span>
    <h3>What you can do</h3>
    <p>Ten concrete tasks — text, training, autocomplete, APIs, websites, images, compression — with the exact commands.</p>
    <div class="card-arrow">Find your use case →</div>
  </a>
  <a class="card" href="cli.html">
    <span class="badge plain">Reference</span>
    <h3>CLI reference</h3>
    <p>Every command, flag, default, and a set of copy-paste recipes.</p>
    <div class="card-arrow">Browse commands →</div>
  </a>
  <a class="card" href="http-api.html">
    <span class="badge plain">Reference</span>
    <h3>HTTP API</h3>
    <p>Health, model info, generation, errors, CORS, limits, and client examples.</p>
    <div class="card-arrow">Call the API →</div>
  </a>
  <a class="card" href="rust-api.html">
    <span class="badge plain">Library</span>
    <h3>Rust API</h3>
    <p>Use Nib as a crate: models, trainers, quantization, generation, and vision.</p>
    <div class="card-arrow">Integrate the crate →</div>
  </a>
  <a class="card" href="vision.html">
    <span class="badge plain">Images</span>
    <h3>Vision &amp; OCR</h3>
    <p>Train the CNN on your own images, recognize single glyphs, and see exactly what OCR would require.</p>
    <div class="card-arrow">Classify images →</div>
  </a>
  <a class="card" href="models.html">
    <span class="badge plain">Deep dive</span>
    <h3>Models &amp; training</h3>
    <p>Architecture, parameter math, 4-bit format, data provenance, pruning, benchmarks.</p>
    <div class="card-arrow">Understand the model →</div>
  </a>
  <a class="card" href="deployment.html">
    <span class="badge plain">Operations</span>
    <h3>Deployment</h3>
    <p>Docker, VPS, GitHub Pages, Render/Railway/Fly, reverse proxies, and memory sizing.</p>
    <div class="card-arrow">Ship it →</div>
  </a>
</div>

<h2>Honest scope</h2>
<div class="callout warn">
  <strong>Small model, small corpus</strong>
  <p>
    The shipped model is trained on the first 200,000 bytes of Tiny Shakespeare, a
    public-domain corpus. It is a character-level model with a 64-character context. It learns
    spelling, speaker labels, line breaks, and short Shakespeare-like phrases; it is not a
    general assistant. Reported losses are training losses; there is no held-out validation
    split in the shipped run.
  </p>
</div>
<div class="callout">
  <strong>Size is physics, not branding</strong>
  <p>
    90M weights at 4 bits is about 45 MB. A 100B-parameter model cannot fit under 100 MB:
    even 1 bit per weight needs roughly 12.5 GB. Nib targets the achievable end of the size
    curve, and the math is shown on the <a href="models.html#size-math">models page</a>.
  </p>
</div>
`,
  },
  {
    slug: "getting-started",
    title: "Getting started",
    description:
      "Install Rust, build nib-rs, serve the shipped 90M dense model, call the API, and train your first character-level model.",
    body: `
<section class="hero">
  <p class="eyebrow">Start here</p>
  <h1>Getting started</h1>
  <p class="lead">
    From a clean machine to a running model in five minutes. The repository already contains a
    trained 90M dense model and a compact Turbo model, so you can serve something useful before
    training anything yourself.
  </p>
</section>

<h2>Prerequisites</h2>
<ul>
  <li><strong>Rust stable</strong> (2021 edition). Install with <a href="https://rustup.rs" rel="noopener">rustup</a>.</li>
  <li><strong>Windows:</strong> the GNU toolchain and MinGW. <code>build.ps1</code> locates MinGW automatically; plain <code>cargo build</code> also works if your toolchain is configured.</li>
  <li><strong>No Python, Node, CUDA, or GPU.</strong> The server, trainer, tokenizer, and UI are all in the Rust binary.</li>
  <li><strong>Disk and RAM:</strong> about 500 MB for <code>target/</code>; dense inference needs roughly 400 MB of RAM for the f32 weights.</li>
</ul>
<pre><code># Windows (PowerShell)
winget install Rustlang.Rustup
winget install BrechtSanders.WinLibs.POSIX.UCRT

# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh</code></pre>
<p class="muted small">Restart the terminal after installing so <code>cargo</code> is on <code>PATH</code>.</p>

<h2>1. Clone and build</h2>
<pre><code>git clone ${repo}.git
cd nib-ai

# Windows: finds MinGW and uses the GNU toolchain
./build.ps1 -Release

# Any platform with cargo on PATH
cargo build --release</code></pre>
<p>
  The release profile enables LTO, one codegen unit, <code>panic = "abort"</code>, and symbol
  stripping. The resulting binary is about <strong>3 MB</strong>.
</p>

<h2>2. Verify the binary</h2>
<pre><code>./target/release/nib version
./target/release/nib --help

# Windows
./target/release/nib.exe version</code></pre>
<p>Expected output includes <code>nib-rs ${version}</code> and the full command list.</p>

<h2>3. Run the shipped model</h2>
<p>
  <code>nib.bin</code> is a 45.12 MB dense model with 90,004,008 independent weights. Start the
  server on localhost:
</p>
<pre><code>./target/release/nib serve --model nib.bin --tokenizer tokenizer.json --host 127.0.0.1 --port 8080

# Windows
./target/release/nib.exe serve --model nib.bin --tokenizer tokenizer.json --host 127.0.0.1 --port 8080</code></pre>
<p>You should see:</p>
<pre><code>nib server listening on http://127.0.0.1:8080/
API: POST http://127.0.0.1:8080/api/generate
model: nib.bin (45117066 bytes, 45.117 MB), CORS enabled</code></pre>
<p>Check that it is alive and inspect the model metadata:</p>
<pre><code>curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/api/info</code></pre>
<pre><code>{"status":"ok"}

{
  "name": "DenseLM",
  "format": "dense-int4",
  "architecture": "dense MLP, 4-bit weights on disk",
  "logical_parameters": 90004008,
  "stored_parameters": 90004008,
  "file_bytes": 45117066,
  "config": { "vocab_size": 62, "embed_dim": 128, "hidden_dim": 10902, "context": 64 }
}</code></pre>

<h2>4. Generate text</h2>
<h3>Browser UI</h3>
<p>
  Open <a href="http://127.0.0.1:8080/">http://127.0.0.1:8080/</a>. The page detects the model,
  shows its size, and generates text. It also produces an <code>&lt;iframe&gt;</code> embed
  snippet that preserves the API endpoint.
</p>
<h3>CLI</h3>
<pre><code>./target/release/nib dense-generate --model nib.bin --tokenizer tokenizer.json \\
  --seed "First Citizen: Before we proceed any further, hear me speak." \\
  --length 160 --temperature 0.7</code></pre>
<h3>HTTP</h3>
<pre><code>curl -X POST http://127.0.0.1:8080/api/generate \\
  -H "content-type: application/json" \\
  -d '{"prompt":"First Citizen: Before we proceed any further, hear me speak.","length":120,"temperature":0.7}'</code></pre>
<p>
  The prompt is repeated to fill the 64-character context when it is shorter than that. Longer,
  natural prompts produce the best results. Dense generation takes seconds per short request on
  a laptop CPU; serve <code>turbo.bin</code> when you need a fast interactive demo.
</p>

<h2>5. Train your first model (CharLM)</h2>
<p>
  CharLM is the smallest and easiest model. It uses a plain JSON file, trains in seconds to
  minutes on a laptop, and is a good way to verify the full pipeline.
</p>
<pre><code>./get-data.ps1

./target/release/nib train --input shakespeare.txt --output model.json \\
  --tokenizer tokenizer.json --context 12 --hidden 128 --embed 24 \\
  --epochs 40 --lr 0.1</code></pre>
<p>Then generate:</p>
<pre><code>./target/release/nib generate --model model.json --tokenizer tokenizer.json \\
  --seed "ROMEO: " --length 300 --temperature 0.7</code></pre>
<div class="callout">
  <strong>What the loss means</strong>
  <p>
    The trainer prints cross-entropy loss. Lower is better; on the 200 KB Shakespeare prefix a
    small CharLM typically falls from about 2.35 to about 1.46 after 40 epochs. These are
    training losses, not held-out validation numbers.
  </p>
</div>

<h2>6. Train TurboLM</h2>
<p>
  Turbo stores a 100M-entry shape as low-rank factors and quantizes them to int8. It trains
  quickly and is the best choice for a fast web demo.
</p>
<pre><code>./target/release/nib turbo-train --input shakespeare.txt --output turbo.bin \\
  --tokenizer tokenizer.json --target 100000000 --context 64 --embed 128 \\
  --rank 30 --epochs 5 --lr 0.05 --batch-size 32</code></pre>
<p>
  A verified 5-epoch run on the 200 KB prefix went from loss <strong>2.3209</strong> to
  <strong>1.7765</strong> and produced a <strong>994,656-byte</strong> file.
</p>
<pre><code>./target/release/nib turbo-generate --model turbo.bin --tokenizer tokenizer.json \\
  --seed "ROMEO: " --length 300 --temperature 0.7</code></pre>

<h2>7. Build and prune DenseLM</h2>
<p>
  A dense model starts by materializing the trained Turbo factors into full matrices, then
  fine-tuning every weight with dense gradients. The shipped model then prunes 1,212 hidden
  units to land under 50 MB and fine-tunes the remaining sub-network.
</p>
<pre><code># Materialize the 100M dense model from Turbo
./target/release/nib dense-build --turbo turbo.bin --output dense-100m.bin \\
  --tokenizer tokenizer.json

# Fine-tune all 100M weights
./target/release/nib dense-train --input shakespeare.txt --output dense-100m.bin \\
  --tokenizer tokenizer.json --init-turbo turbo.bin \\
  --epochs 2 --max-samples 20000 --batch-size 128 --lr 0.02

# Prune to 10,902 hidden units and fine-tune the sub-network
./target/release/nib dense-train --input shakespeare.txt --output nib.bin \\
  --tokenizer tokenizer.json --init-dense dense-100m.bin --hidden 10902 \\
  --epochs 3 --max-samples 20000 --batch-size 128 --lr 0.01</code></pre>
<p>
  The final file is <strong>45,117,066 bytes</strong> with <strong>90,004,008</strong>
  parameters. Dense training is CPU-heavy: batch 128 is roughly 3× faster than batch 32 on the
  development machine.
</p>

<h2>Project layout</h2>
<pre><code>nib-ai/
  src/
    main.rs      CLI commands and training loops
    model.rs     Tokenizer, CharLM, generic trainer
    turbo.rs     Low-rank TurboLM and int8 format
    dense.rs     DenseLM, pruning, 4-bit format, dense trainer
    web.rs       HTTP server, JSON API, embedded UI
    vision.rs    CNN, convolution, pooling, image loading
    layers.rs    Linear and Embedding layers
    tensor.rs    Tensor wrappers over ndarray
    optimizer.rs SGD and Adam
  docs/          This static documentation site
  nib.bin        Shipped 90M dense model (45.12 MB)
  turbo.bin      Shipped compact model (994,656 bytes)
  tokenizer.json 62-character vocabulary
  index.html     Static UI export
  Dockerfile     Multi-stage build and runtime image
  get-data.ps1   Public corpus downloader
  build.ps1      Windows GNU build helper</code></pre>

<h2>Troubleshooting</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Symptom</th><th>Cause and fix</th></tr></thead>
  <tbody>
    <tr><td><code>cargo</code> is not recognized</td><td>Restart the terminal after installing rustup, or add <code>%USERPROFILE%\\.cargo\\bin</code> to <code>PATH</code>.</td></tr>
    <tr><td>Linker or <code>dlltool</code> errors on Windows</td><td>Install WinLibs MinGW and use <code>./build.ps1 -Release</code>, which finds it automatically.</td></tr>
    <tr><td><code>bind 0.0.0.0:8080</code> failed</td><td>Another process owns the port. Use <code>--port 8081</code> or stop the old process. On Windows: <code>Get-NetTCPConnection -LocalPort 8080</code>.</td></tr>
    <tr><td><code>tokenizer vocab (...) does not match model vocab</code></td><td>The tokenizer was built from different text. Use the <code>tokenizer.json</code> that belongs to the model, or retrain both on the same corpus.</td></tr>
    <tr><td><code>is not a nib model</code></td><td>The file is not a Turbo int8 (<code>CTURBO01</code>) or dense int4 (<code>CDENSE4</code>) artifact. Rebuild it with the commands above.</td></tr>
    <tr><td>Generation is very slow</td><td>Expected for the 90M dense model on CPU. Serve <code>turbo.bin</code> for interactive use, shorten <code>length</code>, or use a faster machine.</td></tr>
    <tr><td><code>413 Payload Too Large</code></td><td>The request body exceeds 64 KB. Send a shorter prompt or fewer fields.</td></tr>
    <tr><td><code>prompt is limited to 4096 characters</code></td><td>The API caps prompts at 4,096 characters by design.</td></tr>
    <tr><td>GitHub rejects a model file</td><td>GitHub blocks files over 100 MB. <code>nib.bin</code> is 45 MB and fits; use Git LFS or Releases for larger variants.</td></tr>
  </tbody>
</table>
</div>

<h2>Next steps</h2>
<div class="grid two">
  <a class="card" href="use-cases.html"><h3>What you can do</h3><p>Ten practical recipes: text generation, custom training, autocomplete, APIs, websites, images, and compression.</p><div class="card-arrow">Find your use case →</div></a>
  <a class="card" href="cli.html"><h3>CLI reference</h3><p>Every command and flag, including resume and prune recipes.</p><div class="card-arrow">Open reference →</div></a>
  <a class="card" href="http-api.html"><h3>HTTP API</h3><p>Request and response schemas, errors, CORS, and client examples.</p><div class="card-arrow">Open API docs →</div></a>
  <a class="card" href="models.html"><h3>Models &amp; training</h3><p>Architecture, size math, data provenance, and benchmarks.</p><div class="card-arrow">Read the deep dive →</div></a>
  <a class="card" href="deployment.html"><h3>Deployment</h3><p>Docker, Pages, VPS, reverse proxies, and memory sizing.</p><div class="card-arrow">Deploy Nib →</div></a>
</div>
`,
  },
  {
    slug: "use-cases",
    title: "What you can do",
    description:
      "Concrete things nib-rs can do today: continue text, train on your own corpus, build autocomplete, serve an API, embed a generator on any website, classify images, compress models, and teach how training works.",
    body: `
<section class="hero">
  <p class="eyebrow">Capabilities</p>
  <h1>What you can do with Nib</h1>
  <p class="lead">
    Nib is not a chatbot. It is a small, trainable character model plus a complete toolkit
    around it. These are the concrete tasks it handles today, with the exact command or API
    for each one.
  </p>
</section>

<h2>Pick the right model first</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>You want…</th><th>Use</th><th>Why</th></tr></thead>
  <tbody>
    <tr><td>An instant local demo</td><td><code>turbo.bin</code></td><td>Under 1 MB, fast CPU generation, low memory</td></tr>
    <tr><td>A real dense 90M-weight artifact</td><td><code>nib.bin</code></td><td>Every weight stored, 45 MB at 4 bits, slower inference</td></tr>
    <tr><td>To learn or modify the pipeline</td><td><code>CharLM</code> (<code>nib train</code>)</td><td>Plain JSON weights, tiny configurable MLP</td></tr>
    <tr><td>Image classification</td><td><code>VisionCNN</code></td><td>Convolution and pooling trained from a manifest</td></tr>
  </tbody>
</table>
</div>

<h2>1. Continue text in a learned style</h2>
<p><strong>The problem:</strong> you want offline, deterministic-ish text continuation without an API key, GPU, or Python stack.</p>
<pre><code># Fast model: 200 characters in well under a second
nib turbo-generate --model turbo.bin --tokenizer tokenizer.json \\
  --seed "ROMEO: " --length 200 --temperature 0.7

# Dense model: slower, but every one of the 90M weights is used
nib dense-generate --model nib.bin --tokenizer tokenizer.json \\
  --seed "First Citizen: Before we proceed any further, hear me speak." \\
  --length 120 --temperature 0.7</code></pre>
<p><strong>Real output from the shipped <code>nib.bin</code></strong> (temperature 0.7):</p>
<pre><code>First Citizen: Before we proceed any further, hear me speak.

CORIOLANUS:
Au, I chole vile wishing hear ford fanit. Bubl as, no Rome.

COMINIUS:
I sull were your be somen:</code></pre>
<p>
  It reproduces speaker labels, line breaks, and Elizabethan spelling patterns. It does not
  understand the plot. Longer, natural prompts produce better output than one-word seeds.
</p>

<h2>2. Train on your own text</h2>
<p><strong>The problem:</strong> you have a corpus — notes, lyrics, source code, logs, a novel — and want a model that mimics it.</p>
<pre><code># 1. Put UTF-8 text in a file, e.g. mydata.txt
# 2. Train a small readable CharLM (JSON, fast)
nib train --input mydata.txt --output model.json --tokenizer tokenizer.json \\
  --context 12 --hidden 128 --embed 24 --epochs 40 --lr 0.1

# 3. Or train the compact Turbo model for better quality at ~1 MB
nib turbo-train --input mydata.txt --output turbo.bin --tokenizer tokenizer.json \\
  --target 100000000 --context 64 --embed 128 --rank 30 \\
  --epochs 5 --batch-size 32 --max-samples 50000

# 4. Or train a dense model from random weights (slow on CPU)
nib dense-train --input mydata.txt --output my-dense.bin --tokenizer tokenizer.json \\
  --target 100000000 --epochs 1 --max-samples 2000 --batch-size 128 --lr 0.01</code></pre>
<ul>
  <li><strong>Use at least ~100 KB of clean text.</strong> More data helps more than more epochs.</li>
  <li><strong>Use a matching tokenizer.</strong> <code>train</code> and <code>turbo-train</code> build one from the corpus if the file is missing. Keep the tokenizer with the model.</li>
  <li><strong>Character-level means every character counts.</strong> Mixed encodings, HTML boilerplate, or binary junk become part of the vocabulary.</li>
  <li><strong>Validate by sampling, not just loss.</strong> The trainer reports training cross-entropy; there is no held-out split.</li>
</ul>

<h2>3. Build a next-character predictor or autocomplete</h2>
<p><strong>The problem:</strong> you need a tiny model that scores the next character given a short context, inside a Rust program.</p>
<pre><code>use nib_rs::{dense, turbo, Tokenizer};

let tok = Tokenizer::load("tokenizer.json")?;
let mut model = dense::load_int4("nib.bin")?;
let context = model.config.context;

let mut ctx = turbo::context_from_seed(&amp;tok, "Once upon a time", context);
let mut out = String::from("Once upon a time");
for _ in 0..40 {
    let probs = model.predict(&amp;ctx);
    let next = turbo::sample(&amp;probs, 0.7);
    out.push(tok.id_to_char.get(&amp;next).copied().unwrap_or('?'));
    ctx.remove(0);
    ctx.push(next);
}
println!("{out}");</code></pre>
<p>
  <code>predict</code> returns a probability vector over the 62-character vocabulary. Use
  <code>argmax</code> for deterministic completion or <code>turbo::sample</code> for creative
  continuation. The context is fixed at 64 characters; slide the window as you go.
</p>

<h2>4. Run a JSON API for any client</h2>
<p><strong>The problem:</strong> a website, script, or mobile app needs a simple HTTP endpoint.</p>
<pre><code>nib serve --model nib.bin --tokenizer tokenizer.json --host 0.0.0.0 --port 8080</code></pre>
<pre><code>curl -X POST http://127.0.0.1:8080/api/generate \\
  -H "content-type: application/json" \\
  -d '{"prompt":"ROMEO: ","length":120,"temperature":0.7}'</code></pre>
<p>
  The response is plain JSON with <code>text</code>, <code>prompt</code>, <code>length</code>,
  <code>temperature</code>, and <code>ms</code>. CORS is open, so a page on any origin can call
  it. This is a local endpoint by default; the hosted Pages UI has no backend. To get a public
  HTTPS URL, use a tunnel for a demo or deploy the binary, then open the UI with
  <code>?api=https://your-host</code>. See
  <a href="http-api.html#endpoint">HTTP API → What is the endpoint?</a> for every step.
</p>

<h2>5. Put a generator on any website</h2>
<p><strong>The problem:</strong> you want a working demo on GitHub Pages, Netlify, Vercel, Cloudflare Pages, or an existing site, without a frontend build.</p>
<pre><code># Export the single-file UI
nib web-export --output index.html

# Host index.html anywhere static, then point it at the API
https://your-site.example/?api=https://your-nib-api.example</code></pre>
<pre><code>&lt;iframe src="https://your-site.example/?api=https://your-nib-api.example"
        width="100%" height="760"
        style="border:0;border-radius:16px"
        title="nib-rs language model"&gt;&lt;/iframe&gt;</code></pre>
<p>
  The UI stores the endpoint in <code>localStorage</code>, checks <code>/health</code>, and
  shows the ready-made iframe snippet. Static hosting cannot run Rust, so the API lives on a
  separate VPS, Docker host, Render, Railway, Fly.io, or Cloud Run service.
</p>

<h2>6. Embed the model in a Rust application</h2>
<p><strong>The problem:</strong> you want generation in-process, with no server and no network hop.</p>
<pre><code>[dependencies]
nib-rs = { git = "${repo}.git" }</code></pre>
<pre><code>use nib_rs::{dense, Tokenizer};

let tok = Tokenizer::load("tokenizer.json")?;
let mut model = dense::load_int4("nib.bin")?;
let text = dense::generate_text(&amp;mut model, &amp;tok, "ROMEO: ", 120, 0.7);
println!("{text}");</code></pre>
<p>
  The <a href="rust-api.html">Rust API page</a> covers training, quantization, pruning,
  low-rank models, vision, and the HTTP server. Dense loading needs roughly 400 MB of RAM;
  Turbo stays under about 20 MB.
</p>

<h2>7. Ship it with Docker or a single binary</h2>
<p><strong>The problem:</strong> you need a reproducible deployment with no runtime dependencies.</p>
<pre><code># Fast Turbo image
docker build -t nib-turbo .
docker run --rm -p 8080:8080 nib-turbo

# 90M dense image
docker build --build-arg MODEL=nib.bin -t nib-dense .
docker run --rm -p 8080:8080 --memory=1g nib-dense</code></pre>
<p>
  The runtime image contains the <code>nib</code> binary, one model, and the tokenizer. See
  <a href="deployment.html">Deployment</a> for systemd, Caddy, nginx, GitHub Pages, and
  platform-specific settings.
</p>

<h2 id="classify-images">8. Classify images with the CNN</h2>
<p><strong>The problem:</strong> you want a from-scratch convolutional classifier for a small, controlled image set — shapes, colors, symbols, simple objects.</p>
<pre><code># manifest.txt: one "path label" pair per line
# data/red/01.png 0
# data/blue/01.png 1

nib vision-train --manifest manifest.txt --output vision.json \\
  --image-size 16 --channels 3 --classes 2 --epochs 20 --batch-size 16 --lr 0.01

nib vision-predict --model vision.json --image data/red/02.png \\
  --classes classes.txt</code></pre>
<p>
  <code>classes.txt</code> is optional: one display name per line. The CNN resizes images,
  converts channels, normalizes pixels, and trains convolution, pooling, and the final linear
  layer with hand-written backprop. It is suitable for toy or narrow datasets, not ImageNet.
</p>
<p>
  <strong>OCR is not part of this.</strong> The CNN returns one label for the whole image. It can
  classify a pre-cropped character, but it cannot find or read a line of text. The
  <a href="vision.html">Vision &amp; OCR guide</a> has a measured digit example (95.33% on 300
  held-out images), the single-glyph recipe, and the full list of OCR requirements.
</p>

<h2>9. Explore size and compression tradeoffs</h2>
<p><strong>The problem:</strong> you want to see exactly what fits in a size budget before committing to a training run.</p>
<pre><code># Turbo: logical shape vs stored factors vs file bytes
nib turbo-info --target 100000000 --tokenizer tokenizer.json

# Dense: exact parameters and 4-bit bytes for a hidden width
nib dense-info --hidden 10902 --tokenizer tokenizer.json

# Materialize a dense model, then prune and fine-tune it
nib dense-build --turbo turbo.bin --output dense-100m.bin --tokenizer tokenizer.json
nib dense-train --input shakespeare.txt --output nib.bin --tokenizer tokenizer.json \\
  --init-dense dense-100m.bin --hidden 10902 \\
  --epochs 3 --max-samples 20000 --batch-size 128 --lr 0.01</code></pre>
<p>
  <code>turbo-info</code> and <code>dense-info</code> are instant and do not allocate the
  model. Use them to choose a shape, then train once. The <a href="models.html">Models &amp;
  training</a> page has the full size math and the 100B impossibility argument.
</p>

<h2>10. Learn or teach how a language model works</h2>
<p><strong>The problem:</strong> you want a complete, readable implementation instead of a framework call.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Topic</th><th>File</th><th>What is hand-written</th></tr></thead>
  <tbody>
    <tr><td>Tensors and indexing</td><td><code>src/tensor.rs</code></td><td><code>Tensor1</code>, <code>Tensor2</code>, <code>Tensor4</code></td></tr>
    <tr><td>Layers</td><td><code>src/layers.rs</code></td><td>Linear and embedding forward/backward</td></tr>
    <tr><td>Text model</td><td><code>src/model.rs</code></td><td>CharLM, tokenizer, trainer, sampling</td></tr>
    <tr><td>Compression</td><td><code>src/turbo.rs</code>, <code>src/dense.rs</code></td><td>Low-rank factors, int8, 4-bit, pruning</td></tr>
    <tr><td>Vision</td><td><code>src/vision.rs</code></td><td>Conv2d, max pooling, CNN backprop</td></tr>
    <tr><td>Serving</td><td><code>src/web.rs</code></td><td>HTTP parsing, JSON API, CORS, UI</td></tr>
  </tbody>
</table>
</div>
<p>
  Every layer is a normal Rust struct with explicit gradients. Run <code>cargo test</code> to
  see shape, roundtrip, training, and HTTP tests. There is no autograd and no hidden
  framework magic to trace through.
</p>

<h2>What it cannot do</h2>
<div class="callout warn">
  <strong>Set expectations before you build on it</strong>
  <p>
    The shipped models are character-level MLPs trained on 200 KB of Shakespeare. They
    continue text; they do not answer questions, follow instructions, browse, call tools, or
    hold a conversation. There is no attention, no BPE tokenizer, no chat template, and no
    world knowledge beyond the corpus.
  </p>
</div>
<ul>
  <li><strong>No question answering or reasoning.</strong> “What is the capital of France?” is outside the training objective.</li>
  <li><strong>No long context.</strong> The window is 64 characters; longer prompts are truncated from the left.</li>
  <li><strong>No streaming or batching.</strong> One JSON response per request; generation is serialized.</li>
  <li><strong>No GPU backend.</strong> Everything runs on the CPU through <code>ndarray</code>.</li>
  <li><strong>No built-in auth, TLS, or rate limiting.</strong> Put a reverse proxy in front of a public instance.</li>
  <li><strong>No 100B model in 100 MB.</strong> That is physically impossible at any usable precision; Nib targets 90M weights in 45 MB.</li>
</ul>

<h2>Recipe chooser</h2>
<div class="grid two">
  <div class="card">
    <h3>“I just want to try it”</h3>
    <pre><code>nib turbo-generate --model turbo.bin \\
  --tokenizer tokenizer.json --seed "ROMEO: "</code></pre>
    <p>Fast, tiny, no extra setup.</p>
  </div>
  <div class="card">
    <h3>“I want the real dense model”</h3>
    <pre><code>nib dense-generate --model nib.bin \\
  --tokenizer tokenizer.json \\
  --seed "First Citizen: Before we proceed any further, hear me speak."</code></pre>
    <p>All 90,004,008 weights, 45.12 MB on disk.</p>
  </div>
  <div class="card">
    <h3>“I have my own text”</h3>
    <pre><code>nib turbo-train --input mydata.txt \\
  --output turbo.bin --tokenizer tokenizer.json \\
  --epochs 5 --max-samples 50000</code></pre>
    <p>Build a model in minutes on a laptop CPU.</p>
  </div>
  <div class="card">
    <h3>“I need a web API”</h3>
    <pre><code>nib serve --model turbo.bin \\
  --tokenizer tokenizer.json --port 8080</code></pre>
    <p>Then POST to <code>/api/generate</code> from any origin.</p>
  </div>
</div>
`,
  },
  {
    slug: "cli",
    title: "CLI reference",
    description:
      "Complete nib-rs command-line reference: train, generate, Turbo, dense, serve, web-export, vision, and version commands with every flag and default.",
    body: `
<section class="hero">
  <p class="eyebrow">Reference</p>
  <h1>CLI reference</h1>
  <p class="lead">
    One binary, <code>nib</code>, contains the trainers, generators, HTTP server, static UI
    exporter, and vision tools. Run <code>nib &lt;command&gt; --help</code> for the same
    information from the terminal.
  </p>
</section>

<h2>Synopsis</h2>
<pre><code>nib &lt;command&gt; [options]</code></pre>
<p>Windows users can call <code>./target/release/nib.exe</code>; every example below uses the portable forward-slash form.</p>

<h2>Command index</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Command</th><th>Purpose</th><th>Output</th></tr></thead>
  <tbody>
    <tr><td><code>train</code></td><td>Train a character-level MLP from text</td><td><code>model.json</code></td></tr>
    <tr><td><code>generate</code></td><td>Sample text from a CharLM JSON model</td><td>stdout</td></tr>
    <tr><td><code>turbo-info</code></td><td>Show Turbo parameter and file-size math</td><td>stdout</td></tr>
    <tr><td><code>turbo-train</code></td><td>Train a compact low-rank model</td><td><code>turbo.bin</code></td></tr>
    <tr><td><code>turbo-generate</code></td><td>Sample text from a Turbo model</td><td>stdout</td></tr>
    <tr><td><code>dense-info</code></td><td>Show dense parameter and 4-bit size math</td><td>stdout</td></tr>
    <tr><td><code>dense-build</code></td><td>Materialize Turbo factors into dense matrices</td><td><code>nib.bin</code></td></tr>
    <tr><td><code>dense-train</code></td><td>Train, fine-tune, resume, or prune a dense model</td><td><code>nib.bin</code></td></tr>
    <tr><td><code>dense-generate</code></td><td>Sample text from a dense 4-bit model</td><td>stdout</td></tr>
    <tr><td><code>serve</code></td><td>Serve Turbo or dense model over HTTP</td><td>long-running server</td></tr>
    <tr><td><code>web-export</code></td><td>Write the browser UI to a static HTML file</td><td><code>index.html</code></td></tr>
    <tr><td><code>vision-train</code></td><td>Train a CNN from a labeled image manifest</td><td><code>vision.json</code></td></tr>
    <tr><td><code>vision-predict</code></td><td>Classify one image</td><td>stdout</td></tr>
    <tr><td><code>version</code></td><td>Print the package version</td><td>stdout</td></tr>
  </tbody>
</table>
</div>

<h2>Conventions</h2>
<ul>
  <li><strong>Short flags</strong> exist where noted: <code>-i</code> input, <code>-o</code> output, <code>-m</code> model, <code>-s</code> seed, <code>-e</code> epochs.</li>
  <li><strong>Paths</strong> are resolved relative to the current directory. Run commands from the repository root unless you pass absolute paths.</li>
  <li><strong>Tokenizers must match the model vocabulary.</strong> A 62-token tokenizer cannot load a 65-token model.</li>
  <li><strong>All training is CPU-only</strong> and uses plain SGD. The <code>--optimizer</code> flag on <code>train</code> is accepted but the current training step always applies SGD.</li>
  <li><strong>Loss</strong> is mean cross-entropy over the epoch or logging window. It is a training loss, not validation accuracy.</li>
</ul>

<h2>Text model commands</h2>

<h3><code>nib train</code></h3>
<p>Train a small character-level MLP. The tokenizer is built from the input text and saved to the tokenizer path.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--input, -i</code></td><td><code>data.txt</code></td><td>Training text file</td></tr>
    <tr><td><code>--output, -o</code></td><td><code>model.json</code></td><td>Output model path</td></tr>
    <tr><td><code>--tokenizer</code></td><td><code>tokenizer.json</code></td><td>Where to save the built tokenizer</td></tr>
    <tr><td><code>--context</code></td><td><code>8</code></td><td>Characters of context</td></tr>
    <tr><td><code>--hidden</code></td><td><code>64</code></td><td>Hidden units</td></tr>
    <tr><td><code>--embed</code></td><td><code>16</code></td><td>Embedding width</td></tr>
    <tr><td><code>--lr</code></td><td><code>0.1</code></td><td>Learning rate</td></tr>
    <tr><td><code>--epochs, -e</code></td><td><code>200</code></td><td>Passes over the training windows</td></tr>
    <tr><td><code>--optimizer</code></td><td><code>sgd</code></td><td>Accepted for compatibility; SGD is used</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib train --input shakespeare.txt --output model.json --tokenizer tokenizer.json \\
  --context 12 --hidden 128 --embed 24 --epochs 40 --lr 0.1</code></pre>

<h3><code>nib generate</code></h3>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--model, -m</code></td><td><code>model.json</code></td><td>CharLM JSON model</td></tr>
    <tr><td><code>--tokenizer</code></td><td><code>tokenizer.json</code></td><td>Tokenizer file</td></tr>
    <tr><td><code>--seed, -s</code></td><td>empty</td><td>Prompt text</td></tr>
    <tr><td><code>--length</code></td><td><code>200</code></td><td>Characters to generate</td></tr>
    <tr><td><code>--temperature</code></td><td><code>1.0</code></td><td>Sampling temperature</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib generate --model model.json --tokenizer tokenizer.json \\
  --seed "ROMEO: " --length 500 --temperature 0.7</code></pre>

<h2>Turbo commands</h2>

<h3><code>nib turbo-info</code></h3>
<p>Prints the logical dense shape, stored parameter count, and estimated int8 file size. Nothing is written.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--target</code></td><td><code>100000000</code></td><td>Dense logical parameter target</td></tr>
    <tr><td><code>--vocab</code></td><td><code>62</code></td><td>Vocabulary size</td></tr>
    <tr><td><code>--context</code></td><td><code>64</code></td><td>Context length</td></tr>
    <tr><td><code>--embed</code></td><td><code>128</code></td><td>Embedding width</td></tr>
    <tr><td><code>--rank</code></td><td><code>30</code></td><td>Low-rank factor width</td></tr>
    <tr><td><code>--tokenizer</code></td><td>none</td><td>Read the vocabulary size from a tokenizer file</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib turbo-info --target 100000000 --tokenizer tokenizer.json</code></pre>

<h3><code>nib turbo-train</code></h3>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--input, -i</code></td><td><code>shakespeare.txt</code></td><td>Training text</td></tr>
    <tr><td><code>--output, -o</code></td><td><code>turbo.bin</code></td><td>Output artifact</td></tr>
    <tr><td><code>--tokenizer</code></td><td><code>tokenizer.json</code></td><td>Tokenizer path; built if missing</td></tr>
    <tr><td><code>--target</code></td><td><code>100000000</code></td><td>Dense logical parameter target</td></tr>
    <tr><td><code>--context</code></td><td><code>64</code></td><td>Context length</td></tr>
    <tr><td><code>--embed</code></td><td><code>128</code></td><td>Embedding width</td></tr>
    <tr><td><code>--rank</code></td><td><code>30</code></td><td>Low-rank factor width</td></tr>
    <tr><td><code>--lr</code></td><td><code>0.05</code></td><td>Learning rate</td></tr>
    <tr><td><code>--epochs, -e</code></td><td><code>5</code></td><td>Training epochs</td></tr>
    <tr><td><code>--batch-size</code></td><td><code>32</code></td><td>Mini-batch size</td></tr>
    <tr><td><code>--max-samples</code></td><td><code>0</code></td><td>Windows per epoch; <code>0</code> means all</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib turbo-train --input shakespeare.txt --output turbo.bin \\
  --tokenizer tokenizer.json --target 100000000 --context 64 \\
  --embed 128 --rank 30 --epochs 5 --lr 0.05 --batch-size 32</code></pre>

<h3><code>nib turbo-generate</code></h3>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--model, -m</code></td><td><code>turbo.bin</code></td><td>Turbo int8 model</td></tr>
    <tr><td><code>--tokenizer</code></td><td><code>tokenizer.json</code></td><td>Tokenizer file</td></tr>
    <tr><td><code>--seed, -s</code></td><td><code>ROMEO: </code></td><td>Prompt text</td></tr>
    <tr><td><code>--length</code></td><td><code>500</code></td><td>Characters to generate</td></tr>
    <tr><td><code>--temperature</code></td><td><code>0.7</code></td><td>Sampling temperature</td></tr>
  </tbody>
</table>
</div>

<h2>Dense commands</h2>

<h3><code>nib dense-info</code></h3>
<p>Prints the dense shape, exact parameter count, and exact 4-bit file size. Add <code>--hidden</code> to check a pruned configuration without building it.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--target</code></td><td><code>100000000</code></td><td>Parameter target when <code>--hidden</code> is 0</td></tr>
    <tr><td><code>--vocab</code></td><td><code>62</code></td><td>Vocabulary size</td></tr>
    <tr><td><code>--context</code></td><td><code>64</code></td><td>Context length</td></tr>
    <tr><td><code>--embed</code></td><td><code>128</code></td><td>Embedding width</td></tr>
    <tr><td><code>--hidden</code></td><td><code>0</code></td><td>Explicit hidden size; <code>0</code> derives it from <code>--target</code></td></tr>
    <tr><td><code>--tokenizer</code></td><td>none</td><td>Read vocabulary size from a tokenizer file</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib dense-info --hidden 10902 --tokenizer tokenizer.json</code></pre>

<h3><code>nib dense-build</code></h3>
<p>Expands a trained Turbo model into full dense matrices (<code>W = U·V</code>). The output has the same predictions as the Turbo model, up to f32 rounding.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--turbo</code></td><td><code>turbo.bin</code></td><td>Source Turbo model</td></tr>
    <tr><td><code>--output, -o</code></td><td><code>nib.bin</code></td><td>Dense output path</td></tr>
    <tr><td><code>--tokenizer</code></td><td><code>tokenizer.json</code></td><td>Tokenizer for vocabulary validation</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib dense-build --turbo turbo.bin --output dense-100m.bin --tokenizer tokenizer.json</code></pre>

<h3><code>nib dense-train</code></h3>
<p>
  Trains a dense model from random weights, fine-tunes a Turbo expansion, resumes a dense
  artifact, or prunes and fine-tunes a smaller sub-network. The output is always a complete
  dense 4-bit file.
</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--input, -i</code></td><td><code>shakespeare.txt</code></td><td>Training text</td></tr>
    <tr><td><code>--output, -o</code></td><td><code>nib.bin</code></td><td>Dense output path</td></tr>
    <tr><td><code>--tokenizer</code></td><td><code>tokenizer.json</code></td><td>Tokenizer path</td></tr>
    <tr><td><code>--init-turbo</code></td><td>none</td><td>Expand this Turbo model, then fine-tune all weights</td></tr>
    <tr><td><code>--init-dense</code></td><td>none</td><td>Resume from this dense 4-bit model</td></tr>
    <tr><td><code>--hidden</code></td><td><code>0</code></td><td>Random init: derive hidden size. With <code>--init-dense</code>: prune to this size</td></tr>
    <tr><td><code>--target</code></td><td><code>100000000</code></td><td>Random-init parameter target</td></tr>
    <tr><td><code>--context</code></td><td><code>64</code></td><td>Random-init context length</td></tr>
    <tr><td><code>--embed</code></td><td><code>128</code></td><td>Random-init embedding width</td></tr>
    <tr><td><code>--lr</code></td><td><code>0.02</code></td><td>Learning rate</td></tr>
    <tr><td><code>--epochs, -e</code></td><td><code>1</code></td><td>Training epochs</td></tr>
    <tr><td><code>--batch-size</code></td><td><code>128</code></td><td>Mini-batch size</td></tr>
    <tr><td><code>--max-samples</code></td><td><code>20000</code></td><td>Windows per epoch; <code>0</code> means all</td></tr>
  </tbody>
</table>
</div>
<div class="callout">
  <strong>How pruning works</strong>
  <p>
    <code>--hidden</code> with <code>--init-dense</code> ranks hidden units by the L2 norm of
    their outgoing <code>fc2</code> weights, keeps the strongest units in their original order,
    and fine-tunes the resulting sub-network. The saved file is fully dense, not low-rank.
  </p>
</div>
<pre><code># Fine-tune a 100M dense model from Turbo
nib dense-train --input shakespeare.txt --output dense-100m.bin \\
  --tokenizer tokenizer.json --init-turbo turbo.bin \\
  --epochs 2 --max-samples 20000 --batch-size 128 --lr 0.02

# Prune to 90M and fine-tune
nib dense-train --input shakespeare.txt --output nib.bin \\
  --tokenizer tokenizer.json --init-dense dense-100m.bin --hidden 10902 \\
  --epochs 3 --max-samples 20000 --batch-size 128 --lr 0.01</code></pre>

<h3><code>nib dense-generate</code></h3>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--model, -m</code></td><td><code>nib.bin</code></td><td>Dense 4-bit model</td></tr>
    <tr><td><code>--tokenizer</code></td><td><code>tokenizer.json</code></td><td>Tokenizer file</td></tr>
    <tr><td><code>--seed, -s</code></td><td><code>ROMEO: </code></td><td>Prompt text</td></tr>
    <tr><td><code>--length</code></td><td><code>300</code></td><td>Characters to generate</td></tr>
    <tr><td><code>--temperature</code></td><td><code>0.7</code></td><td>Sampling temperature</td></tr>
  </tbody>
</table>
</div>

<h2>Serving commands</h2>

<h3><code>nib serve</code></h3>
<p>Starts the HTTP API, CORS preflight, OpenAPI document, and browser UI. The model format is detected from the first eight bytes, so the same command works for <code>turbo.bin</code> and <code>nib.bin</code>.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--model, -m</code></td><td><code>turbo.bin</code></td><td>Turbo or dense model</td></tr>
    <tr><td><code>--tokenizer</code></td><td><code>tokenizer.json</code></td><td>Tokenizer file</td></tr>
    <tr><td><code>--host</code></td><td><code>0.0.0.0</code></td><td>Bind address; use <code>127.0.0.1</code> for local-only</td></tr>
    <tr><td><code>--port</code></td><td><code>8080</code></td><td>TCP port</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib serve --model nib.bin --tokenizer tokenizer.json --host 127.0.0.1 --port 8080
nib serve --model turbo.bin --tokenizer tokenizer.json --host 0.0.0.0 --port 8080</code></pre>

<h3><code>nib web-export</code></h3>
<p>Writes the embedded browser UI to a single HTML file. The page has no build step and no external assets; host it anywhere static and add <code>?api=https://your-host</code>.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody><tr><td><code>--output, -o</code></td><td><code>index.html</code></td><td>Destination HTML file</td></tr></tbody>
</table>
</div>
<pre><code>nib web-export --output index.html</code></pre>

<h2 id="vision-commands">Vision commands</h2>
<p>
  Full guide with a measured example, data rules, OCR limits, and troubleshooting:
  <a href="vision.html">Vision &amp; OCR</a>.
</p>

<h3><code>nib vision-train</code></h3>
<p>The manifest is one <code>image_path label</code> pair per line. Labels are zero-based integers.</p>
<pre><code>data/cats/001.jpg 0
data/dogs/002.jpg 1</code></pre>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--manifest, -i</code></td><td><code>manifest.txt</code></td><td>Image/label list</td></tr>
    <tr><td><code>--output, -o</code></td><td><code>vision.json</code></td><td>Output model</td></tr>
    <tr><td><code>--image-size</code></td><td><code>16</code></td><td>Square input size</td></tr>
    <tr><td><code>--channels</code></td><td><code>1</code></td><td>1 = grayscale, 3 = RGB</td></tr>
    <tr><td><code>--classes</code></td><td><code>10</code></td><td>Number of labels</td></tr>
    <tr><td><code>--lr</code></td><td><code>0.01</code></td><td>Learning rate</td></tr>
    <tr><td><code>--epochs, -e</code></td><td><code>10</code></td><td>Training epochs</td></tr>
    <tr><td><code>--batch-size</code></td><td><code>16</code></td><td>Mini-batch size</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib vision-train --manifest manifest.txt --output vision.json \\
  --classes 2 --image-size 32 --epochs 20</code></pre>
<p class="muted small">
  <code>--classes</code> must be greater than every label in the manifest. Out-of-range labels
  are silently ignored by the loss, so a manifest mistake can look like zero loss.
</p>

<h3><code>nib vision-predict</code></h3>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>--model, -m</code></td><td><code>vision.json</code></td><td>Trained CNN</td></tr>
    <tr><td><code>--image, -i</code></td><td>required</td><td>Image to classify</td></tr>
    <tr><td><code>--classes</code></td><td>none</td><td>Optional names, one per line</td></tr>
  </tbody>
</table>
</div>
<pre><code>nib vision-predict --model vision.json --image data/cats/new.jpg --classes classes.txt</code></pre>
<p class="muted small">
  Prints only the winning class and its confidence. The Rust API returns all class
  probabilities. Vision models are CLI/Rust only — <code>nib serve</code> accepts
  <code>CTURBO01</code> and <code>CDENSE4</code> text models, not <code>vision.json</code>.
</p>

<h2><code>nib version</code></h2>
<pre><code>nib version
# nib-rs 0.1.0</code></pre>

<h2>Recipes</h2>
<h3>Use your own corpus</h3>
<pre><code>nib train --input mydata.txt --output model.json --tokenizer my-tokenizer.json \\
  --context 24 --hidden 192 --embed 32 --epochs 60

nib turbo-train --input mydata.txt --output my-turbo.bin \\
  --tokenizer my-tokenizer.json --context 64 --embed 128 --rank 30 --epochs 5</code></pre>
<h3>Produce a 90M model under 50 MB from scratch</h3>
<pre><code>nib turbo-train --input shakespeare.txt --output turbo.bin --tokenizer tokenizer.json \\
  --context 64 --embed 128 --rank 30 --epochs 5

nib dense-build --turbo turbo.bin --output dense-100m.bin --tokenizer tokenizer.json

nib dense-train --input shakespeare.txt --output dense-100m.bin --tokenizer tokenizer.json \\
  --init-turbo turbo.bin --epochs 2 --max-samples 20000 --batch-size 128 --lr 0.02

nib dense-train --input shakespeare.txt --output nib.bin --tokenizer tokenizer.json \\
  --init-dense dense-100m.bin --hidden 10902 --epochs 3 --max-samples 20000 --batch-size 128 --lr 0.01

nib dense-info --hidden 10902 --tokenizer tokenizer.json</code></pre>
<h3>Resume dense training</h3>
<pre><code>nib dense-train --input shakespeare.txt --output nib.bin --tokenizer tokenizer.json \\
  --init-dense nib.bin --epochs 1 --max-samples 20000 --batch-size 128 --lr 0.005</code></pre>
<h3>Smoke test a server</h3>
<pre><code>nib serve --model nib.bin --tokenizer tokenizer.json --host 127.0.0.1 --port 8092
curl http://127.0.0.1:8092/health
curl http://127.0.0.1:8092/api/info</code></pre>
`,
  },
  {
    slug: "http-api",
    title: "HTTP API",
    description:
      "nib-rs HTTP API reference: health, model info, text generation, request limits, errors, CORS, OpenAPI, and curl, PowerShell, Python, and JavaScript clients.",
    body: `
<section class="hero">
  <p class="eyebrow">Reference</p>
  <h1>HTTP API</h1>
  <p class="lead">
    <code>nib serve</code> starts a zero-dependency HTTP/1.1 server with a JSON API, CORS, an
    OpenAPI document, and a browser UI. The same endpoints work for Turbo and dense models.
  </p>
</section>

<div class="callout warn">
  <strong>There is no public API endpoint out of the box</strong>
  <p>
    The hosted UI at <code>vamshavardhan50.github.io/nib-ai</code> is static and has no backend.
    Until you run or deploy <code>nib serve</code>, the only endpoint is on your own machine.
    This section covers a local URL, a temporary public URL, and a permanent HTTPS endpoint.
  </p>
</div>

<h2 id="endpoint">What is the endpoint?</h2>
<p>
  The base URL depends on where the Rust process runs. The generation path is always
  <code>/api/generate</code>.
</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Where the server runs</th><th>Base URL</th><th>Who can reach it</th></tr></thead>
  <tbody>
    <tr><td>Your machine (local only)</td><td><code>http://127.0.0.1:8080</code></td><td>Only that machine</td></tr>
    <tr><td>Your machine (LAN / Docker)</td><td><code>http://192.168.1.20:8080</code></td><td>Devices on the same network</td></tr>
    <tr><td>VPS or cloud service</td><td><code>https://api.example.com</code></td><td>Anyone on the internet</td></tr>
    <tr><td>Temporary tunnel</td><td><code>https://random-name.trycloudflare.com</code></td><td>Anyone, while the tunnel runs</td></tr>
  </tbody>
</table>
</div>
<p>
  <strong>Local generation endpoint:</strong> <code>POST http://127.0.0.1:8080/api/generate</code>.
  The static UI on GitHub Pages needs a public HTTPS base URL before it can generate anything.
</p>

<h2>Step-by-step: a local endpoint</h2>
<div class="steps">
  <div class="step">
    <div>
      <h4>1. Get the binary, model, and tokenizer</h4>
      <pre><code>git clone ${repo}.git
cd nib-ai
cargo build --release

# Windows: .\\build.ps1 -Release</code></pre>
      <p>
        The repository tracks both <code>nib.bin</code> (90M dense, 45.12 MB) and
        <code>turbo.bin</code> (994,656 bytes), plus <code>tokenizer.json</code>.
      </p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>2. Start the server</h4>
      <pre><code># macOS / Linux — dense model
./target/release/nib serve --model nib.bin --tokenizer tokenizer.json \\
  --host 127.0.0.1 --port 8080

# Windows PowerShell
.\\target\\release\\nib.exe serve --model nib.bin --tokenizer tokenizer.json \\
  --host 127.0.0.1 --port 8080</code></pre>
      <p>
        Use <code>turbo.bin</code> for a faster demo. Bind <code>127.0.0.1</code> for local-only
        access; bind <code>0.0.0.0</code> when a container, reverse proxy, or another device must
        reach the server.
      </p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>3. Confirm the endpoint</h4>
      <pre><code>curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/api/info</code></pre>
      <p>The server prints its own URLs at startup:</p>
      <pre><code>nib server listening on http://localhost:8080/
API: POST http://localhost:8080/api/generate
model: nib.bin (45117066 bytes, 45.117 MB), CORS enabled</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>4. Generate text</h4>
      <pre><code>curl -X POST http://127.0.0.1:8080/api/generate \\
  -H "content-type: application/json" \\
  -d '{"prompt":"ROMEO: ","length":80,"temperature":0.7}'</code></pre>
      <p>
        If this works, the endpoint is correct. The browser UI is at
        <code>http://127.0.0.1:8080/</code>.
      </p>
    </div>
  </div>
</div>

<h2 id="public-endpoint">Step-by-step: a public HTTPS endpoint</h2>
<p>
  A public endpoint is required for the hosted Pages UI, another website, or a mobile app.
  Choose one host, then follow the same six steps.
</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Host</th><th>Best for</th><th>HTTPS</th></tr></thead>
  <tbody>
    <tr><td>Render, Railway, Fly.io, Cloud Run</td><td>Managed Docker deploy, least server maintenance</td><td>Provided</td></tr>
    <tr><td>VPS + Caddy or nginx</td><td>Full control, persistent process, predictable cost</td><td>Let's Encrypt</td></tr>
    <tr><td>Cloudflare Tunnel / ngrok</td><td>A quick demo from your laptop</td><td>Provided</td></tr>
    <tr><td>GitHub Pages, Netlify, Vercel</td><td>Static UI and docs only</td><td>Cannot run the API</td></tr>
  </tbody>
</table>
</div>
<div class="steps">
  <div class="step">
    <div>
      <h4>1. Build once</h4>
      <pre><code>cargo build --release</code></pre>
      <p>You need three files: the <code>nib</code> binary, <code>nib.bin</code> or <code>turbo.bin</code>, and <code>tokenizer.json</code>.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>2. Copy the files to the host</h4>
      <pre><code>scp target/release/nib nib.bin tokenizer.json user@your-host:/opt/nib/</code></pre>
      <p>For Docker, use the included Dockerfile instead of copying files by hand.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>3. Run on all interfaces</h4>
      <pre><code>./nib serve --model nib.bin --tokenizer tokenizer.json \\
  --host 0.0.0.0 --port 8080</code></pre>
      <p>
        Keep the process alive with systemd, Docker, or the platform's process manager.
        <code>127.0.0.1</code> is not reachable from outside the machine.
      </p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>4. Put HTTPS in front</h4>
      <p>On a VPS, Caddy is the shortest path:</p>
      <pre><code>api.example.com {
    reverse_proxy 127.0.0.1:8080 {
        transport http {
            read_timeout 300s
            write_timeout 300s
        }
    }
}</code></pre>
      <p>Render, Railway, Fly.io, and Cloud Run terminate TLS for you and give you an HTTPS URL.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>5. Test from outside your network</h4>
      <pre><code>curl https://api.example.com/health
curl https://api.example.com/api/info

curl -X POST https://api.example.com/api/generate \\
  -H "content-type: application/json" \\
  -d '{"prompt":"ROMEO: ","length":40,"temperature":0.7}'</code></pre>
      <p>Test from a phone on mobile data, not just from the same machine.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>6. Connect the hosted UI</h4>
      <pre><code>https://vamshavardhan50.github.io/nib-ai/?api=https://api.example.com</code></pre>
      <p>
        The UI saves the endpoint in <code>localStorage</code> and includes it in the iframe
        snippet. The API must be HTTPS: an HTTPS page cannot call an HTTP endpoint.
      </p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>7. Lock it down</h4>
      <p>
        The built-in server has no authentication, TLS, or rate limiting, and CORS is
        <code>*</code>. Add auth and limits at the reverse proxy before making a dense model
        public. Prefer Turbo for open demos; it is roughly two orders of magnitude cheaper per
        character.
      </p>
    </div>
  </div>
</div>

<h3>Temporary public URL with a tunnel</h3>
<p>
  For a demo, you can expose your local server without deploying. The laptop must stay awake,
  the URL changes each run, and there is no SLA. Do not use this for production.
</p>
<pre><code># Terminal 1: local server
nib serve --model turbo.bin --tokenizer tokenizer.json --host 127.0.0.1 --port 8080

# Terminal 2: Cloudflare Tunnel (no account needed for a quick tunnel)
cloudflared tunnel --url http://127.0.0.1:8080
# prints: https://random-name.trycloudflare.com</code></pre>
<p>Then open:</p>
<pre><code>https://vamshavardhan50.github.io/nib-ai/?api=https://random-name.trycloudflare.com</code></pre>
<p>
  <code>ngrok http 8080</code> works the same way. Both provide HTTPS, so the static UI can
  call them without mixed-content errors.
</p>

<h2>How the UI chooses an endpoint</h2>
<ol>
  <li>The <code>?api=</code> query parameter, if present.</li>
  <li>The <code>nib-api</code> value saved in <code>localStorage</code> from a previous visit.</li>
  <li>The page's own origin, when it is served by <code>nib serve</code> on a non-static host.</li>
  <li>Otherwise, an empty endpoint. On GitHub Pages the UI shows “enter an API endpoint”.</li>
</ol>
<p>Trailing slashes are removed. The API base is stored exactly as entered.</p>

<h2>Endpoint troubleshooting</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Symptom</th><th>Cause and fix</th></tr></thead>
  <tbody>
    <tr><td>UI says “enter an API endpoint”</td><td>You are on a static host with no backend. Start or deploy the API and add <code>?api=</code>.</td></tr>
    <tr><td>Works on the server, not from your phone</td><td>The server is bound to <code>127.0.0.1</code>. Restart with <code>--host 0.0.0.0</code> and allow the port in the firewall.</td></tr>
    <tr><td><code>Failed to fetch</code> / CORS error</td><td>Wrong URL, server down, or an HTTPS page calling HTTP. Use the exact HTTPS base URL.</td></tr>
    <tr><td><code>404</code> on generation</td><td>Use <code>POST /api/generate</code>, not <code>GET</code>, and include <code>/api</code>.</td></tr>
    <tr><td>Connection refused in Docker</td><td>Bind <code>0.0.0.0</code> inside the container and publish the port: <code>-p 8080:8080</code>.</td></tr>
    <tr><td>Request times out on dense</td><td>Raise the proxy timeout to 300 s and shorten <code>length</code>. Use Turbo for interactive traffic.</td></tr>
    <tr><td><code>413</code> or <code>prompt is limited</code></td><td>Body limit is 64 KB and prompts are capped at 4,096 characters.</td></tr>
    <tr><td>Health works but generation is slow</td><td>Expected for the 90M dense model on CPU. Serve <code>turbo.bin</code> for a fast demo.</td></tr>
  </tbody>
</table>
</div>

<h2>Endpoints</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Method</th><th>Path</th><th>Purpose</th></tr></thead>
  <tbody>
    <tr><td><code>GET</code></td><td><code>/health</code></td><td>Liveness probe</td></tr>
    <tr><td><code>GET</code></td><td><code>/api/info</code></td><td>Model name, format, parameter counts, config, file size, endpoint map</td></tr>
    <tr><td><code>POST</code></td><td><code>/api/generate</code></td><td>Generate text</td></tr>
    <tr><td><code>GET</code></td><td><code>/openapi.json</code></td><td>OpenAPI 3.1 description</td></tr>
    <tr><td><code>GET</code></td><td><code>/</code></td><td>Embedded browser UI</td></tr>
    <tr><td><code>OPTIONS</code></td><td>any path</td><td>CORS preflight; responds <code>204</code></td></tr>
    <tr><td><code>GET</code></td><td><code>/favicon.ico</code></td><td>Empty <code>204</code></td></tr>
  </tbody>
</table>
</div>
<p>All JSON responses use <code>application/json; charset=utf-8</code>. Unknown paths return <code>404</code> with a list of valid paths.</p>

<h2>CORS</h2>
<p>
  Cross-origin requests are allowed from any origin. The server answers <code>OPTIONS</code>
  preflight with <code>204 No Content</code> and includes:
</p>
<pre><code>Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: content-type</code></pre>
<p>This is what lets the static UI on GitHub Pages, Netlify, Vercel, or Cloudflare Pages call a separately hosted Rust API.</p>

<h2>GET /health</h2>
<pre><code>curl http://127.0.0.1:8080/health</code></pre>
<pre><code>{"status":"ok"}</code></pre>

<h2>GET /api/info</h2>
<h3>Dense model response</h3>
<pre><code>{
  "name": "DenseLM",
  "format": "dense-int4",
  "architecture": "dense MLP, 4-bit weights on disk",
  "logical_parameters": 90004008,
  "stored_parameters": 90004008,
  "file_bytes": 45117066,
  "config": {
    "vocab_size": 62,
    "embed_dim": 128,
    "hidden_dim": 10902,
    "context": 64
  },
  "endpoints": {
    "health": "/health",
    "info": "/api/info",
    "generate": "/api/generate",
    "openapi": "/openapi.json"
  }
}</code></pre>
<h3>Turbo model response</h3>
<pre><code>{
  "name": "TurboLM",
  "format": "turbo-int8",
  "architecture": "low-rank MLP, int8 weights on disk",
  "logical_parameters": 100009068,
  "stored_parameters": 994572,
  "file_bytes": 994656,
  "config": {
    "vocab_size": 62,
    "embed_dim": 128,
    "hidden_dim": 12114,
    "context": 64,
    "rank": 30
  },
  "endpoints": { "...": "..." }
}</code></pre>
<p>
  <code>logical_parameters</code> is the dense shape; <code>stored_parameters</code> is what is
  actually written to disk. For DenseLM the two are equal because the artifact is fully dense.
</p>

<h2>POST /api/generate</h2>
<h3>Request body</h3>
<div class="table-wrap">
<table>
  <thead><tr><th>Field</th><th>Type</th><th>Default</th><th>Rules</th></tr></thead>
  <tbody>
    <tr><td><code>prompt</code></td><td>string</td><td>Shakespeare line</td><td>Optional; at most 4,096 characters</td></tr>
    <tr><td><code>length</code></td><td>integer</td><td><code>300</code></td><td>Clamped to <code>1..2000</code> characters</td></tr>
    <tr><td><code>temperature</code></td><td>number</td><td><code>0.8</code></td><td>Clamped to <code>0.01..2.0</code>; non-finite values become <code>0.8</code></td></tr>
  </tbody>
</table>
</div>
<p>The body is optional. An empty body uses the default prompt and settings. Unknown JSON fields are ignored by serde.</p>
<h3>Response</h3>
<pre><code>{
  "text": "First Citizen:\\nBefore we proceed any further, hear me speak.\\n\\nSICINIUS:\\nBut, so ma",
  "prompt": "First Citizen:\\nBefore we proceed any further, hear me speak.\\n\\n",
  "length": 120,
  "temperature": 0.7,
  "ms": 2787
}</code></pre>
<p>
  <code>text</code> includes the prompt followed by the generated characters. <code>ms</code>
  is server-side generation time. The response is sent after generation completes; there is no
  streaming.
</p>

<h3>Prompt behavior</h3>
<p>
  Both text models use a fixed 64-character context. If the prompt is shorter, it is repeated
  to fill the context rather than padded with spaces. If it is longer, the last 64 characters
  are used. Natural prompts of 64+ characters produce the most coherent text.
</p>

<h3>Examples</h3>
<h4>curl</h4>
<pre><code>curl -X POST http://127.0.0.1:8080/api/generate \\
  -H "content-type: application/json" \\
  -d '{"prompt":"First Citizen: Before we proceed any further, hear me speak.","length":120,"temperature":0.7}'</code></pre>
<h4>PowerShell</h4>
<pre><code>$body = @{ prompt = "ROMEO: "; length = 80; temperature = 0.7 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/generate \\
  -ContentType "application/json" -Body $body</code></pre>
<h4>Python</h4>
<pre><code>import requests

response = requests.post(
    "http://127.0.0.1:8080/api/generate",
    json={
        "prompt": "First Citizen: Before we proceed any further, hear me speak.",
        "length": 120,
        "temperature": 0.7,
    },
    timeout=300,
)
response.raise_for_status()
print(response.json()["text"])</code></pre>
<h4>JavaScript (browser or Node 18+)</h4>
<pre><code>const response = await fetch("http://127.0.0.1:8080/api/generate", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    prompt: "First Citizen: Before we proceed any further, hear me speak.",
    length: 120,
    temperature: 0.7
  })
});
if (!response.ok) throw new Error("HTTP " + response.status);
const data = await response.json();
console.log(data.text);</code></pre>

<h2>Errors</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Status</th><th>When</th><th>Body</th></tr></thead>
  <tbody>
    <tr><td><code>400</code></td><td>Malformed request line or invalid JSON</td><td><code>{"error":"invalid JSON: ..."}</code></td></tr>
    <tr><td><code>400</code></td><td>Prompt over 4,096 characters</td><td><code>{"error":"prompt is limited to 4096 characters"}</code></td></tr>
    <tr><td><code>404</code></td><td>Unknown path</td><td><code>{"error":"not found","paths":[...]}</code></td></tr>
    <tr><td><code>405</code></td><td><code>GET /api/generate</code></td><td><code>{"error":"use POST /api/generate with a JSON body"}</code></td></tr>
    <tr><td><code>413</code></td><td>Body over 64 KB</td><td><code>{"error":"request body is too large"}</code></td></tr>
    <tr><td><code>431</code></td><td>Headers over 32 KB</td><td><code>{"error":"headers too large"}</code></td></tr>
  </tbody>
</table>
</div>

<h2>Limits and concurrency</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Limit</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Request body</td><td>64 KB</td></tr>
    <tr><td>Request headers</td><td>32 KB</td></tr>
    <tr><td>Prompt</td><td>4,096 characters</td></tr>
    <tr><td>Generated length</td><td>1–2,000 characters</td></tr>
    <tr><td>Read timeout</td><td>30 seconds</td></tr>
    <tr><td>Write timeout</td><td>300 seconds</td></tr>
    <tr><td>Connections</td><td>One thread per connection</td></tr>
    <tr><td>Generation</td><td>Serialized by a model mutex</td></tr>
  </tbody>
</table>
</div>
<p>
  Health checks are cheap, but generation holds the model lock for the whole request. On a
  dense model, a long request can occupy the server for tens of seconds; plan your proxy
  timeouts accordingly. There is no request queue, rate limit, authentication, or TLS in the
  built-in server.
</p>

<h2>OpenAPI</h2>
<p><code>GET /openapi.json</code> returns an OpenAPI 3.1 document describing the health, info, and generation endpoints. Point Swagger UI, Redoc, or a client generator at it.</p>
<pre><code>curl http://127.0.0.1:8080/openapi.json</code></pre>

<h2>Embed the UI</h2>
<p>
  Export the UI with <code>nib web-export</code>, host the HTML on any static host, and set the
  API base with a query parameter:
</p>
<pre><code>https://your-static-host/?api=https://your-nib-api-host</code></pre>
<p>The UI stores the endpoint in <code>localStorage</code> and includes it in the generated iframe snippet.</p>
<pre><code>&lt;iframe src="https://your-static-host/?api=https://your-nib-api-host"
        width="100%" height="760"
        style="border:0;border-radius:16px"
        title="nib-rs language model"&gt;&lt;/iframe&gt;</code></pre>

<h2>Production notes</h2>
<ul>
  <li><strong>Put it behind a reverse proxy</strong> for TLS, authentication, rate limiting, and request logging. The built-in server is intentionally minimal.</li>
  <li><strong>Raise proxy timeouts</strong> for dense generation. A 300-second upstream write timeout is already set in Rust; Caddy, nginx, or a cloud load balancer may be shorter.</li>
  <li><strong>Use Turbo for interactive traffic.</strong> Dense is the artifact to inspect and benchmark, not the fastest chat backend.</li>
  <li><strong>Memory:</strong> dense needs roughly 400 MB of RAM for f32 weights plus activations; allow 1 GB in containers.</li>
  <li><strong>No streaming:</strong> clients receive the full text in one JSON response.</li>
</ul>
`,
  },
  {
    slug: "rust-api",
    title: "Rust API",
    description:
      "Use nib-rs as a Rust crate: tokenizer, CharLM training, Turbo and dense models, quantization, generation, vision, and the HTTP server.",
    body: `
<section class="hero">
  <p class="eyebrow">Library</p>
  <h1>Rust API</h1>
  <p class="lead">
    Nib is both a CLI and a library. Add <code>nib-rs</code> to a project to train, quantize,
    load, and generate with the same code paths the binary uses.
  </p>
</section>

<h2>Add the dependency</h2>
<pre><code>[dependencies]
nib-rs = { path = "../nib-ai" }
anyhow = "1"</code></pre>
<p>Or from Git:</p>
<pre><code>[dependencies]
nib-rs = { git = "${repo}.git" }
anyhow = "1"</code></pre>
<p>The library crate name is <code>nib_rs</code> (Cargo converts the hyphen).</p>

<h2>Module map</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Module</th><th>Contents</th></tr></thead>
  <tbody>
    <tr><td><code>model</code></td><td><code>Tokenizer</code>, <code>ModelConfig</code>, <code>CharLM</code>, <code>Trainer</code></td></tr>
    <tr><td><code>turbo</code></td><td><code>TurboConfig</code>, <code>TurboLM</code>, <code>TurboTrainer</code>, <code>LowRankLinear</code>, <code>generate_text</code>, <code>sample</code></td></tr>
    <tr><td><code>dense</code></td><td><code>params</code>, <code>quantized_bytes</code>, <code>from_turbo</code>, <code>prune_hidden</code>, <code>save_int4</code>, <code>load_int4</code>, <code>DenseTrainer</code>, <code>generate_text</code></td></tr>
    <tr><td><code>layers</code></td><td><code>Init</code>, <code>Linear</code>, <code>Embedding</code></td></tr>
    <tr><td><code>tensor</code></td><td><code>Tensor1</code>, <code>Tensor2</code>, <code>Tensor4</code> wrappers</td></tr>
    <tr><td><code>optimizer</code></td><td><code>SGD</code>, <code>Adam</code> configuration structs</td></tr>
    <tr><td><code>vision</code></td><td><code>Conv2d</code>, <code>MaxPool2d</code>, <code>VisionCNN</code>, <code>VisionTrainer</code>, <code>load_image</code>, <code>load_manifest</code></td></tr>
    <tr><td><code>web</code></td><td><code>serve</code>, <code>index_html</code></td></tr>
  </tbody>
</table>
</div>

<h2>Tokenizer</h2>
<p>The tokenizer is character-level. It sorts and deduplicates every character in the training text, then maps each one to an index.</p>
<pre><code>use nib_rs::Tokenizer;

let text = std::fs::read_to_string("shakespeare.txt")?;
let tok = Tokenizer::from_text(&text);
println!("vocabulary: {} characters", tok.vocab_size());
tok.save("tokenizer.json")?;

let ids = tok.encode("ROMEO: ");
let back = tok.decode(&ids);
assert_eq!(back, "ROMEO: ");</code></pre>
<p>Unknown characters map to index 0 when encoding. Keep the tokenizer that was built with a model; vocab size mismatches are rejected by the CLI and server.</p>

<h2>CharLM: train and sample</h2>
<pre><code>use nib_rs::{CharLM, ModelConfig, Tokenizer, Trainer};

fn main() -&gt; anyhow::Result&lt;()&gt; {
    let text = std::fs::read_to_string("shakespeare.txt")?;
    let tok = Tokenizer::from_text(&text);
    tok.save("tokenizer.json")?;

    let config = ModelConfig {
        vocab_size: tok.vocab_size(),
        context: 16,
        embed_dim: 32,
        hidden_dim: 128,
        ..Default::default()
    };
    let mut trainer = Trainer::new(CharLM::new(config), "sgd".into(), 0.1);

    let ids = tok.encode(&text);
    let context = trainer.model.config.context;
    let mut windows = Vec::new();
    for i in 0..ids.len().saturating_sub(context) {
        windows.push((ids[i..i + context].to_vec(), ids[i + context]));
    }

    for epoch in 0..5 {
        let mut loss = 0.0;
        let mut batches = 0;
        for batch in windows.chunks(32) {
            loss += trainer.train_step(batch);
            batches += 1;
        }
        println!("epoch {epoch}: {:.4}", loss / batches as f32);
    }
    trainer.save("model.json")?;
    Ok(())
}</code></pre>
<p>Sampling uses the model's own <code>sample</code> helper:</p>
<pre><code>use nib_rs::{CharLM, Tokenizer};

let mut model = CharLM::load("model.json")?;
let tok = Tokenizer::load("tokenizer.json")?;
let context = model.config.context;

let mut ctx = tok.encode("ROMEO: ");
if ctx.len() &gt; context {
    ctx = ctx[ctx.len() - context..].to_vec();
}
while ctx.len() &lt; context {
    ctx.insert(0, tok.char_to_id.get(&amp;' ').copied().unwrap_or(0));
}

let mut out = String::from("ROMEO: ");
for _ in 0..200 {
    let probs = model.predict(&amp;ctx);
    let next = CharLM::sample(&amp;probs, 0.7);
    out.push(tok.id_to_char.get(&amp;next).copied().unwrap_or('?'));
    ctx.remove(0);
    ctx.push(next);
}
println!("{out}");</code></pre>

<h2>TurboLM: low-rank training</h2>
<pre><code>use nib_rs::{turbo, Tokenizer, TurboConfig, TurboLM, TurboTrainer};

let tok = Tokenizer::load("tokenizer.json")?;
let config = TurboConfig {
    vocab_size: tok.vocab_size(),
    context: 64,
    embed_dim: 128,
    hidden_dim: 12114,
    rank: 30,
};
let model = TurboLM::new(config);
println!("logical: {}", model.logical_params());
println!("stored:  {}", model.stored_params());

let mut trainer = TurboTrainer::new(model, 0.05);
// let loss = trainer.train_step(&amp;batch);
trainer.save("turbo.bin")?;

let mut model = TurboLM::load("turbo.bin")?;
let text = turbo::generate_text(&amp;mut model, &amp;tok, "ROMEO: ", 200, 0.7);
println!("{text}");</code></pre>
<p>
  <code>TurboTrainer::train_step</code> accepts <code>&amp;[(Vec&lt;usize&gt;, usize)]</code>:
  one context window and one next-token target per sample. It processes the batch as matrices,
  which is much faster than a per-sample loop.
</p>

<h2>DenseLM: load, generate, prune, train</h2>
<pre><code>use nib_rs::{dense, Tokenizer};

let tok = Tokenizer::load("tokenizer.json")?;

// Load a 4-bit dense artifact (nib.bin or any dense file).
let mut model = dense::load_int4("nib.bin")?;
println!("params: {}", dense::params(&amp;model.config));
println!("file bytes: {}", dense::quantized_bytes(&amp;model.config));

let text = dense::generate_text(&amp;mut model, &amp;tok, "ROMEO: ", 120, 0.7);
println!("{text}");

// Prune hidden units by outgoing fc2 weight norm.
let smaller = dense::prune_hidden(&amp;model, 10902)?;
dense::save_int4(&amp;smaller, "nib-90m.bin")?;</code></pre>
<p>Fine-tune a dense model with <code>DenseTrainer</code>:</p>
<pre><code>use nib_rs::dense::{self, DenseTrainer};

let mut trainer = DenseTrainer::new(dense::load_int4("nib.bin")?, 0.01);
let batch: Vec&lt;(Vec&lt;usize&gt;, usize)&gt; = vec![
    (vec![0; 64], 1),
];
let loss = trainer.train_step(&amp;batch);
println!("loss: {loss:.4}");
trainer.save("nib.bin")?;</code></pre>
<div class="callout warn">
  <strong>Memory</strong>
  <p>
    Loading a dense model expands 4-bit weights into f32 matrices. A 90M model needs roughly
    360 MB for weights alone, plus activations and gradients during training. Training holds
    multiple copies of the large matrices; allow well over 1 GB for comfortable dense
    fine-tuning.
  </p>
</div>

<h2>Build dense from Turbo</h2>
<pre><code>use nib_rs::{dense, TurboLM};

let turbo = TurboLM::load("turbo.bin")?;
let full = dense::from_turbo(&amp;turbo);
assert_eq!(full.config.hidden_dim, turbo.config.hidden_dim);
dense::save_int4(&amp;full, "dense-100m.bin")?;</code></pre>
<p>
  <code>from_turbo</code> materializes <code>W = U·V</code> for both low-rank layers. The result
  is a normal dense <code>CharLM</code>; predictions match the Turbo model up to f32 rounding.
</p>

<h2>VisionCNN</h2>
<pre><code>use nib_rs::{load_image, load_manifest, VisionCNN, VisionTrainer};

let entries = load_manifest("manifest.txt")?;
let (path, label) = &amp;entries[0];
let image = load_image(path, 32, 32, 3)?;

let model = VisionCNN::new(3, 32, 32, 2);
let mut trainer = VisionTrainer::new(model, 0.01);
let loss = trainer.train_step(&amp;[image], &amp;[*label]);
println!("loss: {loss:.4}");
trainer.save("vision.json")?;</code></pre>
<p>Manifests contain one <code>path label</code> pair per line. <code>load_image</code> resizes to the requested height/width, converts to the requested channel count, and normalizes pixels.</p>

<h2>Serve from Rust</h2>
<pre><code>nib_rs::web::serve("nib.bin", "tokenizer.json", "127.0.0.1", 8080)?;</code></pre>
<p>
  <code>serve</code> blocks forever and auto-detects Turbo or dense models. To embed the UI in
  another service, call <code>nib_rs::web::index_html()</code> for the static HTML string.
</p>

<h2>File formats</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Format</th><th>Magic</th><th>Precision</th><th>Notes</th></tr></thead>
  <tbody>
    <tr><td>Turbo</td><td><code>CTURBO01</code></td><td>int8 per matrix</td><td>Low-rank factors, biases, embedding</td></tr>
    <tr><td>Dense</td><td><code>CDENSE4\\0</code></td><td>4-bit + one f32 row scale</td><td>Version 1; full dense matrices</td></tr>
    <tr><td>CharLM</td><td>JSON</td><td>f32</td><td>Human-readable, easy to inspect</td></tr>
    <tr><td>VisionCNN</td><td>JSON</td><td>f32</td><td>Human-readable</td></tr>
  </tbody>
</table>
</div>
<p>
  Dense quantization stores each weight as a signed 4-bit nibble in <code>-7..7</code>, scaled
  by <code>max_abs / 7</code> per row. Biases stay f32. The header records dimensions so a
  mismatched file is rejected before inference. Formats are versioned; treat them as internal
  unless you pin a version.
</p>

<h2>Threading</h2>
<p>
  <code>predict</code>, <code>train_step</code>, and <code>generate_text</code> take
  <code>&amp;mut self</code>. Wrap a model in <code>std::sync::Mutex</code> to share it across
  threads. The built-in server does exactly that, which serializes generation while still
  allowing concurrent connections.
</p>
<pre><code>use std::sync::{Arc, Mutex};

let model = Arc::new(Mutex::new(nib_rs::dense::load_int4("nib.bin")?));
let shared = Arc::clone(&amp;model);
std::thread::spawn(move || {
    let mut model = shared.lock().unwrap();
    // model.predict(...)
});</code></pre>

<h2>Docs and tests</h2>
<pre><code>cargo doc --open --no-deps
cargo test
cargo test --doc</code></pre>
<p>
  The test suite covers tensor indexing, layer shapes, tokenizer roundtrips, training steps,
  Turbo and dense save/load roundtrips, quantization prediction stability, pruning, vision
  forward/backward passes, HTTP request parsing, CORS/UI wiring, and an in-process generate
  roundtrip.
</p>
`,
  },
  {
    slug: "vision",
    title: "Vision & OCR",
    description:
      "nib-rs vision guide: train a from-scratch CNN on labeled images, classify photos, recognize single cropped characters, and understand why full OCR is not built in.",
    body: `
<section class="hero">
  <p class="eyebrow">Vision</p>
  <h1>Image classification — and the honest OCR story</h1>
  <p class="lead">
    Nib includes a from-scratch convolutional network that classifies a whole image into one
    label. It can recognize a single, cleanly cropped character. It is <strong>not</strong> an
    OCR engine: it cannot find text in a photo or read a line of characters.
  </p>
  <div class="stats">
    <div class="stat"><b>95.33%</b><span>held-out handwritten digits</span></div>
    <div class="stat"><b>13.5 s</b><span>to train 30 epochs on 1,497 images</span></div>
    <div class="stat"><b>49,200</b><span>bytes for the 10-class model</span></div>
    <div class="stat"><b>1,898</b><span>weights at 16×16, 10 classes</span></div>
  </div>
</section>

<div class="callout warn">
  <strong>OCR is not built in</strong>
  <p>
    There is no text detection, no line segmentation, and no sequence decoder. The CNN returns
    exactly one label and one confidence for the entire image. A photo of a page, a receipt, or
    a sentence is <em>one</em> input, not a string of characters. The closest supported workflow
    is a single-glyph classifier: you crop each character yourself, then classify the crops one
    by one. That is documented below, with its limits.
  </p>
</div>

<h2 id="what-it-is">What the vision model is</h2>
<p>
  <code>VisionCNN</code> is a small convolutional classifier written from scratch in
  <code>src/vision.rs</code>. It starts from random weights, learns from labeled images, and
  stores its weights as JSON. There is no pre-trained backbone, no ONNX, and no Python.
</p>
<pre><code>image (C × H × W, pixels normalized to 0..1)
  → Conv2d(C → 8, 3×3) → ReLU → MaxPool(2×2)
  → Conv2d(8 → 16, 3×3) → ReLU → MaxPool(2×2)
  → flatten → Linear(features → classes) → softmax</code></pre>
<div class="table-wrap">
<table>
  <thead><tr><th>Property</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Task</td><td>Single-label image classification (one label per image)</td></tr>
    <tr><td>Input</td><td>Square image, <code>16×16</code> grayscale by default; configurable size and RGB</td></tr>
    <tr><td>Layers</td><td>2 convolution layers (8 and 16 filters), 2 max-pool layers, 1 linear layer</td></tr>
    <tr><td>Activation</td><td>ReLU in convolutions, softmax at the output</td></tr>
    <tr><td>Loss</td><td>Cross-entropy</td></tr>
    <tr><td>Optimizer</td><td>Plain SGD with a fixed learning rate</td></tr>
    <tr><td>Weights</td><td>1,898 for 10 classes at 16×16; grows with image size and class count</td></tr>
    <tr><td>Model file</td><td><code>vision.json</code>, 49,200 bytes for the 10-class example</td></tr>
    <tr><td>Training speed</td><td>13.5 s for 30 epochs on 1,497 images at 16×16 (CPU, single thread)</td></tr>
    <tr><td>Interfaces</td><td>CLI (<code>vision-train</code>, <code>vision-predict</code>) and the Rust API</td></tr>
    <tr><td>Not included</td><td>HTTP endpoint, webcam capture, object detection, OCR, GPU training</td></tr>
  </tbody>
</table>
</div>
<p class="muted small">
  The model is deliberately tiny. It is suitable for narrow, controlled datasets — digits,
  symbols, shapes, colors, simple defects, cropped objects — not ImageNet-scale recognition.
</p>

<h2 id="what-you-can-do">What you can do with it</h2>

<h3>1. Binary image classification</h3>
<p>
  Sort images into two classes: cat vs dog, pass vs fail, defective vs clean, day vs night,
  red vs blue. Use <code>--classes 2</code> and labels <code>0</code> and <code>1</code>.
  This is the easiest case and needs the least data.
</p>

<h3>2. Small multi-class classification</h3>
<p>
  Recognize digits 0–9, shapes, symbols, traffic-sign crops, product categories, or meter
  readings. Use one label per class and <code>--classes N</code>. A few hundred images per
  class is a reasonable starting point.
</p>

<h3>3. Single-character recognition (the OCR-adjacent case)</h3>
<p>
  Train a 62-class model for <code>0-9</code>, <code>A-Z</code>, <code>a-z</code> on
  <strong>pre-cropped, fixed-font glyphs</strong>. Then crop characters from your image with an
  external tool and classify each crop. This works for clean, uniform text such as meter digits,
  CAPTCHA-style glyphs, or a single known font. It is not general OCR. See
  <a href="#ocr">the OCR section</a> for the exact recipe and caveats.
</p>

<h3>4. Simple visual inspection</h3>
<p>
  Scratch vs no scratch, filled vs empty, aligned vs misaligned. Keep the camera distance,
  lighting, and framing consistent across training and prediction images; the model has no
  invariance tricks beyond pooling.
</p>

<h3>5. Learn how a CNN trains</h3>
<p>
  Every forward pass, convolution, pooling window, gradient, and weight update is readable in
  <code>src/vision.rs</code>. The model is small enough to train in seconds and inspect by hand.
</p>

<h2 id="quickstart">Quick start: train a digit classifier</h2>
<p>
  The numbers in this section are <strong>measured, not illustrative</strong>. The dataset is the
  UCI handwritten digits set (the copy shipped with scikit-learn): 1,797 images at 8×8, upscaled
  to 16×16, split into 1,497 training and 300 held-out images. Any folder of labeled images works
  the same way.
</p>

<div class="steps">
  <div class="step">
    <div>
      <h4>1. Organize the images</h4>
      <pre><code>digits/
  train/
    digit_0001_1.png
    digit_0002_0.png
    ...
  test/
    digit_0000_0.png
    ...</code></pre>
      <p>
        Nib does not require this folder structure. It only reads the manifest, so images can
        live anywhere. The split is yours to make: the trainer does not create one.
      </p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>2. Write the manifest and class names</h4>
      <pre><code># manifest.txt — one "path label" pair per line
digits/train/digit_0001_1.png 1
digits/train/digit_0002_0.png 0
digits/test/digit_0000_0.png 0</code></pre>
      <pre><code># classes.txt — one display name per line, in label order
0
1
2
3
4
5
6
7
8
9</code></pre>
      <p>
        Blank lines and lines starting with <code>#</code> are ignored. The label is always the
        <strong>last</strong> token, so paths containing spaces are fine.
      </p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>3. Train</h4>
      <pre><code>nib vision-train --manifest manifest.txt --output vision.json \\
  --image-size 16 --channels 1 --classes 10 \\
  --epochs 30 --batch-size 16 --lr 0.01</code></pre>
      <p>Real output from this exact run:</p>
      <pre><code>vision training: 1497 images, 10 classes, 1x16x16, epochs=30, lr=0.01, batch=16
epoch 0/30 loss: 2.1589
epoch 1/30 loss: 1.6572
epoch 2/30 loss: 1.2281
...
epoch 28/30 loss: 0.1301
epoch 29/30 loss: 0.1267
vision model saved to vision.json</code></pre>
      <p>
        The whole run took <strong>13.5 seconds</strong>. The model file is
        <strong>49,200 bytes</strong>.
      </p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>4. Predict one image</h4>
      <pre><code>nib vision-predict --model vision.json \\
  --image digits/test/digit_0000_0.png --classes classes.txt</code></pre>
      <pre><code>prediction: 0 (class 0)
confidence: 0.9998</code></pre>
      <p>
        The CLI prints only the top-1 label. The Rust API returns the full probability vector if
        you need per-class scores.
      </p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>5. Measure held-out accuracy</h4>
      <p>
        Run <code>vision-predict</code> over images the model never trained on and compare with
        the manifest labels. On the 300 held-out digits, this model scored
        <strong>286/300 = 95.33%</strong>. Each CLI invocation, including process start and model
        load, averaged <strong>63.7 ms</strong>.
      </p>
    </div>
  </div>
</div>

<div class="callout">
  <strong>Why 16×16?</strong>
  <p>
    Small inputs train fast and keep the model tiny. They also throw away detail. Use 32×32 or
    48×48 when the distinguishing feature is fine — font shape, small defects, thin strokes.
    The default 16×16 is fine for bold digits and simple shapes.
  </p>
</div>

<h2 id="manifest">Manifest and data rules</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Rule</th><th>Detail</th></tr></thead>
  <tbody>
    <tr><td>Format</td><td><code>image_path label</code>, one pair per line; label is a zero-based integer</td></tr>
    <tr><td>Comments</td><td>Blank lines and lines starting with <code>#</code> are skipped</td></tr>
    <tr><td>Spaces in paths</td><td>Supported, because the label is parsed from the end of the line</td></tr>
    <tr><td>Label range</td><td>Must be <code>0 .. classes-1</code>. Out-of-range labels are silently ignored by the loss, which can look like zero loss — validate your manifest.</td></tr>
    <tr><td>Image formats</td><td>Anything the Rust <code>image</code> crate decodes: PNG, JPEG, BMP, GIF, TIFF, WebP, and more</td></tr>
    <tr><td>Resizing</td><td>Every image is resized to <code>image-size × image-size</code> with a triangle filter</td></tr>
    <tr><td>Channels</td><td><code>--channels 1</code> converts to grayscale; <code>--channels 3</code> keeps RGB</td></tr>
    <tr><td>Normalization</td><td>Pixels are divided by 255; grayscale uses 0.299R + 0.587G + 0.114B</td></tr>
    <tr><td>Class names</td><td>Optional <code>classes.txt</code>, one name per line; line N names class N</td></tr>
    <tr><td>Split</td><td>Not automatic. Hold out a test manifest yourself.</td></tr>
    <tr><td>Augmentation</td><td>None. Rotation, blur, brightness, and crop jitter are not applied.</td></tr>
  </tbody>
</table>
</div>

<h2 id="flags">Choosing the flags</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Flag</th><th>Default</th><th>Guidance</th></tr></thead>
  <tbody>
    <tr><td><code>--image-size</code></td><td><code>16</code></td><td>Minimum 10. Use 16 for bold shapes, 32 for fonts and objects, 48 for fine detail. Cost grows roughly with the square of the size.</td></tr>
    <tr><td><code>--channels</code></td><td><code>1</code></td><td>Use 3 when color is the signal (ripe/unripe, red/blue, stained/clean). Use 1 for shape and text.</td></tr>
    <tr><td><code>--classes</code></td><td><code>10</code></td><td>Must be greater than every label in the manifest.</td></tr>
    <tr><td><code>--lr</code></td><td><code>0.01</code></td><td>Lower to 0.003 if the loss oscillates; raise to 0.03 if it barely moves.</td></tr>
    <tr><td><code>--epochs</code></td><td><code>10</code></td><td>Watch the loss. 20–40 is common for small datasets; stop when it flattens.</td></tr>
    <tr><td><code>--batch-size</code></td><td><code>16</code></td><td>Reduce if memory is tight; raise for smoother gradients.</td></tr>
  </tbody>
</table>
</div>
<p>
  Weight count at 16×16 is <code>80 + 1,168 + (64 × classes + classes)</code>. For 10 classes
  that is <strong>1,898</strong> weights. At 32×32 the feature map is 576, so 10 classes need
  <strong>7,018</strong> weights. The model stays tiny either way; the cost is training time and
  image loading, not file size.
</p>

<h2 id="ocr">OCR: what is missing and what you can build</h2>
<p>
  Full OCR is a pipeline, and Nib currently implements only one small piece of it. This is the
  honest breakdown.
</p>
<div class="grid two">
  <div class="card">
    <h3>What Nib gives you</h3>
    <ul>
      <li>A classifier: one image in, one label out.</li>
      <li>Convolution and pooling learned from your data.</li>
      <li>Top-1 confidence for a single glyph.</li>
      <li>A 49 KB model that trains in seconds.</li>
      <li>CLI and Rust interfaces.</li>
    </ul>
  </div>
  <div class="card">
    <h3>What real OCR needs</h3>
    <ul>
      <li><strong>Detection</strong>: find text regions in a page or photo.</li>
      <li><strong>Segmentation</strong>: split lines, words, and touching glyphs.</li>
      <li><strong>Sequence recognition</strong>: map a whole line to a character sequence (CNN + RNN + CTC, or an encoder-decoder).</li>
      <li><strong>Decoding</strong>: turn per-frame probabilities into text.</li>
      <li><strong>Language context</strong>: fix ambiguous characters using words.</li>
    </ul>
  </div>
</div>
<p>
  Nib has none of the detection, segmentation, sequence, or decoding stages. A model trained on
  single glyphs cannot read a line by itself, because it has no way to know where one character
  ends and the next begins.
</p>

<h3 id="single-glyph">Recipe: single-glyph recognition</h3>
<p>
  This is the only OCR-adjacent workflow that the current code supports. It works when the text
  is clean, fixed-font, and already split into one image per character.
</p>
<ol>
  <li>
    <strong>Crop each character externally.</strong> Use OpenCV, PIL, ImageMagick, or your
    capture pipeline. Produce one image per glyph, tightly framed, with a consistent size and
    background. Nib does not crop or segment.
  </li>
  <li>
    <strong>Define the class list.</strong> For example <code>0-9</code>, <code>A-Z</code>,
    <code>a-z</code> = 62 classes. Write <code>classes.txt</code> in exactly that order and use
    the same order for manifest labels.
  </li>
  <li>
    <strong>Collect hundreds of examples per class per font.</strong> Include the variations you
    expect at prediction time: weight, slight blur, ink color, background. More data matters more
    than a bigger network here.
  </li>
  <li>
    <strong>Train at 32×32 or 48×48, grayscale.</strong> The 16×16 default is usually too coarse
    for 62 classes.
    <pre><code>nib vision-train --manifest glyphs.txt --output glyphs.json \\
  --image-size 32 --channels 1 --classes 62 --epochs 40 --lr 0.01</code></pre>
  </li>
  <li>
    <strong>Classify each crop and join the results.</strong> Run <code>vision-predict</code> per
    crop, or call <code>VisionCNN::predict</code> in Rust. Preserve reading order in your own
    code. For spaces, use a blank-column or gap heuristic in the segmentation step.
  </li>
</ol>
<div class="callout warn">
  <strong>Where this breaks</strong>
  <p>
    Handwriting, multiple fonts, rotated or skewed text, touching characters, low contrast,
    variable spacing, and full pages. The classifier has no context: an isolated <code>l</code>,
    <code>1</code>, and <code>I</code> are three unrelated labels, and nothing in the model uses
    the surrounding characters to disambiguate them. Do not present this as general OCR.
  </p>
</div>
<p>
  For production OCR, use a dedicated engine: <strong>Tesseract</strong>,
  <strong>PaddleOCR</strong>, <strong>EasyOCR</strong>, <strong>docTR</strong>, or a cloud vision
  API. They include detection, segmentation, sequence models, and language correction that Nib
  does not have.
</p>
<p>
  Adding real OCR to Nib would mean a new model, not a new flag: a detection stage, a
  line-level sequence head (CTC or attention), wider inputs such as 32×128, more capacity, and a
  line-level dataset. That is a separate project from the current image classifier.
</p>

<h2 id="rust">Use it from Rust</h2>
<pre><code>use nib_rs::{load_image, load_manifest, VisionCNN, VisionTrainer};

fn main() -&gt; anyhow::Result&lt;()&gt; {
    let entries = load_manifest("manifest.txt")?;
    let images: Vec&lt;_&gt; = entries
        .iter()
        .map(|(path, _)| load_image(path, 32, 32, 1))
        .collect::&lt;Result&lt;_, _&gt;&gt;()?;
    let labels: Vec&lt;usize&gt; = entries.iter().map(|(_, label)| *label).collect();

    let mut trainer = VisionTrainer::new(VisionCNN::new(1, 32, 32, 10), 0.01);
    for epoch in 0..20 {
        println!("epoch {epoch}: loss {:.4}", trainer.train_step(&amp;images, &amp;labels));
    }
    trainer.save("vision.json")?;

    // Predict: predict() needs &amp;mut self because forward caches activations.
    let mut model = VisionCNN::load("vision.json")?;
    let image = load_image("glyph.png", model.input_h, model.input_w, model.input_channels)?;
    let probs = model.predict(&amp;image);
    let (index, confidence) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("class {index} at {confidence:.4}");
    Ok(())
}</code></pre>
<p>
  <code>VisionCNN</code> exposes <code>forward</code>, <code>backward</code>,
  <code>predict</code>, <code>save</code>, and <code>load</code>.
  <code>VisionTrainer::train_step</code> takes a slice of single-image tensors and labels and
  returns the batch loss.
</p>

<h2 id="limits">Limits</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Limitation</th><th>Consequence</th></tr></thead>
  <tbody>
    <tr><td>One label per image</td><td>Cannot read a line, a word, or a page. No sequence output.</td></tr>
    <tr><td>No detection or segmentation</td><td>You must crop the object or glyph yourself.</td></tr>
    <tr><td>No OCR</td><td>No text-in-the-wild support. Use Tesseract or PaddleOCR for that.</td></tr>
    <tr><td>No HTTP endpoint</td><td><code>nib serve</code> only serves text models. Vision is CLI or Rust.</td></tr>
    <tr><td>No GPU</td><td>CPU only, single thread.</td></tr>
    <tr><td>No augmentation</td><td>Invariance must come from your dataset.</td></tr>
    <tr><td>Small input</td><td>16×16 by default; fine detail is lost. Use 32 or 48 for text.</td></tr>
    <tr><td>No pre-trained weights</td><td>Every model starts random and needs labeled data.</td></tr>
    <tr><td>JSON weights</td><td>Easy to inspect, but larger than a binary format.</td></tr>
    <tr><td>Top-1 CLI output</td><td>Only the winning class is printed; use Rust for all probabilities.</td></tr>
  </tbody>
</table>
</div>

<h2 id="troubleshooting">Troubleshooting</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Symptom</th><th>Cause and fix</th></tr></thead>
  <tbody>
    <tr><td><code>manifest ... has no entries</code></td><td>The file is empty or every line is blank/comment. Add <code>path label</code> pairs.</td></tr>
    <tr><td><code>manifest line N is malformed</code></td><td>A line has no label. The label must be the final token.</td></tr>
    <tr><td><code>invalid label</code></td><td>Labels must be non-negative integers. Remove quotes and decimals.</td></tr>
    <tr><td>Loss stays at ~2.303 for 10 classes</td><td>That is <code>ln(10)</code>, i.e. random guessing. Check labels, raise the learning rate, or check that the images actually differ.</td></tr>
    <tr><td>Loss is suspiciously 0</td><td>A label is outside <code>0..classes-1</code> and is being skipped. Fix the manifest or raise <code>--classes</code>.</td></tr>
    <tr><td>Training accuracy is high, new images fail</td><td>Overfitting or a domain shift. Add more varied images, keep lighting/framing consistent, or hold out a real test set.</td></tr>
    <tr><td>All predictions are one class</td><td>Classes are imbalanced or the images are too small to distinguish. Balance the data and try 32×32 RGB.</td></tr>
    <tr><td>Wrong class names</td><td><code>classes.txt</code> must list names in label order, line 0 = class 0.</td></tr>
    <tr><td>Image size below 10</td><td>The convolution/pooling stack collapses to zero features. Use at least 10; 16 or larger is recommended.</td></tr>
    <tr><td><code>nib serve</code> rejects <code>vision.json</code></td><td>Expected. The HTTP server loads only <code>CTURBO01</code> and <code>CDENSE4</code> text models.</td></tr>
  </tbody>
</table>
</div>

<h2>Next steps</h2>
<div class="grid two">
  <a class="card" href="cli.html#vision-commands"><h3>CLI reference</h3><p>Every vision flag and default in one table.</p><div class="card-arrow">Open the reference →</div></a>
  <a class="card" href="use-cases.html#classify-images"><h3>Use cases</h3><p>Where image classification fits among the other Nib capabilities.</p><div class="card-arrow">See all use cases →</div></a>
  <a class="card" href="rust-api.html"><h3>Rust API</h3><p>Embed the CNN and trainer directly in your application.</p><div class="card-arrow">Read the API guide →</div></a>
  <a class="card" href="models.html"><h3>Models &amp; training</h3><p>How the text models compare, and how training works end to end.</p><div class="card-arrow">Read the deep dive →</div></a>
</div>
`,
  },
  {
    slug: "models",
    title: "Models & training",
    description:
      "How nib-rs models work: CharLM, low-rank TurboLM, dense 90M DenseLM, parameter and file-size math, 4-bit quantization, pruning, real Shakespeare data, training results, and benchmarks.",
    body: `
<section class="hero">
  <p class="eyebrow">Deep dive</p>
  <h1>Models &amp; training</h1>
  <p class="lead">
    Four model families, one set of hand-written layers. This page explains the architectures,
    the exact parameter and file-size math, how the shipped 90M model was trained and pruned,
    and what the numbers do and do not mean.
  </p>
</section>

<h2>Model family</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Model</th><th>Architecture</th><th>Parameters</th><th>Artifact</th></tr></thead>
  <tbody>
    <tr><td><strong>CharLM</strong></td><td>Embedding → flatten → Linear → ReLU → Linear</td><td>Configurable</td><td>JSON, f32</td></tr>
    <tr><td><strong>TurboLM</strong></td><td>Same MLP shape with low-rank <code>U·V</code> factors</td><td>100,009,068 logical / 994,572 stored</td><td>int8, 994,656 bytes</td></tr>
    <tr><td><strong>DenseLM</strong></td><td>Same MLP shape with full dense matrices</td><td>90,004,008</td><td>4-bit, 45,117,066 bytes</td></tr>
    <tr><td><strong>VisionCNN</strong></td><td>Conv → pool → conv → pool → Linear</td><td>Configurable</td><td>JSON, f32</td></tr>
  </tbody>
</table>
</div>

<h2>CharLM</h2>
<p>A character-level two-layer MLP. For a context of <code>C</code> characters, an embedding width <code>E</code>, hidden width <code>H</code>, and vocabulary <code>V</code>:</p>
<pre><code>input ids (C) → embedding (C × E) → flatten (C·E)
→ Linear(C·E → H) → ReLU
→ Linear(H → V) → softmax → next character</code></pre>
<p>Parameters:</p>
<pre><code>P = V·E + C·E·H + H + H·V + V</code></pre>
<p>
  The first term is the embedding table, the second is <code>fc1</code>, then its bias, then
  <code>fc2</code> and its bias. The trainer uses cross-entropy loss and plain SGD with
  gradients averaged over the batch.
</p>

<h2>TurboLM</h2>
<p>
  Turbo keeps the same 100M-entry shape but stores each large matrix as two thin factors:
</p>
<pre><code>fc1: (C·E = 8192) × (H = 12114)  →  U(8192 × 30) · V(30 × 12114)
fc2: (H = 12114) × (V = 62)      →  U(12114 × 30) · V(30 × 62)</code></pre>
<p>With <code>V=62, E=128, C=64, H=12114, rank=30</code>:</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Item</th><th class="num">Count</th></tr></thead>
  <tbody>
    <tr><td>Logical dense parameters</td><td class="num">100,009,068</td></tr>
    <tr><td>Stored factors, embedding, biases</td><td class="num">994,572</td></tr>
    <tr><td>File after int8 quantization</td><td class="num">994,656 bytes</td></tr>
    <tr><td>Compression vs f32 dense</td><td class="num">402×</td></tr>
  </tbody>
</table>
</div>
<p>
  The forward pass multiplies by the factors, so the <em>logical</em> shape is 100M while the
  number of independent stored weights is about 1M. This is why the file is under 1 MB. It is
  not a 100M-parameter dense model; it is a low-rank model of that shape.
</p>

<h2>DenseLM</h2>
<p>
  DenseLM materializes the factors into full matrices and keeps every weight. The shipped
  configuration is:
</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Field</th><th class="num">Value</th></tr></thead>
  <tbody>
    <tr><td>Vocabulary</td><td class="num">62</td></tr>
    <tr><td>Context</td><td class="num">64</td></tr>
    <tr><td>Embedding width</td><td class="num">128</td></tr>
    <tr><td>Hidden width</td><td class="num">10,902</td></tr>
    <tr><td>Embedding weights</td><td class="num">7,936</td></tr>
    <tr><td><code>fc1</code> weights</td><td class="num">89,309,184</td></tr>
    <tr><td><code>fc2</code> weights</td><td class="num">675,924</td></tr>
    <tr><td>Biases</td><td class="num">10,964</td></tr>
    <tr><td><strong>Total parameters</strong></td><td class="num"><strong>90,004,008</strong></td></tr>
    <tr><td><strong>4-bit file</strong></td><td class="num"><strong>45,117,066 bytes</strong></td></tr>
    <tr><td>f32 equivalent</td><td class="num">360.0 MB</td></tr>
  </tbody>
</table>
</div>
<pre><code>P = 62·128 + 64·128·10902 + 10902·62 + 10902 + 62
  = 7,936 + 89,309,184 + 675,924 + 10,964
  = 90,004,008</code></pre>

<h2 id="size-math">Size math</h2>
<p>Each dense matrix is stored as:</p>
<pre><code>8-byte header (rows, cols)
rows × 4-byte f32 row scale
rows × ceil(cols / 2) packed 4-bit bytes</code></pre>
<p>Biases are stored as a 4-byte length plus f32 values. The 32-byte file header records magic, version, vocabulary, embedding width, hidden width, context, and a reserved word.</p>
<pre><code>header                 32 bytes
embedding matrix        4,224 bytes
fc1 matrix         44,687,368 bytes
fc2 matrix            381,578 bytes
fc1 bias               43,612 bytes
fc2 bias                  252 bytes
----------------------------------
total              45,117,066 bytes</code></pre>
<p>
  At 4 bits per weight, the raw weights alone are about 45 MB; scales, headers, and f32 biases
  add roughly 80 KB. For comparison:
</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Precision</th><th class="num">90M weights</th><th class="num">100M weights</th><th class="num">100B weights</th></tr></thead>
  <tbody>
    <tr><td>f32</td><td class="num">360 MB</td><td class="num">400 MB</td><td class="num">400 GB</td></tr>
    <tr><td>int8</td><td class="num">90 MB</td><td class="num">100 MB</td><td class="num">100 GB</td></tr>
    <tr><td>4-bit</td><td class="num">45 MB</td><td class="num">50 MB</td><td class="num">50 GB</td></tr>
    <tr><td>2-bit</td><td class="num">22.5 MB</td><td class="num">25 MB</td><td class="num">25 GB</td></tr>
    <tr><td>1-bit</td><td class="num">11.25 MB</td><td class="num">12.5 MB</td><td class="num">12.5 GB</td></tr>
  </tbody>
</table>
</div>
<div class="callout danger">
  <strong>100B under 100 MB is impossible</strong>
  <p>
    Even at one bit per weight, 100B parameters need about 12.5 GB, before scales, embeddings,
    or runtime overhead. No quantization trick changes that information-theoretic floor. Nib
    targets 90M weights at 4 bits, which genuinely fits under 50 MB.
  </p>
</div>

<h2>4-bit quantization</h2>
<p>Dense weights are quantized row by row after training:</p>
<pre><code>scale = max_abs(row) / 7
q     = round(weight / scale) clamped to -7..7
store q as a signed 4-bit nibble, two values per byte</code></pre>
<p>
  One f32 scale per row preserves the dynamic range of each row without a per-weight scale.
  Biases remain f32. Loading dequantizes to f32, so inference runs on normal matrices; the
  4-bit format is a storage format, not a low-precision compute kernel.
</p>

<h2>Pruning: 100M → 90M</h2>
<p>
  A 100M dense model at 4 bits is 50,128,686 bytes, just over 50 MB. To land comfortably under
  50 MB, the shipped model drops 1,212 hidden units:
</p>
<ul>
  <li>Score every hidden unit by the L2 norm of its outgoing <code>fc2</code> row.</li>
  <li>Keep the 10,902 highest-scoring units, restoring their original order.</li>
  <li>Copy the corresponding <code>fc1</code> columns, <code>fc1</code> bias entries, and <code>fc2</code> rows.</li>
  <li>Fine-tune the resulting sub-network on the same corpus.</li>
</ul>
<p>
  The result is a direct sub-network with 90,004,008 independent weights. It is fully dense;
  pruning only chooses which hidden units survive.
</p>

<h2>Training data</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Item</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Corpus</td><td>Tiny Shakespeare (public domain)</td></tr>
    <tr><td>Source</td><td><a href="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" rel="noopener">Karpathy char-rnn corpus</a></td></tr>
    <tr><td>Bytes used in the shipped run</td><td>200,000 (first 200 KB)</td></tr>
    <tr><td>Lines</td><td>7,554</td></tr>
    <tr><td>Tokenizer</td><td>Character-level, 62 tokens</td></tr>
    <tr><td>Task</td><td>Given 64 characters, predict the next character</td></tr>
    <tr><td>Possible windows</td><td>199,936</td></tr>
    <tr><td>Validation split</td><td>None in the shipped run</td></tr>
  </tbody>
</table>
</div>
<p>
  The 62-token vocabulary contains newline, space, <code>!&amp;',-.:;?</code>, uppercase
  letters, and lowercase letters. Everything is real text; no synthetic samples, no
  pre-trained weights, and no external model artifacts are used.
</p>
<div class="callout warn">
  <strong>Training loss, not validation accuracy</strong>
  <p>
    The numbers below are mean cross-entropy on training windows. There is no held-out split,
    so they measure fit, not generalization. Treat generated samples as the practical quality
    check.
  </p>
</div>

<h2>Training pipeline</h2>
<div class="steps">
  <div class="step">
    <div>
      <h4>Stage 1 — train Turbo from random weights</h4>
      <p>5 epochs, batch 32, SGD, lr 0.05, all 199,936 windows. Loss 2.3209 → 1.7765. Result: 994,656-byte <code>turbo.bin</code>.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Stage 2 — expand and fine-tune all 100M weights</h4>
      <p>Materialize <code>W = U·V</code>, then 2 epochs × 20,000 windows, batch 128, SGD, lr 0.02. Loss 1.5755 → 1.5290. Result: 50,128,686-byte dense artifact.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Stage 3 — prune to 90M and fine-tune</h4>
      <p>Keep 10,902 of 12,114 hidden units, then 3 epochs × 20,000 windows, batch 128, SGD, lr 0.01. Loss <strong>1.5311 → 1.5154</strong> (best epoch 1.5079). Result: 45,117,066-byte <code>nib.bin</code>.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Stage 4 — quantize and verify</h4>
      <p>Save each row as 4-bit nibbles with an f32 scale, reload, and compare predictions. The file is hashed and served through the same API as Turbo.</p>
    </div>
  </div>
</div>

<h3>Reproduce the shipped model</h3>
<pre><code>./get-data.ps1
$bytes = [IO.File]::ReadAllBytes("shakespeare.txt")[0..199999]
[IO.File]::WriteAllBytes("shakespeare-200k.txt", $bytes)

nib turbo-train --input shakespeare-200k.txt --output turbo.bin \\
  --tokenizer tokenizer.json --target 100000000 --context 64 --embed 128 \\
  --rank 30 --epochs 5 --lr 0.05 --batch-size 32

nib dense-build --turbo turbo.bin --output dense-100m.bin --tokenizer tokenizer.json

nib dense-train --input shakespeare-200k.txt --output dense-100m.bin \\
  --tokenizer tokenizer.json --init-turbo turbo.bin \\
  --epochs 2 --max-samples 20000 --batch-size 128 --lr 0.02

nib dense-train --input shakespeare-200k.txt --output nib.bin \\
  --tokenizer tokenizer.json --init-dense dense-100m.bin --hidden 10902 \\
  --epochs 3 --max-samples 20000 --batch-size 128 --lr 0.01</code></pre>
<p>Results vary because the trainer does not fix a random seed.</p>

<h2>Benchmarks</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Workload</th><th>Result</th><th>Notes</th></tr></thead>
  <tbody>
    <tr><td>Release binary</td><td>3,041,792 bytes</td><td>GNU toolchain, LTO, stripped</td></tr>
    <tr><td>Turbo 200 characters</td><td>0.15 s (150 ms)</td><td>HTTP request, CPU, single thread</td></tr>
    <tr><td>Dense 20 characters</td><td>1.7 s</td><td>HTTP request, includes JSON and model lock</td></tr>
    <tr><td>Dense training</td><td>~0.035 s/sample</td><td>Batch 128; batch 32 is about 3× slower</td></tr>
    <tr><td>Dense RAM</td><td>~400 MB</td><td>f32 weights plus activations; allow 1 GB in containers</td></tr>
    <tr><td>Turbo RAM</td><td>&lt; 20 MB</td><td>Compact factors</td></tr>
  </tbody>
</table>
</div>
<p>
  Benchmarks are from one development laptop CPU. Dense generation cost scales with
  <code>length</code>; 120 characters took about 11 seconds in CLI sampling (~90 ms/char).
  Turbo is the interactive choice; dense is the artifact for inspecting a real 90M-weight
  model.
</p>

<h2>Prompt behavior</h2>
<p>
  Both generators use a fixed 64-character context. A short prompt is repeated to fill the
  context instead of being padded with spaces, which keeps the input closer to real text. A
  prompt longer than 64 characters is truncated from the left. Natural prompts of 64+
  characters give the most coherent output.
</p>

<h2>Limitations</h2>
<ul>
  <li><strong>Character-level, not subword.</strong> No BPE tokenizer, no instruction tuning, no chat template.</li>
  <li><strong>Small corpus.</strong> 200 KB of Shakespeare is enough for style and structure, not general knowledge.</li>
  <li><strong>No validation split.</strong> Reported loss is training loss; overfitting is possible.</li>
  <li><strong>Fixed 64-character context.</strong> Longer prompts are truncated, not attended to.</li>
  <li><strong>Quantization after training.</strong> 4-bit storage is not trained with quantization-aware gradients.</li>
  <li><strong>CPU only.</strong> No GPU kernels, no SIMD intrinsics beyond what the compiler emits, no batched inference server.</li>
  <li><strong>Not a foundation model.</strong> Do not compare it to pretrained LLMs; it is a from-scratch educational and embeddable model.</li>
</ul>

<h2>Frequently asked model questions</h2>
<details>
  <summary>Is the shipped model really trained from scratch?</summary>
  <p>Yes. Turbo starts from Xavier-random factors and learns from the Shakespeare file. Dense is expanded from that project's own Turbo factors, then fine-tuned with dense gradients. No external checkpoint is loaded.</p>
</details>
<details>
  <summary>Why is the dense file 45 MB rather than 90 MB?</summary>
  <p>Weights are stored at 4 bits each. 90M × 0.5 bytes is about 45 MB; row scales and biases add about 80 KB.</p>
</details>
<details>
  <summary>Why not keep 100M parameters?</summary>
  <p>A 100M dense model at 4 bits is 50,128,686 bytes, slightly over 50 MB. Pruning 1,212 hidden units lands at 45.12 MB while keeping the model fully dense. The 100M configuration is still available through <code>dense-build</code>.</p>
</details>
<details>
  <summary>Can it run on a GPU?</summary>
  <p>Not currently. The implementation uses <code>ndarray</code> on the CPU. GPU support would require a new backend and is outside the current scope.</p>
</details>
<details>
  <summary>Can I train on my own text?</summary>
  <p>Yes. Point <code>--input</code> at any UTF-8 text file and use a matching tokenizer path. A larger, cleaner corpus improves quality more than adding epochs.</p>
</details>
`,
  },
  {
    slug: "deployment",
    title: "Deployment",
    description:
      "Deploy nib-rs with Docker, a VPS and systemd, GitHub Pages, Netlify, Vercel, Cloudflare Pages, Render, Railway, Fly.io, or Google Cloud Run.",
    body: `
<section class="hero">
  <p class="eyebrow">Operations</p>
  <h1>Deployment</h1>
  <p class="lead">
    Nib ships as one static binary plus a model file and a tokenizer. The API needs a host that
    can execute binaries; the UI can live on any static host. This page covers the common paths.
  </p>
</section>

<h2>What runs where</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Piece</th><th>Needs</th><th>Good hosts</th></tr></thead>
  <tbody>
    <tr><td>Rust API + embedded UI</td><td>Linux/Windows/macOS process, 1 GB RAM for dense</td><td>VPS, Docker, Render, Railway, Fly.io, Cloud Run</td></tr>
    <tr><td>Static UI (<code>index.html</code>)</td><td>Static file hosting only</td><td>GitHub Pages, Netlify, Vercel, Cloudflare Pages</td></tr>
    <tr><td>Documentation site (<code>docs/</code>)</td><td>Static file hosting only</td><td>GitHub Pages, Netlify, Vercel, Cloudflare Pages</td></tr>
    <tr><td>Model files</td><td>45 MB dense or 1 MB Turbo</td><td>Baked into the image or downloaded at startup</td></tr>
  </tbody>
</table>
</div>
<div class="callout">
  <strong>GitHub Pages cannot run Rust</strong>
  <p>
    Pages serves HTML, CSS, and JavaScript only. Host <code>index.html</code> and this docs site
    on Pages, but run <code>nib serve</code> on a separate host. Point the UI at it with
    <code>?api=https://your-api-host</code>.
  </p>
</div>

<h2 id="public-url">Get a public API URL — six steps</h2>
<p>
  This is the short version. The <a href="http-api.html#public-endpoint">HTTP API guide</a> has
  the full endpoint walkthrough, tunnel instructions, UI connection, and troubleshooting.
</p>
<div class="steps">
  <div class="step">
    <div>
      <h4>1. Build</h4>
      <pre><code>cargo build --release</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>2. Copy the three files</h4>
      <p><code>nib</code>, <code>nib.bin</code> or <code>turbo.bin</code>, and <code>tokenizer.json</code>.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>3. Run on <code>0.0.0.0</code></h4>
      <pre><code>./nib serve --model nib.bin --tokenizer tokenizer.json --host 0.0.0.0 --port 8080</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>4. Terminate HTTPS</h4>
      <p>Use the platform's URL, or put Caddy/nginx in front of port 8080.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>5. Test from outside</h4>
      <pre><code>curl https://api.example.com/health
curl -X POST https://api.example.com/api/generate \\
  -H "content-type: application/json" \\
  -d '{"prompt":"ROMEO: ","length":40}'</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>6. Connect the UI</h4>
      <pre><code>https://vamshavardhan50.github.io/nib-ai/?api=https://api.example.com</code></pre>
    </div>
  </div>
</div>
<div class="callout">
  <strong>No public endpoint is included</strong>
  <p>
    The hosted UI and this documentation are static. They cannot execute Rust. Until you
    complete these steps, the only working endpoint is <code>http://127.0.0.1:8080</code>
    on your own machine.
  </p>
</div>

<h2>Build once</h2>
<pre><code>cargo build --release
# binary: target/release/nib
# model:  nib.bin (45.12 MB) or turbo.bin (994,656 bytes)
# vocab:  tokenizer.json</code></pre>
<p>Copy those three files to the host and run:</p>
<pre><code>./nib serve --model nib.bin --tokenizer tokenizer.json --host 0.0.0.0 --port 8080</code></pre>

<h2>Docker</h2>
<p>The included multi-stage Dockerfile builds the release binary and copies a model into a slim Debian runtime.</p>
<pre><code># Fast, compact demo (default)
docker build -t nib-turbo .
docker run --rm -p 8080:8080 nib-turbo

# Real 90M dense model
docker build --build-arg MODEL=nib.bin -t nib-dense .
docker run --rm -p 8080:8080 --memory=1g nib-dense</code></pre>
<p>The image exposes port 8080, runs as an unprivileged user, and reads <code>HOST</code> and <code>PORT</code> environment variables.</p>
<h3>Docker Compose</h3>
<pre><code>services:
  nib:
    build:
      context: .
      args:
        MODEL: nib.bin
    ports:
      - "8080:8080"
    environment:
      HOST: 0.0.0.0
      PORT: 8080
    mem_limit: 1g
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3</code></pre>
<p class="muted small">The runtime image does not include <code>curl</code> by default. Add it to the Dockerfile or use a TCP check if you rely on Docker healthchecks.</p>

<h2>GitHub Pages</h2>
<p>The repository is <a href="${repo}" rel="noopener">Vamshavardhan50/nib-ai</a>. The root <code>index.html</code> is the browser UI; <code>docs/index.html</code> is this documentation site.</p>
<div class="steps">
  <div class="step">
    <div>
      <h4>Push the repository</h4>
      <pre><code>git remote add origin ${repo}.git
git branch -M main
git push -u origin main</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Enable Pages</h4>
      <p>Repository → Settings → Pages → Build and deployment → Source: <strong>Deploy from a branch</strong> → Branch: <strong>main</strong> → Folder: <strong>/ (root)</strong> → Save.</p>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Open the UI and docs</h4>
      <pre><code>https://vamshavardhan50.github.io/nib-ai/
https://vamshavardhan50.github.io/nib-ai/docs/</code></pre>
    </div>
  </div>
  <div class="step">
    <div>
      <h4>Connect the API</h4>
      <pre><code>https://vamshavardhan50.github.io/nib-ai/?api=https://your-api-host</code></pre>
      <p>The UI saves the endpoint and includes it in the embed snippet.</p>
    </div>
  </div>
</div>
<p>
  If you prefer GitHub Actions, use the official Pages actions (<code>actions/configure-pages</code>,
  <code>actions/upload-pages-artifact</code>, <code>actions/deploy-pages</code>) and publish the
  repository root or a generated <code>_site/</code> directory. The docs site is plain HTML and
  needs no build step; <code>docs/build.mjs</code> only regenerates the pages from one layout.
</p>

<h2>Other static hosts</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Host</th><th>Setup</th></tr></thead>
  <tbody>
    <tr><td>Netlify</td><td>Drag the repository root or connect Git; publish directory <code>/</code>. No build command.</td></tr>
    <tr><td>Vercel</td><td>Import the repo as a static project; output directory <code>/</code>.</td></tr>
    <tr><td>Cloudflare Pages</td><td>Connect Git; framework preset “None”; build command empty; output directory <code>/</code>.</td></tr>
    <tr><td>S3 / CloudFront</td><td>Upload <code>index.html</code>, <code>docs/</code>, and assets; set <code>index.html</code> as the index document.</td></tr>
  </tbody>
</table>
</div>
<p>For all of them, set the API base with <code>?api=</code>; the UI is a single HTML file with inline CSS and JavaScript.</p>

<h2>VPS with systemd</h2>
<pre><code># /etc/systemd/system/nib.service
[Unit]
Description=nib-rs language model API
After=network.target

[Service]
Type=simple
User=nib
WorkingDirectory=/opt/nib
ExecStart=/opt/nib/nib serve --model /opt/nib/nib.bin --tokenizer /opt/nib/tokenizer.json --host 127.0.0.1 --port 8080
Restart=on-failure
RestartSec=3
MemoryMax=1G

[Install]
WantedBy=multi-user.target</code></pre>
<pre><code>sudo systemctl daemon-reload
sudo systemctl enable --now nib
curl http://127.0.0.1:8080/health</code></pre>

<h3>Caddy reverse proxy</h3>
<pre><code>api.example.com {
    reverse_proxy 127.0.0.1:8080 {
        transport http {
            read_timeout 300s
            write_timeout 300s
        }
    }
}</code></pre>
<h3>nginx reverse proxy</h3>
<pre><code>server {
    listen 443 ssl;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_set_header Host $host;
    }
}</code></pre>

<h2>Render, Railway, Fly.io, Cloud Run</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Platform</th><th>Key settings</th></tr></thead>
  <tbody>
    <tr><td>Render</td><td>Web Service → Docker → health check path <code>/health</code> → 1 GB RAM for dense → set <code>PORT</code> if required by the platform.</td></tr>
    <tr><td>Railway</td><td>Deploy from the repo; Railway injects <code>PORT</code>; the Dockerfile CMD expands <code>HOST</code> and <code>PORT</code>.</td></tr>
    <tr><td>Fly.io</td><td><code>fly launch</code>, internal port 8080, <code>fly scale memory 1024</code>, health check <code>/health</code>.</td></tr>
    <tr><td>Google Cloud Run</td><td>Container port 8080, memory 1 GiB, request timeout 300 s, concurrency 1 (generation is serialized).</td></tr>
  </tbody>
</table>
</div>
<p>
  Dense generation can exceed default platform timeouts. Prefer Turbo for request/response
  services, or raise the timeout and cap <code>length</code> in your client.
</p>

<h2>Security</h2>
<ul>
  <li><strong>No built-in auth, TLS, or rate limiting.</strong> Put Nib behind a reverse proxy or platform gateway.</li>
  <li><strong>CORS is <code>*</code>.</strong> Any website can call a public instance. Restrict at the proxy or fork the header logic if the model is private.</li>
  <li><strong>Limit request size and rate.</strong> The server caps bodies at 64 KB and prompts at 4,096 characters, but it does not rate-limit clients.</li>
  <li><strong>Treat model files as trusted input.</strong> The loaders validate magic and dimensions, but they are not a sandbox for hostile files.</li>
  <li><strong>Run as an unprivileged user</strong> (the Dockerfile does) and keep the runtime image minimal.</li>
</ul>

<h2>CI</h2>
<pre><code>name: ci
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --release
      - run: cargo build --release</code></pre>
<p>The test suite uses tiny in-memory models, so CI does not need <code>nib.bin</code> or a network download.</p>

<h2>Repository size</h2>
<ul>
  <li><code>nib.bin</code> is <strong>45,117,066 bytes</strong>, under GitHub's 100 MB per-file limit. Normal Git works.</li>
  <li>Each dense variant adds about 45 MB to history. Keep one shipped artifact; publish experiments as release assets.</li>
  <li><code>turbo.bin</code> is under 1 MB and is cheap to keep.</li>
  <li>For files over 100 MB, use Git LFS or GitHub Releases. Do not rewrite history just to remove one model unless the size is a real problem.</li>
</ul>

<h2>Deployment checklist</h2>
<ul class="checklist">
  <li>Release binary built and tested on the target platform.</li>
  <li><code>nib.bin</code> or <code>turbo.bin</code> plus matching <code>tokenizer.json</code> copied.</li>
  <li><code>/health</code> returns <code>{"status":"ok"}</code>.</li>
  <li><code>/api/info</code> reports the expected model name, parameters, and file bytes.</li>
  <li>Reverse proxy timeout at least 300 seconds for dense generation.</li>
  <li>Memory limit at least 1 GB for dense, 128 MB for Turbo.</li>
  <li>Static UI configured with <code>?api=https://your-api-host</code>.</li>
  <li>Authentication and rate limiting added at the proxy if the instance is public.</li>
</ul>
`,
  },
  {
    slug: "faq",
    title: "FAQ",
    description:
      "Frequently asked questions about nib-rs: naming, from-scratch training, model size, 100B impossibility, data, quality, performance, API, hosting, licensing, and limitations.",
    body: `
<section class="hero">
  <p class="eyebrow">Answers</p>
  <h1>Frequently asked questions</h1>
  <p class="lead">
    Direct answers about what Nib is, what it is not, how the sizes work, and why the model
    behaves the way it does.
  </p>
</section>

<h2>Project and naming</h2>

<h3>What is Nib?</h3>
<p>
  Nib (<code>nib-rs</code>) is a from-scratch Rust implementation of character-level language
  models and a small vision CNN. It includes training, quantization, a CLI, a Rust library, a
  zero-dependency HTTP API, and a browser UI. The shipped dense artifact has 90,004,008
  independent weights in 45,117,066 bytes.
</p>

<h3>Why the name “Nib”?</h3>
<p>
  A nib is the writing tip of a pen: small, precise, and the part that actually touches the
  page. It fits a project about compact models and generated text. The package is
  <code>nib-rs</code>, the binary is <code>nib</code>, and the library is <code>nib_rs</code>.
</p>

<h3>Is this the official Cactus project?</h3>
<p>
  No. Nib is an independent implementation inspired by the goals of small, embeddable language
  models. References to Cactus or its Needle model in the README are context, not affiliation.
  All code and weights here are produced by this repository.
</p>

<h3>Did the project get renamed?</h3>
<p>
  Yes. Earlier development used the name <code>cactus-rs</code>. The current name is Nib, and
  all package names, commands, UI titles, docs, and artifacts use it. The new repository is
  <a href="${repo}" rel="noopener">Vamshavardhan50/nib-ai</a>.
</p>

<h2>From scratch and data</h2>

<h3>Is it really trained from scratch?</h3>
<p>
  Yes. TurboLM starts from Xavier-random low-rank factors and trains on real text with
  hand-written forward and backward passes. DenseLM is expanded from this project's own Turbo
  factors, then fine-tuned with dense gradients. No external checkpoint, GGUF, or pretrained
  weight file is loaded.
</p>

<h3>What data was used?</h3>
<p>
  The first 200,000 bytes of <a href="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" rel="noopener">Tiny Shakespeare</a>,
  a public-domain corpus of Shakespeare plays and poems. That is 7,554 lines and a 62-character
  vocabulary. The full corpus is about 1.1 MB; the shipped run used a 200 KB prefix so CPU
  training stayed practical.
</p>

<h3>Is the data mock or synthetic?</h3>
<p>
  No. It is real public-domain text downloaded over HTTPS by <code>get-data.ps1</code>. The
  training windows are slices of that text; no labels or samples are fabricated.
</p>

<h3>Is there a validation split?</h3>
<p>
  Not in the shipped run. Reported losses are training cross-entropy. Adding a held-out split
  is a good next step; the trainer already accepts arbitrary windows, so it is a small change
  to evaluate on a disjoint slice.
</p>

<h3>How long does training take?</h3>
<p>
  On the development laptop CPU, dense fine-tuning runs at roughly 0.035 seconds per sample at
  batch 128, about 12 minutes per 20,000-window epoch. Three epochs is roughly 35–40 minutes.
  Turbo training is much faster; CharLM is seconds to minutes depending on size.
</p>

<h2>Model size and the 100B question</h2>

<h3>Why does the dense model have 90M parameters instead of 100M?</h3>
<p>
  A 100M dense model at 4 bits is 50,128,686 bytes, slightly over 50 MB. Pruning 1,212 hidden
  units produces 90,004,008 parameters and a 45,117,066-byte file, comfortably under 50 MB. The
  100M configuration is still available with <code>dense-build</code> or <code>dense-train</code>.
</p>

<h3>Can 100B parameters fit under 100 MB?</h3>
<p>
  No. 100B weights at 1 bit per weight need about 12.5 GB. At 4 bits they need about 50 GB; at
  int8 about 100 GB; at f32 about 400 GB. This is an information-theoretic floor, not an
  implementation limitation. No compression scheme can put 100B independent weights in 100 MB.
</p>

<h3>How does Turbo fit a 100M shape in under 1 MB?</h3>
<p>
  Turbo stores each large matrix as two thin factors, <code>U·V</code>, with rank 30. The
  logical dense shape is 100,009,068 parameters, but only 994,572 independent values are
  stored. It is a low-rank model of that shape, not a dense 100M-weight model. DenseLM is the
  artifact with 90M independent weights.
</p>

<h3>Is the 4-bit file still a “real” 90M model?</h3>
<p>
  Yes. Every one of the 90,004,008 weights exists as a signed 4-bit value plus one f32 scale per
  row. Loading dequantizes them into f32 matrices. The storage is compressed; the network is
  dense.
</p>

<h3>What does 4-bit quantization cost in quality?</h3>
<p>
  There is some loss because each row shares one scale and weights are rounded to 15 levels.
  The project quantizes after training, then reloads the artifact and checks predictions in
  tests. Fine-tuning before quantization keeps the loss small, but this is not
  quantization-aware training.
</p>

<h2>Quality and behavior</h2>

<h3>Why is the generated text sometimes garbled?</h3>
<p>
  The model is a 90M-parameter character-level MLP trained on 200 KB of Shakespeare. It learns
  spelling, speaker labels, line breaks, and local style, not general language understanding.
  Short prompts are repeated to fill the 64-character context; longer natural prompts produce
  better output. Temperature also matters: very high values increase noise.
</p>

<h3>Can it answer questions or follow instructions?</h3>
<p>
  No. It is not instruction-tuned and has no chat template. It continues text in the style of
  its corpus. Asking “What is the capital of France?” is outside its training objective.
</p>

<h3>Why character-level instead of BPE?</h3>
<p>
  Character-level tokenization is simple, deterministic, and has a tiny vocabulary (62 tokens
  here). It makes the whole pipeline readable and keeps the embedding table small. A BPE
  tokenizer would improve long-range efficiency but add substantial complexity.
</p>

<h3>Why an MLP instead of a transformer?</h3>
<p>
  The goal is a small, fast, from-scratch CPU model with a fully understandable training loop.
  A fixed-context MLP is much simpler than attention and still learns local character patterns.
  Transformers are a possible future direction, not a current feature.
</p>

<h3>Can I train on my own text?</h3>
<p>
  Yes. Point <code>--input</code> at any UTF-8 text file. Use a matching tokenizer path; the
  tokenizer is built from the corpus for CharLM and Turbo. More clean data usually helps more
  than more epochs.
</p>

<h3>Can I use the full 1.1 MB Shakespeare corpus?</h3>
<p>
  Yes. Run <code>get-data.ps1</code> and train on the full file. Expect longer training and
  somewhat better coverage. The shipped 200 KB prefix was a deliberate speed/quality tradeoff.
</p>

<h2>Performance</h2>

<h3>Why is dense generation slow?</h3>
<p>
  Every generated character runs a forward pass through 90M weights on the CPU: about 89M
  multiply-adds in <code>fc1</code> alone. A 20-character HTTP request takes about 1.7 seconds;
  120 characters take about 11 seconds. Turbo is roughly two orders of magnitude cheaper per
  character.
</p>

<h3>How fast is Turbo?</h3>
<p>
  Turbo 200-character generation takes <strong>0.15 s (150 ms)</strong> on the development CPU.
  It is the right choice for an interactive demo.
</p>

<h3>How much RAM does it need?</h3>
<p>
  Dense loads 90M f32 weights, about 360 MB, plus activations and runtime overhead; allow about
  400 MB and use a 1 GB container limit. Turbo stays under about 20 MB. Training needs more
  than inference because it keeps gradients and intermediate matrices.
</p>

<h3>Can it use a GPU?</h3>
<p>
  Not currently. The implementation uses <code>ndarray</code> on the CPU. GPU support would
  require a new backend, kernels, and memory management; it is outside the current scope.
</p>

<h2>Vision and OCR</h2>

<h3>Can Nib read text from an image?</h3>
<p>
  No. Nib has no OCR engine. The vision model is a single-label image classifier: one image in,
  one class and one confidence out. It cannot detect text, segment lines, or return a string of
  characters. It can classify a <em>pre-cropped</em> character if you train it on glyphs and do
  the cropping yourself. For real OCR use Tesseract, PaddleOCR, EasyOCR, docTR, or a cloud vision
  API. Full explanation and the single-glyph recipe:
  <a href="vision.html#ocr">Vision &amp; OCR</a>.
</p>

<h3>How accurate is the vision model?</h3>
<p>
  On a real 10-class handwritten-digit test — 1,497 training images and 300 held-out images at
  16×16 grayscale — it reached <strong>95.33%</strong> (286/300) after 30 epochs in 13.5 seconds.
  That number is specific to that dataset and input size. Accuracy on your images depends on how
  many labeled examples you provide, class balance, and how closely prediction images match the
  training framing.
</p>

<h3>Does <code>nib serve</code> expose the vision model?</h3>
<p>
  No. The HTTP server loads only <code>CTURBO01</code> and <code>CDENSE4</code> text models.
  Vision is available through <code>nib vision-train</code>, <code>nib vision-predict</code>, and
  the Rust API. See <a href="vision.html#limits">Vision limits</a>.
</p>

<h2>API and deployment</h2>

<h3>What is the API endpoint?</h3>
<p>
  There is no public endpoint hosted for you. The Pages UI and docs are static. Start the
  server locally with <code>nib serve --model nib.bin --tokenizer tokenizer.json --host
  127.0.0.1 --port 8080</code> and use <code>http://127.0.0.1:8080</code>. After deploying,
  the base URL is your HTTPS host, for example <code>https://api.example.com</code>, and
  generation is always <code>POST {base}/api/generate</code>. The full walkthrough — local,
  tunnel, public host, UI connection, and troubleshooting — is in
  <a href="http-api.html#endpoint">HTTP API → What is the endpoint?</a>.
</p>

<h3>Does <code>serve</code> work with both models?</h3>
<p>
  Yes. It reads the first eight bytes and dispatches to the Turbo int8 or dense int4 loader.
  The HTTP endpoints, UI, CORS behavior, and response schema are identical.
</p>

<h3>Can I host everything on GitHub Pages?</h3>
<p>
  You can host the UI and this documentation site on Pages. Pages cannot execute Rust, so the
  API must run on a VPS, Docker host, Render, Railway, Fly.io, Cloud Run, or similar. Point the
  static UI at the API with <code>?api=https://your-api-host</code>.
</p>

<h3>Is there authentication or TLS?</h3>
<p>
  Not in the built-in server. Put it behind Caddy, nginx, a cloud load balancer, or a platform
  gateway for TLS, auth, rate limiting, and logging. CORS is intentionally open (<code>*</code>)
  so static demos can call the API.
</p>

<h3>Can I stream tokens?</h3>
<p>
  No. <code>/api/generate</code> returns the full text in one JSON response after generation
  completes. Streaming would require a different response format and incremental decoding.
</p>

<h3>Can I embed the UI on my website?</h3>
<p>
  Yes. Run <code>nib web-export</code>, host the HTML anywhere, and embed it with an iframe
  whose <code>src</code> includes <code>?api=https://your-api-host</code>. The UI also shows a
  ready-made snippet.
</p>

<h3>Does it work offline?</h3>
<p>
  Yes. The binary, model, tokenizer, and UI have no runtime network dependencies. Only the
  optional data downloader needs the internet.
</p>

<h2>Licensing and contributing</h2>

<h3>What license is the code under?</h3>
<p>
  Apache-2.0. You can use, modify, and distribute it, including commercially, provided you
  keep the license and notices. The Shakespeare corpus is public domain.
</p>

<h3>Can I use the trained weights commercially?</h3>
<p>
  The weights are generated by this project from public-domain text. Apache-2.0 covers the
  code; treat the weights as your own artifacts. As always, verify that your own training data
  and use case comply with applicable law and licenses.
</p>

<h3>How can I contribute?</h3>
<p>
  Open an issue or pull request on <a href="${repo}" rel="noopener">GitHub</a>. Run
  <code>cargo test</code> before sending changes. Useful directions include a validation split,
  larger corpora, a BPE tokenizer, SIMD kernels, streaming responses, and deployment examples.
</p>

<h3>Why is the binary about 3 MB?</h3>
<p>
  It includes Rust's standard library, <code>ndarray</code>, serde, clap, and the image decoder,
  with LTO, one codegen unit, abort-on-panic, and symbol stripping. That is small for a complete
  trainer, server, and image pipeline, but not a microcontroller binary.
</p>

<h2>Known limitations</h2>
<ul>
  <li>No validation split or held-out benchmark in the shipped run.</li>
  <li>Character-level only; no BPE or subword tokenizer.</li>
  <li>Fixed 64-character context; no attention or long-range memory.</li>
  <li>CPU-only, single-threaded generation, no batching.</li>
  <li>No streaming, authentication, TLS, or rate limiting in the server.</li>
  <li>4-bit quantization happens after training, not during it.</li>
  <li>The model is a small Shakespeare-style character model, not a general assistant.</li>
  <li>Training does not fix a random seed, so reproduced runs differ slightly.</li>
</ul>
`,
  },
];

const esc = (s) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const nav = pages
  .map((page) => `<a href="${page.slug}.html">${esc(page.title)}</a>`)
  .join("\n        ");

function layout(page) {
  const title =
    page.slug === "index"
      ? "nib-rs documentation"
      : `${page.title} · nib-rs docs`;
  const metaTitle = esc(title);
  const metaDescription = esc(page.description);
  const pageUrl = `${site}/${page.slug === "index" ? "" : page.slug + ".html"}`;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${metaTitle}</title>
<meta name="description" content="${metaDescription}">
<meta name="theme-color" content="#0f766e">
<link rel="canonical" href="${pageUrl}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="nib-rs docs">
<meta property="og:title" content="${metaTitle}">
<meta property="og:description" content="${metaDescription}">
<meta property="og:url" content="${pageUrl}">
<meta name="twitter:card" content="summary">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%230f766e'/%3E%3Ctext x='16' y='23' font-family='monospace' font-size='20' font-weight='bold' fill='%23e8fdf5' text-anchor='middle'%3En%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="style.css">
<script>
  try {
    var saved = localStorage.getItem("nib-docs-theme");
    if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  } catch (e) {}
</script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>

<header class="topbar">
  <a class="brand" href="index.html" aria-label="nib-rs documentation home">
    <svg class="brand-mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M12 19l7-7 3 3-7 7-3-3z"></path>
      <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path>
      <path d="M2 2l7.586 7.586"></path>
      <circle cx="11" cy="11" r="2"></circle>
    </svg>
    <span class="brand-name">nib<span class="brand-suffix">-rs</span></span>
    <span class="brand-tag">docs</span>
  </a>
  <div class="topbar-actions">
    <a class="pill" href="../index.html" title="Open the browser UI">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="4" width="18" height="14" rx="2"></rect><path d="M8 21h8M12 18v3"></path></svg>
      <span class="label">Live UI</span>
    </a>
    <a class="pill" href="${repo}" rel="noopener" title="GitHub repository">
      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.42c.58.1.79-.25.79-.56v-2.17c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.2 1.77 1.2 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.28-5.27-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.8 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.12 3.05.74.81 1.18 1.84 1.18 3.1 0 4.43-2.7 5.4-5.28 5.69.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z"></path></svg>
      <span class="label">GitHub</span>
    </a>
    <button class="icon-btn" id="themeToggle" type="button" aria-label="Toggle theme">☾</button>
    <button class="icon-btn" id="menuToggle" type="button" aria-label="Open navigation" aria-expanded="false" aria-controls="sidebar">☰</button>
  </div>
</header>

<div class="shell">
  <aside class="sidebar" id="sidebar">
    <nav aria-label="Documentation">
      <p class="nav-label">Documentation</p>
      ${nav}
    </nav>
    <div class="toc" id="toc" aria-label="On this page"></div>
    <div class="sidebar-card">
      <span class="k">Shipped dense model</span>
      <span class="v">90,004,008 weights</span>
      <small>45,117,066 bytes · 4-bit · v${version}</small>
    </div>
  </aside>

  <main id="main">
    <article>
      <div class="crumbs"><a href="index.html">Docs</a> <span>/</span> <span>${esc(page.title)}</span></div>
      ${page.body}
      <footer class="footer">
        <div class="footer-row">
          <span><strong>nib-rs</strong> ${version}</span>
          <span class="dotsep">·</span>
          <span>Apache-2.0</span>
          <span class="dotsep">·</span>
          <a href="${repo}" rel="noopener">GitHub</a>
          <span class="dotsep">·</span>
          <a href="${repo}/issues" rel="noopener">Issues</a>
          <span class="dotsep">·</span>
          <a href="index.html">Docs home</a>
        </div>
        <p class="small muted">Built from scratch in Rust. No pre-trained weights, no external model files.</p>
      </footer>
    </article>
  </main>
</div>

<script src="docs.js" defer></script>
</body>
</html>
`;
}

for (const page of pages) {
  writeFileSync(join(docsDir, `${page.slug}.html`), layout(page), "utf8");
  console.log(`wrote docs/${page.slug}.html`);
}
