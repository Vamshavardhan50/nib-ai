//! Zero-dependency HTTP server + browser UI for a trained Turbo model.
//!
//! One command turns the model into a JSON API and a demo website:
//!
//! ```text
//! cactus serve --model turbo.bin --tokenizer tokenizer.json --port 8080
//! ```
//!
//! The API has CORS enabled, so the same page (or your own site) can call it
//! from any origin. The UI is a single embedded HTML file, which means it can
//! also be copied to GitHub Pages, Netlify, Vercel, or Cloudflare Pages and
//! pointed at a hosted API with `?api=https://your-host`.

use crate::model::Tokenizer;
use crate::turbo::{self, TurboLM};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_BODY_BYTES: usize = 64 * 1024;
const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_PROMPT_CHARS: usize = 4096;

struct App {
    model: Mutex<TurboLM>,
    tokenizer: Tokenizer,
    model_bytes: u64,
}

#[derive(Debug, Default, Deserialize)]
struct GenerateRequest {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    length: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
}

/// The embedded browser UI (also used by `cactus web-export`).
pub fn index_html() -> &'static str {
    INDEX_HTML
}

/// Start the HTTP server. Blocks forever (Ctrl+C to stop).
pub fn serve(model_path: &str, tokenizer_path: &str, host: &str, port: u16) -> Result<()> {
    let model = TurboLM::load(model_path)
        .with_context(|| format!("load turbo model {model_path}"))?;
    let tokenizer = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("load tokenizer {tokenizer_path}"))?;
    if tokenizer.vocab_size() != model.config.vocab_size {
        anyhow::bail!(
            "tokenizer vocab ({}) does not match model vocab ({})",
            tokenizer.vocab_size(),
            model.config.vocab_size
        );
    }
    let model_bytes = std::fs::metadata(model_path)
        .with_context(|| format!("read metadata for {model_path}"))?
        .len();

    let app = Arc::new(App {
        model: Mutex::new(model),
        tokenizer,
        model_bytes,
    });

    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)
        .with_context(|| format!("bind {addr} (is another process using this port?)"))?;

    let ui_host = if host == "0.0.0.0" || host == "::" { "localhost" } else { host };
    println!("cactus server listening on http://{ui_host}:{port}/");
    println!("API: POST http://{ui_host}:{port}/api/generate");
    println!(
        "model: {model_path} ({} bytes, {:.3} MB), CORS enabled",
        model_bytes,
        model_bytes as f64 / 1_000_000.0
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let app = Arc::clone(&app);
                std::thread::spawn(move || {
                    if let Err(err) = handle(stream, &app) {
                        eprintln!("request error: {err:#}");
                    }
                });
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream, app: &App) -> Result<()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let mut reader = BufReader::new(stream.try_clone()?);

    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let (method, path) = match parse_request_line(&request_line) {
        Some(parsed) => parsed,
        None => {
            respond_json(
                &mut stream,
                "400 Bad Request",
                &json!({ "error": "malformed request line" }),
            )?;
            return Ok(());
        }
    };
    let method = method.to_ascii_uppercase();

    // Headers: only Content-Length matters for this small API.
    let mut content_length = 0usize;
    let mut header_bytes = 0usize;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            break;
        }
        header_bytes += read;
        if header_bytes > MAX_HEADER_BYTES {
            respond_json(
                &mut stream,
                "431 Request Header Fields Too Large",
                &json!({ "error": "headers too large" }),
            )?;
            return Ok(());
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = match value.trim().parse() {
                    Ok(n) => n,
                    Err(_) => {
                        respond_json(
                            &mut stream,
                            "400 Bad Request",
                            &json!({ "error": "invalid content-length" }),
                        )?;
                        return Ok(());
                    }
                };
            }
        }
    }

    if content_length > MAX_BODY_BYTES {
        respond_json(
            &mut stream,
            "413 Payload Too Large",
            &json!({ "error": "request body is too large" }),
        )?;
        return Ok(());
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }

    if method == "OPTIONS" {
        return respond_empty(&mut stream, "204 No Content");
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes(),
        ),
        ("GET", "/health") => {
            respond_json(&mut stream, "200 OK", &json!({ "status": "ok" }))
        }
        ("GET", "/api/info") => respond_json(&mut stream, "200 OK", &model_info(app)),
        ("GET", "/openapi.json") => respond_json(&mut stream, "200 OK", &openapi()),
        ("POST", "/api/generate") => generate(&mut stream, app, &body),
        ("GET", "/api/generate") => respond_json(
            &mut stream,
            "405 Method Not Allowed",
            &json!({ "error": "use POST /api/generate with a JSON body" }),
        ),
        ("GET", "/favicon.ico") => respond_empty(&mut stream, "204 No Content"),
        _ => respond_json(
            &mut stream,
            "404 Not Found",
            &json!({ "error": "not found", "paths": ["/", "/health", "/api/info", "/api/generate", "/openapi.json"] }),
        ),
    }
}

fn generate(stream: &mut TcpStream, app: &App, body: &[u8]) -> Result<()> {
    let request: GenerateRequest = if body.is_empty() {
        GenerateRequest::default()
    } else {
        match serde_json::from_slice(body) {
            Ok(value) => value,
            Err(err) => {
                respond_json(
                    stream,
                    "400 Bad Request",
                    &json!({ "error": format!("invalid JSON: {err}") }),
                )?;
                return Ok(());
            }
        }
    };

    let prompt = request
        .prompt
        .unwrap_or_else(|| String::from("ROMEO: "));
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        respond_json(
            stream,
            "400 Bad Request",
            &json!({ "error": format!("prompt is limited to {MAX_PROMPT_CHARS} characters") }),
        )?;
        return Ok(());
    }

    let length = clamp_length(request.length);
    let temperature = clamp_temperature(request.temperature);
    let started = Instant::now();
    let text = {
        let mut model = app.model.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        turbo::generate_text(&mut model, &app.tokenizer, &prompt, length, temperature)
    };
    let elapsed_ms = started.elapsed().as_millis() as u64;

    respond_json(
        stream,
        "200 OK",
        &json!({
            "text": text,
            "prompt": prompt,
            "length": length,
            "temperature": temperature,
            "ms": elapsed_ms,
        }),
    )
}

fn model_info(app: &App) -> Value {
    let model = app.model.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let config = &model.config;
    json!({
        "name": "TurboLM",
        "architecture": "low-rank MLP, int8 weights on disk",
        "logical_parameters": model.logical_params(),
        "stored_parameters": model.stored_params(),
        "file_bytes": app.model_bytes,
        "config": {
            "vocab_size": config.vocab_size,
            "embed_dim": config.embed_dim,
            "hidden_dim": config.hidden_dim,
            "context": config.context,
            "rank": config.rank,
        },
        "endpoints": {
            "health": "/health",
            "info": "/api/info",
            "generate": "/api/generate",
            "openapi": "/openapi.json",
        }
    })
}

fn openapi() -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "cactus-rs TurboLM API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Generate text with a from-scratch compact language model."
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": { "200": { "description": "Server is ready" } }
                }
            },
            "/api/info": {
                "get": {
                    "summary": "Model metadata",
                    "responses": { "200": { "description": "Parameter counts, config, file size" } }
                }
            },
            "/api/generate": {
                "post": {
                    "summary": "Generate text",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "prompt": { "type": "string", "maxLength": MAX_PROMPT_CHARS },
                                        "length": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 300 },
                                        "temperature": { "type": "number", "minimum": 0.01, "maximum": 2.0, "default": 0.8 }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Generated text and timing" },
                        "400": { "description": "Invalid request" }
                    }
                }
            }
        }
    })
}

fn parse_request_line(line: &str) -> Option<(String, String)> {
    let mut parts = line.trim_end_matches(['\r', '\n']).split_whitespace();
    let method = parts.next()?;
    let target = parts.next()?;
    let version = parts.next()?;
    if !version.starts_with("HTTP/") || method.is_empty() {
        return None;
    }
    let path = target.split('?').next().unwrap_or("/");
    Some((method.to_string(), path.to_string()))
}

fn clamp_length(value: Option<usize>) -> usize {
    value.unwrap_or(300).clamp(1, 2000)
}

fn clamp_temperature(value: Option<f32>) -> f32 {
    let value = value.unwrap_or(0.8);
    if value.is_finite() {
        value.clamp(0.01, 2.0)
    } else {
        0.8
    }
}

fn respond_json(stream: &mut TcpStream, status: &str, value: &Value) -> Result<()> {
    let body = serde_json::to_vec(value)
        .unwrap_or_else(|_| b"{\"error\":\"serialization failed\"}".to_vec());
    respond(stream, status, "application/json; charset=utf-8", &body)
}

fn respond_empty(stream: &mut TcpStream, status: &str) -> Result<()> {
    respond(stream, status, "text/plain; charset=utf-8", b"")
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Access-Control-Max-Age: 86400\r\n\
         Content-Security-Policy: frame-ancestors *\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Cache-Control: no-store\r\n\
         \r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

const INDEX_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cactus-rs · TurboLM</title>
<style>
:root{--bg:#0b0d12;--panel:#12151d;--line:#252a36;--text:#eef1f7;--muted:#929caf;--accent:#7ce38b;--accent2:#6ea8fe;--danger:#ff7b72}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:radial-gradient(1200px 700px at 20% -10%,#173225 0%,transparent 55%),radial-gradient(900px 600px at 100% 0%,#1b2438 0%,transparent 50%),var(--bg);color:var(--text);font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:900px;margin:0 auto;padding:32px 20px 56px}
header{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:22px}
.brand{display:flex;gap:12px;align-items:center}
.logo{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,var(--accent),#2fbf71);display:grid;place-items:center;color:#07130c;font-weight:800;font-size:18px}
h1{font-size:22px;margin:0;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:13px;margin:0}
.badge{border:1px solid var(--line);background:#141923;color:var(--muted);border-radius:999px;padding:6px 11px;font-size:12px;white-space:nowrap}
.badge b{color:var(--accent)}
.card{background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 18px 50px #0006;backdrop-filter:blur(8px)}
label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:7px}
input,textarea,button{font:inherit}
input[type=url],textarea{width:100%;border:1px solid var(--line);background:#0d1017;color:var(--text);border-radius:11px;padding:11px 13px;outline:none}
input[type=url]:focus,textarea:focus{border-color:#3f6f52;box-shadow:0 0 0 3px #7ce38b1a}
textarea{min-height:96px;resize:vertical}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:640px){.grid{grid-template-columns:1fr}}
input[type=range]{width:100%;accent-color:var(--accent)}
.val{color:var(--accent);font-variant-numeric:tabular-nums}
button{border:0;border-radius:11px;padding:11px 18px;font-weight:650;cursor:pointer;transition:.15s}
button:disabled{opacity:.55;cursor:wait}
.primary{background:linear-gradient(135deg,var(--accent),#42c977);color:#06110a}
.ghost{background:#181d27;color:var(--text);border:1px solid var(--line)}
button:hover:not(:disabled){transform:translateY(-1px)}
.bar{display:flex;gap:10px;align-items:center;margin-top:16px;flex-wrap:wrap}
.status{margin-left:auto;font-size:12px;color:var(--muted);display:flex;gap:7px;align-items:center}
.dot{width:8px;height:8px;border-radius:50%;background:#f0b429}
.dot.ok{background:var(--accent);box-shadow:0 0 10px #7ce38b88}
.dot.err{background:var(--danger)}
.out{margin-top:18px;border:1px solid var(--line);border-radius:14px;background:#0a0c11;min-height:180px;padding:16px;white-space:pre-wrap;word-break:break-word;font:14px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;color:#d7e2d9;max-height:420px;overflow:auto}
.meta{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px;color:var(--muted);font-size:12px}
.meta b{color:var(--text);font-weight:600}
details{margin-top:18px;border-top:1px solid var(--line);padding-top:14px;color:var(--muted);font-size:13px}
summary{cursor:pointer;color:var(--text)}
code{background:#0a0c11;border:1px solid var(--line);border-radius:6px;padding:2px 6px;font-size:12px}
pre code{display:block;padding:12px;overflow:auto;white-space:pre-wrap;word-break:break-all}
a{color:var(--accent2)}
footer{margin-top:26px;color:var(--muted);font-size:12px;text-align:center}
</style>
</head>
<body>
<main>
<header>
  <div class="brand"><div class="logo">c</div><div><h1>cactus-rs · TurboLM</h1><p class="sub">100M-shape language model in under 1 MB</p></div></div>
  <div class="badge">model <b id="sizeBadge">0.995 MB</b></div>
</header>

<section class="card">
  <label for="endpoint">API endpoint</label>
  <input id="endpoint" type="url" placeholder="https://your-cactus-host" spellcheck="false">
  <div class="grid">
    <div><label for="prompt">Prompt</label><textarea id="prompt" spellcheck="false">ROMEO: </textarea></div>
    <div>
      <label for="length">Characters <span class="val" id="lengthVal">300</span></label>
      <input id="length" type="range" min="50" max="1000" step="50" value="300">
      <label for="temp" style="margin-top:18px">Temperature <span class="val" id="tempVal">0.70</span></label>
      <input id="temp" type="range" min="0.1" max="1.5" step="0.05" value="0.7">
    </div>
  </div>
  <div class="bar">
    <button class="primary" id="go">Generate</button>
    <button class="ghost" id="stop" disabled>Stop</button>
    <button class="ghost" id="copy">Copy</button>
    <span class="status"><span class="dot" id="dot"></span><span id="status">checking API…</span></span>
  </div>
  <div class="out" id="out">Train a model, start the server, and press Generate.</div>
  <div class="meta" id="meta">…</div>
  <details>
    <summary>Embed this on any website</summary>
    <pre><code id="embed">&lt;iframe src="https://YOUR-HOST/" width="100%" height="760" style="border:0;border-radius:16px" title="cactus-rs TurboLM"&gt;&lt;/iframe&gt;</code></pre>
    <p>The page works from GitHub Pages, Netlify, Vercel, Cloudflare Pages, or any static host. Set <code>?api=https://your-host</code> or enter the API endpoint above; CORS is enabled on the server.</p>
  </details>
</section>
<footer>From-scratch Rust · no pre-trained weights · <a href="/api/info">model info</a></footer>
</main>
<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const endpoint = $("endpoint"), prompt = $("prompt"), out = $("out"), dot = $("dot"), status = $("status");
  const go = $("go"), stop = $("stop"), copy = $("copy"), meta = $("meta"), sizeBadge = $("sizeBadge"), embed = $("embed");
  const params = new URLSearchParams(location.search);
  const staticHost = /(^|\.)(github\.io|netlify\.app|vercel\.app|pages\.dev|web\.app|surge\.sh)$/.test(location.hostname);
  let saved = "";
  try { saved = localStorage.getItem("cactus-api") || ""; } catch (e) {}
  let api = params.get("api") || saved || (location.protocol.startsWith("http") && !staticHost ? location.origin : "");
  let abort = null;
  endpoint.value = api;
  if (location.protocol.startsWith("http")) {
    embed.textContent = '<iframe src="' + location.origin + location.pathname + '" width="100%" height="760" style="border:0;border-radius:16px" title="cactus-rs TurboLM"></iframe>';
  }

  const fmt = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n);
  const base = () => api.trim().replace(/\/+$/, "");
  const setStatus = (text, kind) => { status.textContent = text; dot.className = "dot" + (kind ? " " + kind : ""); };

  async function check() {
    api = endpoint.value.trim().replace(/\/+$/, "");
    try { localStorage.setItem("cactus-api", api); } catch (e) {}
    if (!api) { setStatus("enter an API endpoint", "err"); meta.textContent = "Static hosting: point this page at a running cactus serve instance."; return; }
    setStatus("checking API…", "");
    try {
      const r = await fetch(base() + "/api/info", { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();
      setStatus("API online", "ok");
      sizeBadge.textContent = (d.file_bytes / 1e6).toFixed(3) + " MB";
      meta.innerHTML = "";
      const bits = [
        ["logical params", fmt(d.logical_parameters)],
        ["stored params", fmt(d.stored_parameters)],
        ["context", d.config.context],
        ["rank", d.config.rank],
        ["device", "CPU"],
      ];
      for (const [k, v] of bits) {
        const s = document.createElement("span");
        s.innerHTML = "<b>" + v + "</b> " + k;
        meta.appendChild(s);
      }
    } catch (e) {
      setStatus("API offline", "err");
      meta.textContent = "Start it with: cactus serve --model turbo.bin --tokenizer tokenizer.json";
    }
  }

  async function generate() {
    if (!base()) { endpoint.focus(); setStatus("enter an API endpoint", "err"); return; }
    go.disabled = true; stop.disabled = false;
    out.textContent = "";
    setStatus("generating…", "");
    abort = new AbortController();
    try {
      const r = await fetch(base() + "/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.value,
          length: Number($("length").value),
          temperature: Number($("temp").value),
        }),
        signal: abort.signal,
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
      out.textContent = d.text || "";
      setStatus("done", "ok");
    } catch (e) {
      if (e.name === "AbortError") setStatus("stopped", "");
      else { out.textContent = "Error: " + e.message; setStatus("request failed", "err"); }
    } finally {
      go.disabled = false; stop.disabled = true; abort = null;
    }
  }

  $("length").addEventListener("input", (e) => $("lengthVal").textContent = e.target.value);
  $("temp").addEventListener("input", (e) => $("tempVal").textContent = Number(e.target.value).toFixed(2));
  endpoint.addEventListener("change", check);
  go.addEventListener("click", generate);
  stop.addEventListener("click", () => abort && abort.abort());
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(out.textContent); setStatus("copied", "ok"); }
    catch { setStatus("copy failed", "err"); }
  });
  prompt.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate(); });

  check();
})();
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_request_lines() {
        assert_eq!(
            parse_request_line("POST /api/generate?x=1 HTTP/1.1\r\n"),
            Some(("POST".to_string(), "/api/generate".to_string()))
        );
        assert_eq!(
            parse_request_line("GET / HTTP/1.0\n"),
            Some(("GET".to_string(), "/".to_string()))
        );
        assert!(parse_request_line("not-http").is_none());
        assert!(parse_request_line("GET /things").is_none());
    }

    #[test]
    fn clamps_generation_options() {
        assert_eq!(clamp_length(None), 300);
        assert_eq!(clamp_length(Some(0)), 1);
        assert_eq!(clamp_length(Some(999_999)), 2000);
        assert_eq!(clamp_temperature(None), 0.8);
        assert_eq!(clamp_temperature(Some(0.0)), 0.01);
        assert_eq!(clamp_temperature(Some(9.0)), 2.0);
        assert_eq!(clamp_temperature(Some(f32::NAN)), 0.8);
    }

    #[test]
    fn ui_and_openapi_are_wired() {
        assert!(INDEX_HTML.contains("/api/generate"));
        assert!(INDEX_HTML.contains("?api="));
        assert!(INDEX_HTML.contains("github"));
        let spec = openapi();
        assert_eq!(spec["openapi"], "3.1.0");
        assert!(spec["paths"]["/api/generate"]["post"].is_object());
        assert!(spec["paths"]["/health"]["get"].is_object());
    }

    #[test]
    fn http_generate_roundtrip() {
        use crate::TurboConfig;
        use std::io::{Read, Write};

        let tokenizer = Tokenizer::from_text("hello world ");
        let config = TurboConfig {
            vocab_size: tokenizer.vocab_size(),
            embed_dim: 4,
            hidden_dim: 8,
            context: 4,
            rank: 2,
        };
        let app = Arc::new(App {
            model: Mutex::new(TurboLM::new(config)),
            tokenizer,
            model_bytes: 123,
        });

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle(stream, &app).unwrap();
        });

        let body = br#"{"prompt":"hello","length":3,"temperature":0.8}"#;
        let mut client = TcpStream::connect(addr).unwrap();
        write!(
            client,
            "POST /api/generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        client.write_all(body).unwrap();

        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains("Access-Control-Allow-Origin: *"), "{response}");
        assert!(response.contains("\"text\":"), "{response}");
        server.join().unwrap();
    }
}
