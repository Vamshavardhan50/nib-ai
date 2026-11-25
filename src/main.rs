//! CLI for nib-rs — from-scratch neural network (text + vision).
//!
//! Commands:
//!   train           Train a char-level language model on a text file
//!   generate        Generate text from a trained model
//!   turbo-*         Compact 100M-shape model commands
//!   dense-*         True 90M-weight dense model, 4-bit (45 MB)
//!   serve           Serve a Turbo or dense model over HTTP with a browser UI
//!   vision-train    Train a from-scratch CNN on labeled images
//!   vision-predict  Classify an image with a trained CNN
//!   version         Print version

use anyhow::{Context, Result};
use clap::Parser;
use rand::Rng;

use nib_rs::{CharLM, DenseTrainer, Init, ModelConfig, Tokenizer, Trainer, TurboConfig, TurboLM, TurboTrainer, VisionCNN, VisionTrainer};

#[derive(Parser, Debug)]
#[command(author, version, about = "nib-rs: from-scratch neural network for developers", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Train a from-scratch char-level language model on a text file.
    Train {
        #[arg(long, short = 'i', default_value = "data.txt")]
        input: String,
        #[arg(long, short = 'o', default_value = "model.json")]
        output: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        #[arg(long, default_value_t = 8)]
        context: usize,
        #[arg(long, default_value_t = 64)]
        hidden: usize,
        #[arg(long, default_value_t = 16)]
        embed: usize,
        #[arg(long, default_value_t = 0.1)]
        lr: f32,
        #[arg(long, short = 'e', default_value_t = 200)]
        epochs: usize,
        #[arg(long, default_value_t = String::from("sgd"))]
        optimizer: String,
    },
    /// Generate text from a trained model.
    Generate {
        #[arg(long, short = 'm', default_value = "model.json")]
        model: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        #[arg(long, short = 's', default_value = "")]
        seed: String,
        #[arg(long, default_value_t = 200)]
        length: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
    },
    /// Train a from-scratch CNN on labeled images.
    ///
    /// The manifest is a text file with one `image_path label` pair per line,
    /// e.g.:
    ///   data/cats/001.jpg 0
    ///   data/dogs/002.jpg 1
    #[command(name = "vision-train")]
    VisionTrain {
        #[arg(long, short = 'i', default_value = "manifest.txt")]
        manifest: String,
        #[arg(long, short = 'o', default_value = "vision.json")]
        output: String,
        #[arg(long, default_value_t = 16)]
        image_size: usize,
        #[arg(long, default_value_t = 1)]
        channels: usize,
        #[arg(long, default_value_t = 10)]
        classes: usize,
        #[arg(long, default_value_t = 0.01)]
        lr: f32,
        #[arg(long, short = 'e', default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
    },
    /// Classify an image with a trained CNN.
    #[command(name = "vision-predict")]
    VisionPredict {
        #[arg(long, short = 'm', default_value = "vision.json")]
        model: String,
        #[arg(long, short = 'i')]
        image: String,
        /// Optional class names file (one name per line).
        #[arg(long)]
        classes: Option<String>,
    },
    /// Print Turbo model parameters and compressed file size.
    #[command(name = "turbo-info")]
    TurboInfo {
        /// Dense logical parameter target (the compressed model keeps this shape).
        #[arg(long, default_value_t = 100_000_000)]
        target: usize,
        #[arg(long, default_value_t = 62)]
        vocab: usize,
        #[arg(long, default_value_t = 64)]
        context: usize,
        #[arg(long, default_value_t = 128)]
        embed: usize,
        #[arg(long, default_value_t = 30)]
        rank: usize,
        /// Read vocab size from a tokenizer.json instead of --vocab.
        #[arg(long)]
        tokenizer: Option<String>,
    },
    /// Train a compact Turbo model (low-rank + int8 on disk).
    #[command(name = "turbo-train")]
    TurboTrain {
        #[arg(long, short = 'i', default_value = "shakespeare.txt")]
        input: String,
        #[arg(long, short = 'o', default_value = "turbo.bin")]
        output: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        #[arg(long, default_value_t = 100_000_000)]
        target: usize,
        #[arg(long, default_value_t = 64)]
        context: usize,
        #[arg(long, default_value_t = 128)]
        embed: usize,
        #[arg(long, default_value_t = 30)]
        rank: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, short = 'e', default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        /// 0 = use every window in the file.
        #[arg(long, default_value_t = 0)]
        max_samples: usize,
    },
    /// Generate text from a compact Turbo model.
    #[command(name = "turbo-generate")]
    TurboGenerate {
        #[arg(long, short = 'm', default_value = "turbo.bin")]
        model: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        #[arg(long, short = 's', default_value = "ROMEO: ")]
        seed: String,
        #[arg(long, default_value_t = 500)]
        length: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
    },
    /// Print dense parameter counts and 4-bit file size.
    #[command(name = "dense-info")]
    DenseInfo {
        #[arg(long, default_value_t = 100_000_000)]
        target: usize,
        #[arg(long, default_value_t = 62)]
        vocab: usize,
        #[arg(long, default_value_t = 64)]
        context: usize,
        #[arg(long, default_value_t = 128)]
        embed: usize,
        /// 0 = derive hidden size from --target.
        #[arg(long, default_value_t = 0)]
        hidden: usize,
        /// Read vocab size from a tokenizer.json instead of --vocab.
        #[arg(long)]
        tokenizer: Option<String>,
    },
    /// Materialize a trained Turbo model into a true dense 4-bit file.
    #[command(name = "dense-build")]
    DenseBuild {
        #[arg(long, default_value = "turbo.bin")]
        turbo: String,
        #[arg(long, short = 'o', default_value = "nib.bin")]
        output: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
    },
    /// Train a dense model directly, or fine-tune/prune from a checkpoint.
    #[command(name = "dense-train")]
    DenseTrain {
        #[arg(long, short = 'i', default_value = "shakespeare.txt")]
        input: String,
        #[arg(long, short = 'o', default_value = "nib.bin")]
        output: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        /// Optional trained Turbo model to expand and fine-tune.
        #[arg(long)]
        init_turbo: Option<String>,
        /// Optional dense 4-bit model to continue training (pruned with --hidden).
        #[arg(long)]
        init_dense: Option<String>,
        /// Used only when training from random weights; ignored with --init-turbo.
        #[arg(long, default_value_t = 100_000_000)]
        target: usize,
        /// Used only when training from random weights; ignored with --init-turbo.
        #[arg(long, default_value_t = 64)]
        context: usize,
        /// Used only when training from random weights; ignored with --init-turbo.
        #[arg(long, default_value_t = 128)]
        embed: usize,
        /// Random init: derive hidden size. With --init-dense: prune to this size.
        #[arg(long, default_value_t = 0)]
        hidden: usize,
        #[arg(long, default_value_t = 0.02)]
        lr: f32,
        #[arg(long, short = 'e', default_value_t = 1)]
        epochs: usize,
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        /// 0 = every window in the file. Dense training is much slower.
        #[arg(long, default_value_t = 20_000)]
        max_samples: usize,
    },
    /// Generate text from a dense 4-bit model.
    #[command(name = "dense-generate")]
    DenseGenerate {
        #[arg(long, short = 'm', default_value = "nib.bin")]
        model: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        #[arg(long, short = 's', default_value = "ROMEO: ")]
        seed: String,
        #[arg(long, default_value_t = 300)]
        length: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
    },
    /// Serve a Turbo or dense model over HTTP with a browser UI and JSON API.
    Serve {
        #[arg(long, short = 'm', default_value = "turbo.bin")]
        model: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        /// 0.0.0.0 works in Docker/containers; use 127.0.0.1 for local-only.
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Write the browser UI to an HTML file for GitHub Pages or any static host.
    #[command(name = "web-export")]
    WebExport {
        #[arg(long, short = 'o', default_value = "index.html")]
        output: String,
    },
    /// Print version.
    Version,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train {
            input,
            output,
            tokenizer,
            context,
            hidden,
            embed,
            lr,
            epochs,
            optimizer,
        } => train(&input, &output, &tokenizer, context, hidden, embed, lr, epochs, &optimizer),
        Commands::Generate {
            model,
            tokenizer,
            seed,
            length,
            temperature,
        } => generate(&model, &tokenizer, &seed, length, temperature),
        Commands::VisionTrain {
            manifest,
            output,
            image_size,
            channels,
            classes,
            lr,
            epochs,
            batch_size,
        } => vision_train(&manifest, &output, image_size, channels, classes, lr, epochs, batch_size),
        Commands::VisionPredict {
            model,
            image,
            classes,
        } => vision_predict(&model, &image, classes.as_deref()),
        Commands::TurboInfo {
            target,
            vocab,
            context,
            embed,
            rank,
            tokenizer,
        } => turbo_info(target, vocab, context, embed, rank, tokenizer.as_deref()),
        Commands::TurboTrain {
            input,
            output,
            tokenizer,
            target,
            context,
            embed,
            rank,
            lr,
            epochs,
            batch_size,
            max_samples,
        } => turbo_train(
            &input,
            &output,
            &tokenizer,
            target,
            context,
            embed,
            rank,
            lr,
            epochs,
            batch_size,
            max_samples,
        ),
        Commands::TurboGenerate {
            model,
            tokenizer,
            seed,
            length,
            temperature,
        } => turbo_generate(&model, &tokenizer, &seed, length, temperature),
        Commands::DenseInfo {
            target,
            vocab,
            context,
            embed,
            hidden,
            tokenizer,
        } => dense_info(target, vocab, context, embed, hidden, tokenizer.as_deref()),
        Commands::DenseBuild {
            turbo,
            output,
            tokenizer,
        } => dense_build(&turbo, &output, &tokenizer),
        Commands::DenseTrain {
            input,
            output,
            tokenizer,
            init_turbo,
            init_dense,
            target,
            context,
            embed,
            hidden,
            lr,
            epochs,
            batch_size,
            max_samples,
        } => dense_train(
            &input,
            &output,
            &tokenizer,
            init_turbo.as_deref(),
            init_dense.as_deref(),
            target,
            context,
            embed,
            hidden,
            lr,
            epochs,
            batch_size,
            max_samples,
        ),
        Commands::DenseGenerate {
            model,
            tokenizer,
            seed,
            length,
            temperature,
        } => dense_generate(&model, &tokenizer, &seed, length, temperature),
        Commands::Serve {
            model,
            tokenizer,
            host,
            port,
        } => nib_rs::web::serve(&model, &tokenizer, &host, port),
        Commands::WebExport { output } => {
            std::fs::write(&output, nib_rs::web::index_html())
                .with_context(|| format!("write web UI to {output}"))?;
            println!("web UI written to {output}");
            println!("host it anywhere static, then add ?api=https://your-nib-host");
            Ok(())
        }
        Commands::Version => {
            println!("nib-rs {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
    }
}

fn train(
    input: &str,
    output: &str,
    tokenizer_path: &str,
    context: usize,
    hidden: usize,
    embed: usize,
    lr: f32,
    epochs: usize,
    optimizer: &str,
) -> Result<()> {
    let text = std::fs::read_to_string(input)
        .with_context(|| format!("read input file {input}"))?;
    if text.is_empty() {
        anyhow::bail!("input file {input} is empty");
    }

    let tok = Tokenizer::from_text(&text);
    tok.save(tokenizer_path)?;

    let config = ModelConfig {
        vocab_size: tok.vocab_size(),
        embed_dim: embed,
        hidden_dim: hidden,
        context,
        init: nib_rs::Init::Xavier,
    };
    let model = CharLM::new(config);
    let mut trainer = Trainer::new(model, optimizer.to_string(), lr);

    let ids = tok.encode(&text);
    let n = ids.len();
    if n <= context {
        anyhow::bail!("input text too short for context size {context}");
    }

    println!(
        "training: vocab={} context={} hidden={} embed={} epochs={} opt={}",
        tok.vocab_size(), context, hidden, embed, epochs, optimizer
    );

    for epoch in 0..epochs {
        let mut batches: Vec<(Vec<usize>, usize)> = (0..n - context)
            .map(|i| {
                let ctx: Vec<usize> = ids[i..i + context].to_vec();
                let target = ids[i + context];
                (ctx, target)
            })
            .collect();

        let mut rng = rand::thread_rng();
        for i in (1..batches.len()).rev() {
            let j = rng.gen_range(0..=i);
            batches.swap(i, j);
        }

        let batch_size: usize = 32;
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        for chunk in batches.chunks(batch_size) {
            let loss = trainer.train_step(chunk);
            epoch_loss += loss;
            num_batches += 1;
        }
        let avg = epoch_loss / num_batches.max(1) as f32;

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("epoch {epoch}/{epochs} loss: {:.4}", avg);
        }
    }

    trainer.save(output)?;
    println!("model saved to {output}");
    println!("tokenizer saved to {tokenizer_path}");
    Ok(())
}

fn generate(
    model_path: &str,
    tokenizer_path: &str,
    seed: &str,
    length: usize,
    temperature: f32,
) -> Result<()> {
    let mut model = CharLM::load(model_path)
        .with_context(|| format!("load model {model_path}"))?;
    let tok = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("load tokenizer {tokenizer_path}"))?;

    let context = model.config.context;
    let mut seed_ids = tok.encode(seed);
    if seed_ids.len() > context {
        seed_ids = seed_ids[seed_ids.len() - context..].to_vec();
    }
    while seed_ids.len() < context {
        seed_ids.insert(0, 0);
    }

    let mut generated: String = seed.to_string();
    let mut ctx = seed_ids;

    for _ in 0..length {
        let probs = model.predict(&ctx);
        let next = CharLM::sample(&probs, temperature);
        let next_char = tok.id_to_char.get(&next).copied().unwrap_or('?');
        generated.push(next_char);
        ctx.remove(0);
        ctx.push(next);
    }

    print!("{generated}");
    Ok(())
}

fn turbo_info(
    target: usize,
    vocab: usize,
    context: usize,
    embed: usize,
    rank: usize,
    tokenizer_path: Option<&str>,
) -> Result<()> {
    if context == 0 || embed == 0 || rank == 0 {
        anyhow::bail!("context, embed, and rank must all be at least 1");
    }
    let vocab = match tokenizer_path {
        Some(path) => Tokenizer::load(path)
            .with_context(|| format!("load tokenizer {path}"))?
            .vocab_size(),
        None => vocab,
    };

    let config = TurboConfig {
        vocab_size: vocab,
        embed_dim: embed,
        hidden_dim: 64,
        context,
        rank,
    };
    let model = TurboLM::new(config).with_logical_target(target);
    let logical = model.logical_params();
    let stored = model.stored_params();
    let bytes = model.quantized_bytes();

    println!("Turbo compact language model");
    println!(
        "shape: vocab={} context={} embed={} hidden={} rank={}",
        model.config.vocab_size,
        model.config.context,
        model.config.embed_dim,
        model.config.hidden_dim,
        model.config.rank
    );
    println!(
        "logical (dense) parameters: {} ({:.2}M)",
        logical,
        logical as f64 / 1_000_000.0
    );
    println!(
        "stored parameters:          {} ({:.3}M)",
        stored,
        stored as f64 / 1_000_000.0
    );
    println!("estimated file size:        {} bytes ({:.3} MB)", bytes, bytes as f64 / 1_000_000.0);
    println!(
        "dense f32 equivalent:       {:.1} MB",
        logical as f64 * 4.0 / 1_000_000.0
    );
    println!("compression vs f32:         {:.0}x", logical as f64 * 4.0 / bytes as f64);
    if bytes > 1_000_000 {
        println!("warning: estimated file is over 1 MB — lower --rank or --target.");
    }
    println!();
    println!(
        "Note: the 100M-entry matrix is stored as two factors (rank {}), so the",
        model.config.rank
    );
    println!("number of independent weights is the stored count, not the dense shape.");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn turbo_train(
    input: &str,
    output: &str,
    tokenizer_path: &str,
    target: usize,
    context: usize,
    embed: usize,
    rank: usize,
    lr: f32,
    epochs: usize,
    batch_size: usize,
    max_samples: usize,
) -> Result<()> {
    if context == 0 || embed == 0 || rank == 0 {
        anyhow::bail!("context, embed, and rank must all be at least 1");
    }
    let text = std::fs::read_to_string(input)
        .with_context(|| format!("read input file {input}"))?;
    if text.is_empty() {
        anyhow::bail!("input file {input} is empty");
    }

    let tok = Tokenizer::from_text(&text);
    tok.save(tokenizer_path)?;
    let ids = tok.encode(&text);
    if ids.len() <= context {
        anyhow::bail!("input text too short for context size {context}");
    }

    let config = TurboConfig {
        vocab_size: tok.vocab_size(),
        embed_dim: embed,
        hidden_dim: 64,
        context,
        rank,
    };
    let model = TurboLM::new(config).with_logical_target(target);
    let logical = model.logical_params();
    let stored = model.stored_params();
    let estimated = model.quantized_bytes();

    println!(
        "turbo training: vocab={} context={} embed={} hidden={} rank={}",
        model.config.vocab_size,
        model.config.context,
        model.config.embed_dim,
        model.config.hidden_dim,
        model.config.rank
    );
    println!(
        "logical={} ({:.2}M) stored={} ({:.3}M) estimated_file={} bytes",
        logical,
        logical as f64 / 1_000_000.0,
        stored,
        stored as f64 / 1_000_000.0,
        estimated
    );
    if estimated > 1_000_000 {
        println!("warning: estimated file is over 1 MB — lower --rank or --target.");
    }

    let mut trainer = TurboTrainer::new(model, lr);
    let n = ids.len() - context;
    let take = if max_samples == 0 { n } else { n.min(max_samples) };
    if take == 0 {
        anyhow::bail!("no training windows available");
    }
    println!("training windows per epoch: {take} of {n} (batch={batch_size})");

    for epoch in 0..epochs {
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();
        for i in (1..order.len()).rev() {
            let j = rng.gen_range(0..=i);
            order.swap(i, j);
        }

        let mut epoch_loss = 0.0;
        let mut num_batches = 0usize;
        for chunk in order[..take].chunks(batch_size.max(1)) {
            let batch: Vec<(Vec<usize>, usize)> = chunk
                .iter()
                .map(|&i| (ids[i..i + context].to_vec(), ids[i + context]))
                .collect();
            epoch_loss += trainer.train_step(&batch);
            num_batches += 1;
        }
        let avg = epoch_loss / num_batches.max(1) as f32;
        println!("epoch {epoch}/{epochs} loss: {:.4}", avg);
    }

    trainer.save(output)?;
    let bytes = std::fs::metadata(output)?.len();
    println!(
        "turbo model saved to {output} ({} bytes, {:.3} MB)",
        bytes,
        bytes as f64 / 1_000_000.0
    );
    println!(
        "logical params: {} ({:.2}M), stored params: {} ({:.3}M)",
        trainer.model.logical_params(),
        trainer.model.logical_params() as f64 / 1_000_000.0,
        trainer.model.stored_params(),
        trainer.model.stored_params() as f64 / 1_000_000.0
    );
    Ok(())
}

fn turbo_generate(
    model_path: &str,
    tokenizer_path: &str,
    seed: &str,
    length: usize,
    temperature: f32,
) -> Result<()> {
    let mut model = TurboLM::load(model_path)
        .with_context(|| format!("load turbo model {model_path}"))?;
    let tok = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("load tokenizer {tokenizer_path}"))?;
    if tok.vocab_size() != model.config.vocab_size {
        anyhow::bail!(
            "tokenizer vocab ({}) does not match model vocab ({})",
            tok.vocab_size(),
            model.config.vocab_size
        );
    }

    let generated = nib_rs::turbo::generate_text(&mut model, &tok, seed, length, temperature);

    print!("{generated}");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    Ok(())
}

fn dense_config(
    target: usize,
    vocab: usize,
    context: usize,
    embed: usize,
    hidden: usize,
) -> Result<ModelConfig> {
    if vocab == 0 || context == 0 || embed == 0 {
        anyhow::bail!("vocab, context, and embed must all be at least 1");
    }
    let hidden = if hidden > 0 {
        hidden
    } else {
        let denom = context * embed + vocab;
        target.saturating_sub(vocab * embed) / denom.max(1)
    };
    if hidden < 2 {
        anyhow::bail!(
            "hidden size is too small ({hidden}); raise --target or pass --hidden"
        );
    }
    Ok(ModelConfig {
        vocab_size: vocab,
        embed_dim: embed,
        hidden_dim: hidden,
        context,
        init: Init::Xavier,
    })
}

fn dense_info(
    target: usize,
    vocab: usize,
    context: usize,
    embed: usize,
    hidden: usize,
    tokenizer_path: Option<&str>,
) -> Result<()> {
    let vocab = match tokenizer_path {
        Some(path) => Tokenizer::load(path)
            .with_context(|| format!("load tokenizer {path}"))?
            .vocab_size(),
        None => vocab,
    };
    let config = dense_config(target, vocab, context, embed, hidden)?;
    let params = nib_rs::dense::params(&config);
    let bytes = nib_rs::dense::quantized_bytes(&config);
    println!("Dense language model (4-bit)");
    println!(
        "shape: vocab={} context={} embed={} hidden={}",
        config.vocab_size, config.context, config.embed_dim, config.hidden_dim
    );
    println!(
        "dense parameters:          {} ({:.2}M)",
        params,
        params as f64 / 1_000_000.0
    );
    println!(
        "4-bit file estimate:       {} bytes ({:.2} MB)",
        bytes,
        bytes as f64 / 1_000_000.0
    );
    println!(
        "dense f32 equivalent:      {:.1} MB",
        params as f64 * 4.0 / 1_000_000.0
    );
    if bytes > 100_000_000 {
        println!("warning: estimated file is over 100 MB — lower --target.");
    }
    Ok(())
}

fn dense_build(turbo_path: &str, output: &str, tokenizer_path: &str) -> Result<()> {
    let turbo = TurboLM::load(turbo_path)
        .with_context(|| format!("load turbo model {turbo_path}"))?;
    let tok = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("load tokenizer {tokenizer_path}"))?;
    if tok.vocab_size() != turbo.config.vocab_size {
        anyhow::bail!(
            "tokenizer vocab ({}) does not match model vocab ({})",
            tok.vocab_size(),
            turbo.config.vocab_size
        );
    }

    println!(
        "materializing dense model from rank-{} factors...",
        turbo.config.rank
    );
    let dense = nib_rs::dense::from_turbo(&turbo);
    let params = nib_rs::dense::params(&dense.config);
    let expected = nib_rs::dense::quantized_bytes(&dense.config);
    nib_rs::dense::save_int4(&dense, output)?;
    let bytes = std::fs::metadata(output)?.len();
    println!(
        "dense model saved to {output} ({} bytes, {:.2} MB)",
        bytes,
        bytes as f64 / 1_000_000.0
    );
    println!(
        "dense parameters: {} ({:.2}M), estimated 4-bit file: {} bytes",
        params,
        params as f64 / 1_000_000.0,
        expected
    );
    if bytes > 100_000_000 {
        anyhow::bail!("dense file is over 100 MB");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dense_train(
    input: &str,
    output: &str,
    tokenizer_path: &str,
    init_turbo: Option<&str>,
    init_dense: Option<&str>,
    target: usize,
    context: usize,
    embed: usize,
    hidden: usize,
    lr: f32,
    epochs: usize,
    batch_size: usize,
    max_samples: usize,
) -> Result<()> {
    if batch_size == 0 {
        anyhow::bail!("batch-size must be at least 1");
    }
    let text = std::fs::read_to_string(input)
        .with_context(|| format!("read input file {input}"))?;
    if text.is_empty() {
        anyhow::bail!("input file {input} is empty");
    }

    let (tok, model) = if let Some(dense_path) = init_dense {
        let tok = Tokenizer::load(tokenizer_path)
            .with_context(|| format!("load tokenizer {tokenizer_path}"))?;
        let mut model = nib_rs::dense::load_int4(dense_path)
            .with_context(|| format!("load dense model {dense_path}"))?;
        if tok.vocab_size() != model.config.vocab_size {
            anyhow::bail!(
                "tokenizer vocab ({}) does not match model vocab ({})",
                tok.vocab_size(),
                model.config.vocab_size
            );
        }
        if hidden > 0 && hidden < model.config.hidden_dim {
            println!(
                "pruning hidden units {} -> {} using fc2 weight norms",
                model.config.hidden_dim, hidden
            );
            model = nib_rs::dense::prune_hidden(&model, hidden)?;
        }
        (tok, model)
    } else if let Some(turbo_path) = init_turbo {
        let turbo = TurboLM::load(turbo_path)
            .with_context(|| format!("load turbo model {turbo_path}"))?;
        let tok = Tokenizer::load(tokenizer_path)
            .with_context(|| format!("load tokenizer {tokenizer_path}"))?;
        if tok.vocab_size() != turbo.config.vocab_size {
            anyhow::bail!(
                "tokenizer vocab ({}) does not match model vocab ({})",
                tok.vocab_size(),
                turbo.config.vocab_size
            );
        }
        println!(
            "initializing dense model from {turbo_path} (rank {})",
            turbo.config.rank
        );
        (tok, nib_rs::dense::from_turbo(&turbo))
    } else {
        let tok = Tokenizer::from_text(&text);
        tok.save(tokenizer_path)?;
        let config = dense_config(target, tok.vocab_size(), context, embed, hidden)?;
        (tok, CharLM::new(config))
    };
    let config = model.config.clone();
    let params = nib_rs::dense::params(&config);
    let estimated = nib_rs::dense::quantized_bytes(&config);
    println!(
        "dense training: vocab={} context={} embed={} hidden={} params={} ({:.2}M)",
        config.vocab_size,
        config.context,
        config.embed_dim,
        config.hidden_dim,
        params,
        params as f64 / 1_000_000.0
    );
    println!(
        "estimated 4-bit file: {} bytes ({:.2} MB)",
        estimated,
        estimated as f64 / 1_000_000.0
    );
    if estimated > 100_000_000 {
        println!("warning: estimated file is over 100 MB — lower --target.");
    }

    let ids = tok.encode(&text);
    if ids.len() <= config.context {
        anyhow::bail!("input text too short for context size {}", config.context);
    }
    let n = ids.len() - config.context;
    let take = if max_samples == 0 { n } else { n.min(max_samples) };
    if take == 0 {
        anyhow::bail!("no training windows available");
    }
    println!(
        "training windows per epoch: {take} of {n} (batch={batch_size}, lr={lr})"
    );

    let mut trainer = DenseTrainer::new(model, lr);
    for epoch in 0..epochs {
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();
        for i in (1..order.len()).rev() {
            let j = rng.gen_range(0..=i);
            order.swap(i, j);
        }

        let mut epoch_loss = 0.0;
        let mut num_batches = 0usize;
        let total_batches = take.div_ceil(batch_size);
        let log_every = (total_batches / 10).max(1);
        for chunk in order[..take].chunks(batch_size) {
            let batch: Vec<(Vec<usize>, usize)> = chunk
                .iter()
                .map(|&i| {
                    (
                        ids[i..i + config.context].to_vec(),
                        ids[i + config.context],
                    )
                })
                .collect();
            epoch_loss += trainer.train_step(&batch);
            num_batches += 1;
            if num_batches % log_every == 0 || num_batches == total_batches {
                println!(
                    "epoch {epoch}/{epochs} batch {num_batches}/{total_batches} loss: {:.4}",
                    epoch_loss / num_batches as f32
                );
            }
        }
        println!(
            "epoch {epoch}/{epochs} loss: {:.4}",
            epoch_loss / num_batches.max(1) as f32
        );
    }

    trainer.save(output)?;
    let bytes = std::fs::metadata(output)?.len();
    println!(
        "dense model saved to {output} ({} bytes, {:.2} MB)",
        bytes,
        bytes as f64 / 1_000_000.0
    );
    if bytes > 100_000_000 {
        anyhow::bail!("dense file is over 100 MB");
    }
    let params = nib_rs::dense::params(&trainer.model.config);
    println!(
        "dense parameters: {} ({:.2}M)",
        params,
        params as f64 / 1_000_000.0
    );
    Ok(())
}

fn dense_generate(
    model_path: &str,
    tokenizer_path: &str,
    seed: &str,
    length: usize,
    temperature: f32,
) -> Result<()> {
    let mut model = nib_rs::dense::load_int4(model_path)
        .with_context(|| format!("load dense model {model_path}"))?;
    let tok = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("load tokenizer {tokenizer_path}"))?;
    if tok.vocab_size() != model.config.vocab_size {
        anyhow::bail!(
            "tokenizer vocab ({}) does not match model vocab ({})",
            tok.vocab_size(),
            model.config.vocab_size
        );
    }

    let generated =
        nib_rs::dense::generate_text(&mut model, &tok, seed, length, temperature);
    print!("{generated}");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    Ok(())
}

fn vision_train(
    manifest_path: &str,
    output: &str,
    image_size: usize,
    channels: usize,
    classes: usize,
    lr: f32,
    epochs: usize,
    batch_size: usize,
) -> Result<()> {
    let entries = nib_rs::load_manifest(manifest_path)?;
    if entries.is_empty() {
        anyhow::bail!("manifest {manifest_path} has no entries");
    }

    println!(
        "vision training: {} images, {} classes, {}x{}x{}, epochs={}, lr={}, batch={}",
        entries.len(), classes, channels, image_size, image_size, epochs, lr, batch_size
    );

    // Preload all images once.
    let mut images = Vec::with_capacity(entries.len());
    let mut labels = Vec::with_capacity(entries.len());
    for (path, label) in &entries {
        let img = nib_rs::load_image(path, image_size, image_size, channels)
            .with_context(|| format!("load image {path}"))?;
        images.push(img);
        labels.push(*label);
    }

    let cnn = VisionCNN::new(channels, image_size, image_size, classes);
    let mut trainer = VisionTrainer::new(cnn, lr);

    for epoch in 0..epochs {
        // Shuffle indices each epoch.
        let mut order: Vec<usize> = (0..images.len()).collect();
        let mut rng = rand::thread_rng();
        for i in (1..order.len()).rev() {
            let j = rng.gen_range(0..=i);
            order.swap(i, j);
        }

        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        for chunk in order.chunks(batch_size) {
            let batch_imgs: Vec<nib_rs::Tensor4> =
                chunk.iter().map(|&i| images[i].clone()).collect();
            let batch_labels: Vec<usize> = chunk.iter().map(|&i| labels[i]).collect();
            let loss = trainer.train_step(&batch_imgs, &batch_labels);
            epoch_loss += loss;
            num_batches += 1;
        }
        let avg = epoch_loss / num_batches.max(1) as f32;
        println!("epoch {epoch}/{epochs} loss: {:.4}", avg);
    }

    trainer.save(output)?;
    println!("vision model saved to {output}");
    Ok(())
}

fn vision_predict(model_path: &str, image_path: &str, classes_path: Option<&str>) -> Result<()> {
    let mut model = VisionCNN::load(model_path)
        .with_context(|| format!("load vision model {model_path}"))?;

    let class_names: Option<Vec<String>> = match classes_path {
        Some(p) => Some(
            std::fs::read_to_string(p)
                .with_context(|| format!("read classes file {p}"))?
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect(),
        ),
        None => None,
    };

    let img = nib_rs::load_image(
        image_path,
        model.input_h,
        model.input_w,
        model.input_channels,
    )?;

    let probs = model.predict(&img);
    let (best_idx, best_prob) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    let name = class_names
        .as_ref()
        .and_then(|names| names.get(best_idx).cloned())
        .unwrap_or_else(|| format!("class {best_idx}"));

    println!("prediction: {name} (class {best_idx})");
    println!("confidence: {:.4}", best_prob);
    Ok(())
}
