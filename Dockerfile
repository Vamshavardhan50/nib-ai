# Build the smallest release binary, then run it on a tiny Debian image.
# The model and tokenizer are trained locally (see README) and copied in.
# Select the model at build time: --build-arg MODEL=dense.bin
FROM rust:1-slim AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
ARG MODEL=turbo.bin
RUN useradd --create-home --uid 10001 cactus
WORKDIR /app
COPY --from=build /app/target/release/cactus /usr/local/bin/cactus
COPY ${MODEL} /app/model.bin
COPY tokenizer.json ./
USER cactus
EXPOSE 8080
ENV HOST=0.0.0.0 PORT=8080
CMD ["sh", "-c", "exec cactus serve --model model.bin --tokenizer tokenizer.json --host ${HOST} --port ${PORT}"]
