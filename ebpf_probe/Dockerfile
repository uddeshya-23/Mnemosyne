FROM rust:1.76-bookworm as builder

# Install Dependencies for adding LLVM repo
RUN apt-get update && apt-get install -y \
    wget \
    lsb-release \
    software-properties-common \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install LLVM 16 (Required for modern bpf-linker)
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 16 && \
    rm llvm.sh

# Install Dependencies for bpf-linker
# Note: linking against llvm-16
RUN apt-get update && apt-get install -y \
    libllvm16 \
    llvm-16 \
    llvm-16-dev \
    clang-16 \
    libclang-16-dev \
    libpolly-16-dev \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Fix path and force environment for linkage
ENV LLVM_SYS_160_PREFIX=/usr/lib/llvm-16
ENV LD_LIBRARY_PATH=/usr/lib/llvm-16/lib

# Install bpf-linker
RUN cargo install bpf-linker --locked --no-default-features --features system-llvm

# Setup Nightly for eBPF
RUN rustup toolchain install nightly
RUN rustup component add rust-src --toolchain nightly
RUN rustup target add bpfel-unknown-none --toolchain nightly

WORKDIR /app
COPY . .

# Build Kernel Program
RUN cargo +nightly build --package ebpf_program --target bpfel-unknown-none -Z build-std=core

# Build User Loader
RUN cargo build --release --package user_loader

# Final Runtime Image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates iproute2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/user_loader ./user_loader
CMD ["./user_loader", "--iface", "eth0"]
