FROM rust:1.75-bookworm as builder

# Install LLVM/Clang (Required for bpf-linker)
RUN apt-get update && apt-get install -y \
    llvm \
    clang \
    libclang-dev \
    gcc-multilib \
    && rm -rf /var/lib/apt/lists/*

# Install bpf-linker
RUN cargo install bpf-linker

# Setup Nightly for eBPF (Required for build-std)
RUN rustup toolchain install nightly
RUN rustup component add rust-src --toolchain nightly
RUN rustup target add bpfel-unknown-none --toolchain nightly

WORKDIR /app
COPY . .

# Build Kernel Program (needs nightly + bpf target)
# Correct path for artifact: target/bpfel-unknown-none/debug/ebpf_program
RUN cargo +nightly build --package ebpf_program --target bpfel-unknown-none -Z build-std=core

# Build User Loader
RUN cargo build --release --package user_loader

# Final Runtime Image (Small, but needs glibc, so using debian:bookworm-slim)
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates iproute2 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/user_loader ./user_loader
# Copy the compiled BPF ELF to the expected location relative to binary
# Loader expects: ../../target/bpfel-unknown-none/debug/ebpf_program
# We might need to adjust loader path or copy file to match structure.
# But `include_bytes!` embeds it into the binary at BUILD time! 
# So we DON'T need to copy the BPF file to runtime image.
# The binary `user_loader` is self-contained.

CMD ["./user_loader", "--iface", "eth0"]
