# eBPF Probe Prototype (Linux/WSL)

This directory contains a **Proof-of-Concept Kernel-Space Firewall** using Rust and eBPF (Aya framework).

## Prerequisites
*   **Linux Interface**: Real Linux kernel required (WSL2 or Native).
*   **Docker**: Easiest way to build and run (handles toolchains).
*   **Privileges**: eBPF requires `sudo` or `--privileged`.

## 🚀 How to Run (Native WSL - Recommended)

Since Docker build can be complex, running natively in WSL is often easier.

1.  **Install Prerequisites**:
    ```bash
    sudo apt update
    sudo apt install -y llvm clang libclang-dev gcc-multilib build-essential git pkg-config libssl-dev
    ```

2.  **Install bpf-linker**:
    ```bash
    cargo install bpf-linker
    ```

3.  **Setup Rust Nightly**:
    ```bash
    rustup toolchain install nightly
    rustup target add bpfel-unknown-none --toolchain nightly
    restup component add rust-src --toolchain nightly
    ```

4.  **Run the Loader**:
    ```bash
    # Build kernel (requires nightly)
    cargo +nightly build --package ebpf_program --target bpfel-unknown-none -Z build-std=core

    # Run user loader (needs sudo)
    sudo -E cargo run --package user_loader -- --iface eth0
    ```
    *(Note: `-E` preserves env vars if needed)*

## 🐳 Docker Method (Alternate)
