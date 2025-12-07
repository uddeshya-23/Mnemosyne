# eBPF Probe Prototype (Linux/WSL)

This directory contains a **Proof-of-Concept Kernel-Space Firewall** using Rust and eBPF (Aya framework).

## Prerequisites
*   **Linux Interface**: Real Linux kernel required (WSL2 or Native).
*   **Docker**: Easiest way to build and run (handles toolchains).
*   **Privileges**: eBPF requires `sudo` or `--privileged`.

## 🚀 How to Run (Docker Method)

This is the recommended way as it installs all dependencies (LLVM, bpf-linker, Nightly Rust) for you.

1.  **Build the Image**:
    ```bash
    cd ebpf_probe
    docker build -t ebpf-probe .
    ```

2.  **Run with Privileges**:
    *   We use `--privileged` to allow kernel loading.
    *   We use `--network host` to see real traffic.
    *   `RUST_LOG=info` enables the logs.
    
    ```bash
    docker run --privileged --network host -e RUST_LOG=info ebpf-probe
    ```

3.  **Generate Traffic**:
    In another terminal, ping or curl locally:
    ```bash
    curl http://localhost
    ```
    You should see logs in the Docker output:
    `INFO: Packet received on interface`

## 🛠️ Manual Build (Expert)
If you want to run natively in WSL:
1.  Install LLVM: `sudo apt install llvm clang libclang-dev`
2.  Install Tool: `cargo install bpf-linker`
3.  Add Target: `rustup target add bpfel-unknown-none --toolchain nightly`
4.  Build Kernel: 
    ```bash
    cargo +nightly build --package ebpf_program --target bpfel-unknown-none -Z build-std=core
    ```
5.  Run Loader:
    ```bash
    RUST_LOG=info cargo run --package user_loader -- --iface eth0
    ```
