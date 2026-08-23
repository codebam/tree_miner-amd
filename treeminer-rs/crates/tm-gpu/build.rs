//! Compiles the Argon2 device kernels with `hipcc` and links them into the crate.
//!
//! The kernel is the only C++ left in the miner (see `PORT.md`), so this is the only place
//! that needs a GPU toolchain. When `hipcc` cannot be found the build still succeeds and
//! the crate falls back to a stub launch shim that reports "built without HIP" at runtime,
//! so a machine without ROCm can still `cargo check` the host logic. It cannot link a
//! binary: the HIP runtime calls in `src/hip.rs` are declared, not loaded dynamically.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Architectures compiled when nothing is configured and no device can be interrogated.
/// Covers the desktop RDNA parts and the CDNA accelerators the miner is likely to meet;
/// a fat binary costs build time, not runtime.
const FALLBACK_ARCHS: &[&str] = &["gfx1100", "gfx1030", "gfx90a", "gfx942"];

fn main() {
    println!("cargo:rustc-check-cfg=cfg(tm_gpu_stub)");
    println!("cargo:rerun-if-changed=kernel/argon2_kernel.hip");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TM_GPU_OFFLOAD_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    let rocm = rocm_path();

    let Some(hipcc) = find_hipcc(rocm.as_deref()) else {
        println!(
            "cargo:warning=hipcc not found (set HIP_PATH or ROCM_PATH); tm-gpu is building \
             without device kernels: `cargo check` and the library build work, but any \
             binary still needs libamdhip64 to link and every launch returns an error"
        );
        println!("cargo:rustc-cfg=tm_gpu_stub");
        return;
    };

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let object = out_dir.join("argon2_kernel.o");

    let mut command = Command::new(&hipcc);
    command
        .arg("-c")
        .arg("kernel/argon2_kernel.hip")
        .arg("-o")
        .arg(&object)
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-fPIC")
        .arg("-DTREEMINER_GPU_HIP=1");
    for arch in offload_archs(rocm.as_deref()) {
        command.arg(format!("--offload-arch={arch}"));
    }

    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", hipcc.display()));
    assert!(status.success(), "hipcc failed to compile the Argon2 kernel");

    let archive = out_dir.join("libtm_argon2_kernel.a");
    let _ = std::fs::remove_file(&archive);
    let ar_status = Command::new("ar")
        .arg("crs")
        .arg(&archive)
        .arg(&object)
        .status()
        .unwrap_or_else(|error| panic!("failed to run ar: {error}"));
    assert!(ar_status.success(), "ar failed to archive the Argon2 kernel");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=tm_argon2_kernel");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    if let Some(rocm) = rocm.as_deref() {
        let lib = Path::new(rocm).join("lib");
        println!("cargo:rustc-link-search=native={}", lib.display());
        // The HIP runtime is not on the default loader path in a nix environment, and test
        // binaries are run directly rather than through the toolchain wrapper.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}

fn rocm_path() -> Option<String> {
    std::env::var("HIP_PATH")
        .ok()
        .or_else(|| std::env::var("ROCM_PATH").ok())
        .filter(|path| !path.is_empty())
}

fn find_hipcc(rocm: Option<&str>) -> Option<PathBuf> {
    if let Some(rocm) = rocm {
        let candidate = Path::new(rocm).join("bin").join("hipcc");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join("hipcc"))
        .find(|candidate| candidate.is_file())
}

/// `TM_GPU_OFFLOAD_ARCH` wins (comma or space separated), then whatever the installed
/// devices report, then `FALLBACK_ARCHS`.
fn offload_archs(rocm: Option<&str>) -> Vec<String> {
    if let Ok(configured) = std::env::var("TM_GPU_OFFLOAD_ARCH") {
        let archs: Vec<String> = configured
            .split([',', ' '])
            .filter(|arch| !arch.is_empty())
            .map(str::to_owned)
            .collect();
        if !archs.is_empty() {
            return archs;
        }
    }
    for tool in ["amdgpu-arch", "rocm_agent_enumerator"] {
        if let Some(archs) = detect_archs(rocm, tool) {
            return archs;
        }
    }
    FALLBACK_ARCHS.iter().map(|arch| (*arch).to_owned()).collect()
}

fn detect_archs(rocm: Option<&str>, tool: &str) -> Option<Vec<String>> {
    let program = rocm
        .map(|rocm| Path::new(rocm).join("bin").join(tool))
        .filter(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from(tool));
    let output = Command::new(program).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let mut archs: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        // rocm_agent_enumerator lists the host agent as gfx000.
        .filter(|line| line.starts_with("gfx") && *line != "gfx000")
        .map(str::to_owned)
        .collect();
    archs.sort();
    archs.dedup();
    (!archs.is_empty()).then_some(archs)
}
