//! Builds the Argon2 device kernels and links the vendor runtime.
//!
//! Two mutually exclusive shapes, chosen by feature:
//!
//! - `amd` (default, **tested**): `hipcc` compiles `kernel/argon2_kernel.hip`, and the Rust
//!   kernels in `../tm-kernel` are compiled for `amdgcn-amd-amdhsa`. Links `libamdhip64`.
//! - `nvidia` (**never executed — no NVIDIA GPU exists on the machine this was written on**):
//!   the same Rust kernels are compiled for `nvptx64-nvidia-cuda` and emitted as PTX, and the
//!   crate links the CUDA driver library. No `hipcc`, no C++.
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

    let nvidia = std::env::var_os("CARGO_FEATURE_NVIDIA").is_some();
    let amd = std::env::var_os("CARGO_FEATURE_AMD").is_some();
    assert!(
        !(nvidia && amd),
        "tm-gpu's `amd` and `nvidia` features are mutually exclusive: they select different \
         device runtimes to link. Build with `--no-default-features --features nvidia`."
    );
    if nvidia {
        nvidia::main();
        return;
    }

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
    let archs = offload_archs(rocm.as_deref());
    for arch in &archs {
        command.arg(format!("--offload-arch={arch}"));
    }

    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", hipcc.display()));
    assert!(status.success(), "hipcc failed to compile the Argon2 kernel");

    if std::env::var_os("CARGO_FEATURE_RUST_KERNEL").is_some() {
        // hipModuleLoad wants a code object for exactly one architecture, so the Rust
        // kernel is built for the first arch the HIP kernel is being fattened with.
        let arch = archs.first().expect("offload_archs is never empty");
        gpu_kernel::build_amdgcn(rocm.as_deref(), arch);
    }

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

/// Builds `../tm-kernel` for `amdgcn-amd-amdhsa` and points `TM_RUST_KERNEL_ELF` at the
/// resulting code object, which `src/module.rs` embeds and hands to `hipModuleLoadData`.
///
/// `amdgcn-amd-amdhsa` is a tier-3 target: there is no prebuilt `core`, so the kernel needs
/// `-Z build-std=core`, which needs a nightly-flavoured rustc (`RUSTC_BOOTSTRAP=1`) *and* a
/// sysroot that contains the standard-library source. The nix rustc ships the compiler
/// source but not `rust-src`, so both are stitched together here into a symlink farm; see
/// PORT.md, "Rust GPU kernels", for the whole recipe and its failure modes.
// Build scripts do not see feature cfgs, only `CARGO_FEATURE_*`; `main` calls this only
// when `rust-kernel` is enabled.
#[allow(dead_code)]
mod gpu_kernel {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// `../tm-kernel/Cargo.toml`, with the rebuild triggers registered.
    fn kernel_manifest() -> PathBuf {
        let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo"))
            .join("../tm-kernel/Cargo.toml")
            .canonicalize()
            .expect("crates/tm-kernel exists next to crates/tm-gpu");
        let kernel_dir = manifest.parent().expect("a manifest has a directory");
        println!("cargo:rerun-if-changed={}", manifest.display());
        println!("cargo:rerun-if-changed={}", kernel_dir.join("src").display());
        println!("cargo:rerun-if-env-changed=TM_RUST_LIB_SRC");
        manifest
    }

    /// Strips the outer build's settings out of a cargo invocation for a different target:
    /// host rustflags, clippy's wrapper, the workspace target dir would all leak otherwise.
    fn unleak(command: &mut Command) {
        for leaked in [
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_BUILD_TARGET",
            "CARGO_BUILD_RUSTFLAGS",
            "CARGO_BUILD_TARGET_DIR",
        ] {
            command.env_remove(leaked);
        }
    }

    /// Builds `../tm-kernel` for `amdgcn-amd-amdhsa`. This is the tested path.
    pub fn build_amdgcn(rocm: Option<&str>, arch: &str) {
        let manifest = kernel_manifest();
        println!("cargo:rerun-if-env-changed=TM_GPU_LLD");

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
        let sysroot = build_sysroot(&out_dir, &rustc);
        let wrapper = write_rustc_wrapper(&out_dir, &rustc, &sysroot);
        let lld = find_lld(rocm);

        let target_dir = out_dir.join("kernel-target");
        let mut command = Command::new("cargo");
        command
            .arg("build")
            .arg("--release")
            .arg("--manifest-path")
            .arg(&manifest)
            .arg("--target")
            .arg("amdgcn-amd-amdhsa")
            .arg("-Zbuild-std=core")
            .env("RUSTC_BOOTSTRAP", "1")
            .env("RUSTC", &wrapper)
            .env("CARGO_TARGET_DIR", &target_dir)
            .env(
                "RUSTFLAGS",
                format!(
                    "-Ctarget-cpu={arch} -Clinker={} -Clinker-flavor=ld.lld",
                    lld.display()
                ),
            );
        unleak(&mut command);

        let status = command
            .status()
            .unwrap_or_else(|error| panic!("failed to run cargo for tm-kernel: {error}"));
        assert!(status.success(), "building tm-kernel for amdgcn failed");

        let elf = target_dir
            .join("amdgcn-amd-amdhsa/release/tm_kernel.elf")
            .canonicalize()
            .expect("cargo emitted the amdgcn code object");
        println!("cargo:rustc-env=TM_RUST_KERNEL_ELF={}", elf.display());
        println!("cargo:rustc-env=TM_RUST_KERNEL_ARCH={arch}");
    }

    /// Builds `../tm-kernel` for `nvptx64-nvidia-cuda`, emits PTX, and points
    /// `TM_PTX_KERNEL` at it. `src/cuda/module.rs` embeds that text and hands it to
    /// `cuModuleLoadData`, which JITs it for whatever card is present.
    ///
    /// **This output has never been executed.** It is read, not run: see PORT.md.
    ///
    /// Two things differ from the amdgcn build beyond the target triple.
    ///
    /// `--crate-type=rlib` overrides the manifest's `cdylib`, and `--emit=asm` replaces the
    /// link step. rustc links nvptx64 with `llvm-bitcode-linker`, which the nix rustc does
    /// not ship (and rustup only offers as a nightly component); emitting the assembly of a
    /// single codegen unit sidesteps it entirely. That is only sound because the kernel
    /// crate is self-contained — every helper is `#[inline(always)]` or defined in the
    /// crate, and the `sigma!` / `buf_slot!` masks in `tm-kernel` exist precisely so that
    /// nothing calls out to `core`. If a future edit reintroduces an external call, the PTX
    /// will carry an unresolved `.extern .func` and `cuModuleLoadData` will fail to JIT it;
    /// `ptx_is_self_contained` in `src/cuda/module.rs` is the test that catches it first.
    ///
    /// `-Ctarget-cpu` is left at the target default, `sm_70`. It is also the *lowest* value
    /// this rustc accepts for nvptx64, so anything older than Volta cannot be targeted at
    /// all from here.
    pub fn build_nvptx() {
        let manifest = kernel_manifest();

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
        let sysroot = build_sysroot(&out_dir, &rustc);
        let wrapper = write_rustc_wrapper(&out_dir, &rustc, &sysroot);

        let target_dir = out_dir.join("ptx-target");
        let mut command = Command::new("cargo");
        command
            .arg("rustc")
            .arg("--release")
            .arg("--manifest-path")
            .arg(&manifest)
            .arg("--target")
            .arg("nvptx64-nvidia-cuda")
            .arg("--crate-type=rlib")
            .arg("-Zbuild-std=core")
            .arg("--")
            .arg("--emit=asm")
            .arg("-Ccodegen-units=1")
            .env("RUSTC_BOOTSTRAP", "1")
            .env("RUSTC", &wrapper)
            .env("CARGO_TARGET_DIR", &target_dir);
        unleak(&mut command);

        let status = command
            .status()
            .unwrap_or_else(|error| panic!("failed to run cargo for tm-kernel: {error}"));
        assert!(status.success(), "building tm-kernel for nvptx64 failed");

        let deps = target_dir.join("nvptx64-nvidia-cuda/release/deps");
        let ptx = newest_asm(&deps);
        let destination = out_dir.join("tm_kernel.ptx");
        std::fs::copy(&ptx, &destination)
            .unwrap_or_else(|error| panic!("copying {}: {error}", ptx.display()));
        println!("cargo:rustc-env=TM_PTX_KERNEL={}", destination.display());
    }

    /// The `.s` rustc just wrote. `cargo rustc --emit=asm` names it after the crate *and* a
    /// metadata hash, and leaves older ones behind, so it is picked by modification time.
    fn newest_asm(deps: &Path) -> PathBuf {
        let entries = std::fs::read_dir(deps)
            .unwrap_or_else(|error| panic!("reading {}: {error}", deps.display()));
        let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != "s") {
                continue;
            }
            let Ok(modified) = entry.metadata().and_then(|meta| meta.modified()) else {
                continue;
            };
            if best.as_ref().is_none_or(|(best, _)| modified >= *best) {
                best = Some((modified, path));
            }
        }
        best.map(|(_, path)| path).unwrap_or_else(|| {
            panic!(
                "cargo rustc --emit=asm produced no .s in {}",
                deps.display()
            )
        })
    }

    /// A sysroot that is the real one plus `lib/rustlib/src/rust/library`, assembled from
    /// symlinks so nothing is copied and a rustc upgrade cannot leave a stale mixture.
    fn build_sysroot(out_dir: &Path, rustc: &str) -> PathBuf {
        let real = Command::new(rustc)
            .arg("--print")
            .arg("sysroot")
            .output()
            .unwrap_or_else(|error| panic!("failed to run {rustc}: {error}"));
        assert!(real.status.success(), "rustc --print sysroot failed");
        let real = PathBuf::from(String::from_utf8_lossy(&real.stdout).trim().to_owned());

        let sysroot = out_dir.join("gpu-sysroot");
        let _ = std::fs::remove_dir_all(&sysroot);
        std::fs::create_dir_all(sysroot.join("lib/rustlib/src/rust"))
            .expect("OUT_DIR is writable");
        link_children(&real, &sysroot, &["lib"]);
        link_children(&real.join("lib"), &sysroot.join("lib"), &["rustlib"]);
        link_children(
            &real.join("lib/rustlib"),
            &sysroot.join("lib/rustlib"),
            &["src"],
        );

        let library = rust_lib_src(&real);
        std::os::unix::fs::symlink(&library, sysroot.join("lib/rustlib/src/rust/library"))
            .expect("the shadow sysroot is fresh, so the link cannot already exist");
        sysroot
    }

    fn link_children(from: &Path, to: &Path, skip: &[&str]) {
        let entries = std::fs::read_dir(from)
            .unwrap_or_else(|error| panic!("reading {}: {error}", from.display()));
        for entry in entries.flatten() {
            let name = entry.file_name();
            if skip.iter().any(|skipped| name == *skipped) {
                continue;
            }
            let _ = std::os::unix::fs::symlink(entry.path(), to.join(&name));
        }
    }

    /// The `library/` directory of the rust source: `TM_RUST_LIB_SRC` (set by `./rs`), else
    /// a rustup-style `rust-src` component in the sysroot, else nixpkgs' `rustLibSrc`.
    fn rust_lib_src(sysroot: &Path) -> PathBuf {
        if let Some(configured) = std::env::var_os("TM_RUST_LIB_SRC") {
            let path = PathBuf::from(configured);
            assert!(
                path.join("core/src/lib.rs").is_file(),
                "TM_RUST_LIB_SRC={} does not look like the rust `library` directory",
                path.display()
            );
            return path;
        }
        let component = sysroot.join("lib/rustlib/src/rust/library");
        if component.join("core/src/lib.rs").is_file() {
            return component;
        }
        let output = Command::new("nix")
            .args([
                "build",
                "--no-link",
                "--print-out-paths",
                "nixpkgs#rustPlatform.rustLibSrc",
            ])
            .output()
            .expect(
                "no rust-src component in the sysroot and `nix` is unavailable: set \
                 TM_RUST_LIB_SRC to the rust `library` directory",
            );
        assert!(
            output.status.success(),
            "nix build nixpkgs#rustPlatform.rustLibSrc failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let path = String::from_utf8_lossy(&output.stdout)
            .lines()
            .last()
            .expect("nix printed an output path")
            .trim()
            .to_owned();
        PathBuf::from(path)
    }

    fn write_rustc_wrapper(out_dir: &Path, rustc: &str, sysroot: &Path) -> PathBuf {
        // `--sysroot` cannot go in RUSTFLAGS: cargo passes rustflags only to the final
        // crate, and `core` itself has to be compiled against the same sysroot.
        let script = out_dir.join("rustc-gpu");
        let body = format!(
            "#!/bin/sh\nexec {rustc} --sysroot {} \"$@\"\n",
            sysroot.display()
        );
        std::fs::write(&script, body).expect("OUT_DIR is writable");
        let mut permissions = std::fs::metadata(&script)
            .expect("the script was just written")
            .permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
        std::fs::set_permissions(&script, permissions).expect("OUT_DIR is writable");
        script
    }

    /// rustc's `ld.lld` flavour passes `-flavor gnu`, which only the real lld driver
    /// accepts: nixpkgs wraps `ld.lld` in a shell script that rejects it, and nixpkgs' own
    /// lld 18 rejects the AMDGPU ABI version outright. So candidates are probed rather than
    /// guessed, and the probe *is* the requirement.
    fn find_lld(rocm: Option<&str>) -> PathBuf {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Some(configured) = std::env::var_os("TM_GPU_LLD") {
            roots.push(PathBuf::from(configured));
        }
        if let Some(rocm) = rocm {
            let rocm = Path::new(rocm);
            roots.push(rocm.join("bin"));
            let clang = rocm.join("bin/amdclang");
            if let Ok(output) = Command::new(&clang).arg("--print-prog-name=ld.lld").output() {
                let printed = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                if !printed.is_empty() {
                    roots.push(PathBuf::from(&printed));
                    // That path is usually a nixpkgs wrapper script, which names the real
                    // driver it delegates to.
                    roots.extend(paths_named_in(Path::new(&printed)));
                }
            }
            // hipcc is a script too, and it names the clang installation it drives.
            roots.extend(paths_named_in(&rocm.join("bin/hipcc")));
        }
        roots.push(PathBuf::from("ld.lld"));

        let mut tried = Vec::new();
        for root in &roots {
            for candidate in [
                root.to_owned(),
                root.join("ld.lld"),
                root.join("bin/ld.lld"),
            ] {
                if !candidate.file_name().is_some_and(|name| name == "ld.lld") {
                    continue;
                }
                if tried.contains(&candidate) {
                    continue;
                }
                let probe = Command::new(&candidate)
                    .args(["-flavor", "gnu", "--version"])
                    .output();
                if matches!(probe, Ok(output) if output.status.success()) {
                    return resolve(&candidate);
                }
                tried.push(candidate);
            }
        }
        panic!(
            "no ld.lld accepting `-flavor gnu` found (tried {tried:?}); set TM_GPU_LLD to a \
             ROCm lld binary"
        );
    }

    /// Absolute paths mentioned inside a (possibly binary) file. Used to follow the nix
    /// wrapper scripts back to the tools they wrap.
    fn paths_named_in(file: &Path) -> Vec<PathBuf> {
        let Ok(bytes) = std::fs::read(file) else {
            return Vec::new();
        };
        let text = String::from_utf8_lossy(&bytes);
        let mut found = Vec::new();
        for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || "/._-+".contains(c))) {
            if token.starts_with('/') && token.len() > 1 {
                let path = PathBuf::from(token);
                if !found.contains(&path) {
                    found.push(path);
                }
            }
        }
        found
    }

    fn resolve(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned())
    }
}

/// The `nvidia` feature's half of the build.
///
/// **UNTESTED PATH.** Nothing here has ever run against a real CUDA installation: the
/// machine this was written on has no NVIDIA GPU and no CUDA driver. It builds the PTX and
/// arranges the link; whether the link is *correct* is unproven.
#[allow(dead_code)]
mod nvidia {
    use std::path::{Path, PathBuf};

    /// Where a CUDA driver library normally lives. The toolkit's `stubs/libcuda.so` is an
    /// acceptable answer for linking — it exports the same symbols and the real driver is
    /// resolved at load time — which is how CUDA programs are built on machines without a
    /// GPU. On NixOS the driver is in `/run/opengl-driver/lib`.
    const LIBCUDA_DIRS: &[&str] = &[
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib64/stubs",
        "/opt/cuda/lib64",
        "/opt/cuda/lib64/stubs",
        "/run/opengl-driver/lib",
    ];

    pub fn main() {
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=TM_CUDA_LIB_DIR");
        println!("cargo:rerun-if-env-changed=TM_CUDA_ALLOW_MISSING");
        println!("cargo:rerun-if-changed=build.rs");

        super::gpu_kernel::build_nvptx();

        match find_libcuda() {
            Some(dir) => {
                println!("cargo:rustc-link-search=native={}", dir.display());
                // Test binaries are run directly rather than through a toolchain wrapper,
                // so the search path has to survive into the runtime loader too — same
                // reason the AMD path adds an rpath for ROCm.
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
            }
            None => missing_libcuda(),
        }
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    /// `TM_CUDA_LIB_DIR`, then `CUDA_PATH`'s usual subdirectories, then [`LIBCUDA_DIRS`].
    fn find_libcuda() -> Option<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Some(configured) = std::env::var_os("TM_CUDA_LIB_DIR") {
            roots.push(PathBuf::from(configured));
        }
        if let Some(cuda) = std::env::var_os("CUDA_PATH") {
            let cuda = PathBuf::from(cuda);
            roots.push(cuda.join("lib64"));
            roots.push(cuda.join("lib64/stubs"));
            roots.push(cuda.join("lib/x64"));
        }
        roots.extend(LIBCUDA_DIRS.iter().map(PathBuf::from));
        roots
            .into_iter()
            .find(|dir| has_libcuda(dir))
    }

    fn has_libcuda(dir: &Path) -> bool {
        ["libcuda.so", "libcuda.so.1", "cuda.lib"]
            .iter()
            .any(|name| dir.join(name).exists())
    }

    /// Refuse loudly. A missing driver library would otherwise show up much later as a bare
    /// `cannot find -lcuda` from the linker, or — worse — as a `tm-gpu` that builds as a
    /// library and only fails when someone tries to link the miner.
    fn missing_libcuda() {
        if std::env::var_os("TM_CUDA_ALLOW_MISSING").is_some() {
            println!(
                "cargo:warning=TM_CUDA_ALLOW_MISSING is set and no libcuda was found: tm-gpu \
                 is compile-checking the NVIDIA path only. The library will build; linking \
                 any binary or test against it will fail."
            );
            return;
        }
        panic!(
            "tm-gpu's `nvidia` feature needs the CUDA driver library (libcuda.so) and none \
             was found. Looked at TM_CUDA_LIB_DIR, $CUDA_PATH/lib64[/stubs], and {LIBCUDA_DIRS:?}. \
             Install the NVIDIA driver or the CUDA toolkit (its lib64/stubs/libcuda.so is \
             enough to link), or point TM_CUDA_LIB_DIR at the directory holding it. To \
             compile-check the NVIDIA path on a machine with no CUDA at all — which is the \
             only way it has ever been built — set TM_CUDA_ALLOW_MISSING=1; the resulting \
             rlib cannot be linked into anything."
        );
    }
}
