{
  description = "TreeMiner — outage-proof XenBlocks miner (NVIDIA CUDA / AMD ROCm)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;
        rocm = pkgs.rocmPackages;
        rustPlatform = pkgs.rustPlatform;

        # Everything the miner links against, independent of GPU vendor. The upstream
        # build uses vcpkg for these; CMake also accepts the system copies (SQLite via
        # FindSQLite3, argon2 via pkg-config).
        commonDeps = with pkgs; [
          libargon2
          cryptopp
          libcpr
          nlohmann_json
          openssl
          boost
          secp256k1
          crow
          paho-mqtt-c
          paho-mqtt-cpp
          sqlite
          asio
        ];

        rocmDeps = [
          rocm.clr           # HIP runtime + hipcc/amdclang++ + rocminfo
          rocm.rocm-smi      # optional: power/utilization gauges
          rocm.rocm-device-libs
        ];

        buildTools = with pkgs; [ cmake ninja pkg-config gcc python3 ];

        # The Rust tree (treeminer-rs/). Kernels are built for the tier-3
        # amdgcn-amd-amdhsa target, which needs -Z build-std and therefore the standard
        # library *source* — rustLibSrc — in addition to the compiler.
        rustTools = with pkgs; [ cargo rustc clippy rustfmt ];

        # ---------------------------------------------------------------------------
        # The Rust miner, built entirely by nix.
        # ---------------------------------------------------------------------------

        # treeminer-rs/ minus the build artefacts. `target/` is ~100 MB of cargo output
        # and would otherwise be copied into the store on every `nix build`.
        rustSrc = lib.cleanSourceWith {
          name = "treeminer-rs-source";
          src = ./treeminer-rs;
          filter = path: type:
            let base = baseNameOf (toString path); in
            !(type == "directory" && (base == "target" || base == ".git"));
        };

        # `-Z build-std=core` compiles `core` and `compiler_builtins` out of the rust
        # `library/` tree, which pulls a handful of crates.io dependencies of its own
        # (compiler_builtins' `rustc-std-workspace-core`, `cc`, ...). Those are not in the
        # workspace lockfile, so a plain `cargoLock` vendor directory leaves the inner
        # cargo trying to hit crates.io — which the sandbox forbids.
        #
        # rustLibSrc ships its own `vendor/` holding exactly those crates. Merging the two
        # vendor trees into a single directory makes one `replace-with = "vendored-sources"`
        # cover both the workspace and the standard library, so the kernel build resolves
        # offline. The derivation must be *named* cargo-vendor-dir: nixpkgs' cargoSetupHook
        # copies it to `$(stripHash $cargoDeps)` and the generated .cargo/config.toml points
        # at that name.
        workspaceVendor = rustPlatform.importCargoLock {
          lockFile = ./treeminer-rs/Cargo.lock;
        };

        vendorDir = pkgs.runCommand "cargo-vendor-dir" { } ''
          cp -r --no-preserve=mode,ownership ${workspaceVendor} $out
          for crate in ${rustPlatform.rustLibSrc}/vendor/*; do
            name=$(basename "$crate")
            if [ ! -e "$out/$name" ]; then
              cp -r --no-preserve=mode,ownership "$crate" "$out/$name"
            fi
          done
        '';

        # `gpuArch` is the AMD target the kernels are compiled for. It is a single arch on
        # purpose: `tm-gpu/build.rs` loads the Rust code object with hipModuleLoad, which
        # accepts exactly one architecture, and it takes the *first* entry of the offload
        # list. gfx1100 (RDNA3, RX 7900) is the card this miner is tested on. Override with
        #   nix build --impure --expr '(builtins.getFlake (toString ./.)).packages.x86_64-linux.treeminer.override { gpuArch = "gfx1030"; }'
        # or, in a flake that consumes this one, `.override { gpuArch = "gfx942"; }`.
        mkTreeminer = lib.makeOverridable
          ({ vendor ? "amd"
           , gpuArch ? "gfx1100"
           , cudaLibDir ? null
           }:
            let
              nvidia = vendor == "nvidia";
              # Only touched on the NVIDIA path, and only when the user has opted into
              # unfree packages — nixpkgs' CUDA is unfree and multi-gigabyte, so merely
              # *evaluating* this attribute must not reach for it.
              resolvedCudaLibDir =
                if cudaLibDir != null then cudaLibDir
                else if nvidia && (pkgs.config.allowUnfree or false)
                then "${lib.getLib pkgs.cudaPackages.cuda_cudart}/lib/stubs"
                else null;
            in
            rustPlatform.buildRustPackage {
              pname = if nvidia then "treeminer-nvidia" else "treeminer";
              version = "1.0.0";
              src = rustSrc;

              cargoDeps = vendorDir;

              buildAndTestSubdir = null;
              cargoBuildFlags = [ "-p" "treeminer" ];
              buildNoDefaultFeatures = nvidia;
              buildFeatures = lib.optionals nvidia [ "nvidia" ];

              nativeBuildInputs = with pkgs; [ pkg-config python3 ]
                ++ lib.optionals (!nvidia) [ rocm.clr ];
              buildInputs = lib.optionals (!nvidia) [ rocm.clr rocm.rocm-device-libs ];

              # The workspace lockfile lives at the source root; buildRustPackage's
              # consistency check compares it with the vendored copy.
              cargoRoot = null;

              # Every GPU test needs a real /dev/kfd, which the sandbox does not have, and
              # the CPU-only tests are covered by `./rs cargo test --workspace` outside it.
              doCheck = false;

              env = {
                # amdgcn kernels: tier-3 target, so `-Z build-std=core` needs the standard
                # library *source*. See treeminer-rs/PORT.md, "Rust GPU kernels".
                TM_RUST_LIB_SRC = "${rustPlatform.rustLibSrc}";
              } // lib.optionalAttrs (!nvidia) {
                ROCM_PATH = "${rocm.clr}";
                HIP_PATH = "${rocm.clr}";
                HIP_DEVICE_LIB_PATH = "${rocm.rocm-device-libs}/amdgcn/bitcode";
                TM_ROCM_DEVICE_LIBS = "${rocm.rocm-device-libs}";
                TM_ROCM_SMI_PATH = "${rocm.rocm-smi}";
                # build.rs would otherwise ask the (absent) local GPU and fall back to a
                # four-arch fat binary, of which the Rust kernel would only get the first.
                TM_GPU_OFFLOAD_ARCH = gpuArch;
              } // lib.optionalAttrs (nvidia && resolvedCudaLibDir != null) {
                TM_CUDA_LIB_DIR = resolvedCudaLibDir;
              };

              # The NVIDIA path cannot be built without a libcuda to link against. Say so
              # up front instead of letting it fail three minutes later at the link step.
              preBuild = lib.optionalString (nvidia && resolvedCudaLibDir == null) ''
                cat >&2 <<'EOF'
                treeminer-nvidia needs a CUDA driver library (libcuda.so) to link against,
                and none was configured. nixpkgs' CUDA is unfree, so this package does not
                pull it in unless you ask:

                  NIXPKGS_ALLOW_UNFREE=1 nix build --impure .#treeminer-nvidia

                or point it at a libcuda you already have:

                  .override { cudaLibDir = "/run/opengl-driver/lib"; }

                This package has never been executed on NVIDIA hardware — see
                treeminer-rs/README.md before trusting anything it produces.
                EOF
                exit 1
              '';

              meta = {
                description =
                  if nvidia then
                    "TreeMiner (Rust) for NVIDIA/CUDA — COMPILE-VERIFIED ONLY, never executed on NVIDIA hardware; needs an unfree CUDA to build"
                  else
                    "TreeMiner (Rust): outage-proof XenBlocks Argon2 miner for AMD/ROCm (${gpuArch})";
                mainProgram = "treeminer";
                license = lib.licenses.mit;
                platforms = lib.platforms.linux;
              };
            });
      in
      {
        devShells.default = pkgs.mkShell {
          name = "treeminer-rocm";
          packages = buildTools ++ rustTools ++ commonDeps ++ rocmDeps;

          # hipcc needs to find its device bitcode; on NixOS these do not live under
          # /opt/rocm, so point the toolchain at the store paths explicitly.
          ROCM_PATH = "${rocm.clr}";
          HIP_PATH = "${rocm.clr}";
          HIP_DEVICE_LIB_PATH = "${rocm.rocm-device-libs}/amdgcn/bitcode";
          TM_ROCM_DEVICE_LIBS = "${rocm.rocm-device-libs}";
          TM_ROCM_SMI_PATH = "${rocm.rocm-smi}";
          # Read by treeminer-rs/rs and tm-gpu's build.rs to assemble the GPU sysroot.
          TM_RUST_LIB_SRC = "${pkgs.rustPlatform.rustLibSrc}";

          shellHook = ''
            echo "TreeMiner dev shell — hipcc $(hipcc --version 2>/dev/null | head -1), $(rustc --version)"
            echo
            echo "Rust miner (treeminer-rs/):"
            echo "  cd treeminer-rs && cargo test --workspace && cargo build --release -p treeminer"
            echo "  (or from the repo root: nix build .#treeminer)"
            echo
            echo "C++ miner (treeminer/):"
            echo "  cmake -S treeminer -B build -G Ninja -DTREEMINER_GPU_BACKEND=HIP \\"
            echo "    -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ && cmake --build build -j"
          '';
        };

        # `nix build` / `nix build .#treeminer` — the Rust miner for AMD GPUs, kernels
        # included. This is the miner; treeminer-rocm below is the C++ one it replaces.
        packages.treeminer = mkTreeminer { };

        # `nix build .#treeminer-nvidia` — EVALUATION ONLY. It evaluates without touching
        # nixpkgs' unfree CUDA; building it needs NIXPKGS_ALLOW_UNFREE=1 (or a cudaLibDir
        # override) and has never been done. Nothing here has run on an NVIDIA card.
        packages.treeminer-nvidia = mkTreeminer { vendor = "nvidia"; };

        # `nix build .#treeminer-rocm` — the C++ miner built for AMD GPUs. Pass a gfx target
        # with --override-input or edit hipArch below; gfx1100 covers RDNA3 (RX 7900).
        packages.treeminer-rocm = pkgs.stdenv.mkDerivation {
          pname = "treeminer-rocm";
          version = "1.0";
          src = ./treeminer;
          nativeBuildInputs = buildTools;
          buildInputs = commonDeps ++ rocmDeps;
          cmakeFlags = [
            "-DTREEMINER_GPU_BACKEND=HIP"
            "-DCMAKE_HIP_COMPILER=${rocm.clr}/bin/amdclang++"
            "-DCMAKE_HIP_ARCHITECTURES=gfx1100"
            "-DTREEMINER_BUILD_TESTS=OFF"
          ];
          HIP_DEVICE_LIB_PATH = "${rocm.rocm-device-libs}/amdgcn/bitcode";
          installPhase = ''
            mkdir -p $out/bin
            cp bin/xenblocksMiner $out/bin/
          '';
          # `nix run` needs the binary name, which differs from pname.
          meta.mainProgram = "xenblocksMiner";
        };

        packages.default = self.packages.${system}.treeminer;
      });
}
