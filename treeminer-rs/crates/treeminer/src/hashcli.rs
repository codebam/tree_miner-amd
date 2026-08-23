//! The network-free hash CLI. Port of `src/hashapi/HashApiCli.cpp` and
//! `src/hashapi/HashApiJson.cpp`.
//!
//! `hash-one`, `hash-batch` and `hash-benchmark` hash with the real backends and print the
//! result, without touching the network or the journal. That makes them the differential
//! test harness: `tests/parity/run_parity.sh` runs the same request through this binary and
//! through the C++ one and compares the digests, so the flags and the `--json` field names
//! have to match the C++ exactly rather than being modernised.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Write;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use tm_argon2::{
    first_block_selected_chunk_size, first_block_worker_count,
    recommended_first_block_dynamic_chunk_size, CpuArgon2Host, HashBackend, HashMatch,
    HashRequest, HashResult, HashTimings, RandomHexKeyGenerator, HASH_API_KEY_LENGTH,
};
use tm_core::batch::{
    estimate_memory_batch_limit, recommended_batch_size, GpuRuntimeKind,
    DEFAULT_MEMORY_RESERVE_BYTES,
};

/// Which runtime `tm_core` should size batches for; follows the vendor `tm-gpu` was built
/// with. ROCm needs a much larger VRAM cushion than CUDA — see `tm_core::batch`.
#[cfg(feature = "amd")]
const RUNTIME: GpuRuntimeKind = GpuRuntimeKind::Hip;
#[cfg(feature = "nvidia")]
const RUNTIME: GpuRuntimeKind = GpuRuntimeKind::Cuda;

const USAGE: &str = concat!(
    "Hash API commands:\n",
    "  xenblocksMiner hash-one --salt <hex> --key <64-hex> [--backend cpu|gpu] [--difficulty <n>] [--no-xuni] [--detailed-timings] [--first-block-workers <n>] [--first-block-dynamic-chunk-size <n>] [--first-block-dynamic-chunk-auto] [--gpu-first-blocks] [--json]\n",
    "  xenblocksMiner hash-batch --salt <hex> [--backend cpu|gpu] [--prefix <hex>] [--pattern XEN11] [--batch-size <n>] [--auto-batch-size] [--difficulty <n>] [--no-xuni] [--detailed-timings] [--first-block-workers <n>] [--first-block-dynamic-chunk-size <n>] [--first-block-dynamic-chunk-auto] [--gpu-first-blocks] [--json]\n",
    "  xenblocksMiner hash-benchmark --salt <hex> [--backend cpu|gpu] [--key <64-hex>] [--prefix <hex>] [--seconds <n>] [--batch-size <n>] [--auto-batch-size] [--batch-size-sequence <n,n,...>] [--difficulty <n>] [--difficulty-sequence <n,n,...>] [--no-xuni] [--detailed-timings] [--first-block-workers <n>] [--first-block-dynamic-chunk-size <n>] [--first-block-dynamic-chunk-auto] [--gpu-first-blocks] [--json]\n",
);

/// Flags that take no value; everything else consumes the next argument.
const BOOL_FLAGS: &[&str] = &[
    "--json",
    "--no-xuni",
    "--detailed-timings",
    "--auto-batch-size",
    "--first-block-dynamic-chunk-auto",
    "--gpu-first-blocks",
];

/// The backend names the CLI accepts, folded onto the names `tm-argon2` validates.
///
/// `gpu` is the honest spelling for the device backend on this miner: the kernels are HIP,
/// not CUDA. `cuda` is what the C++ miner calls the same backend, so it stays accepted --
/// `tests/parity/run_parity.sh` drives both binaries with one command line -- but it is not
/// advertised. `reference` is the portable CPU implementation the fixture tests compare
/// against.
fn canonical_backend(requested: &str) -> Option<&'static str> {
    match requested {
        "cpu" => Some("cpu"),
        "reference" => Some("reference"),
        "gpu" | "cuda" => Some("cuda"),
        _ => None,
    }
}

/// True for either spelling of the device backend.
fn is_gpu_backend(backend: &str) -> bool {
    matches!(backend, "gpu" | "cuda")
}

/// `request.backend` carries the spelling the operator typed, because the JSON echoes it
/// back and the parity harness diffs that field. `tm-argon2` only knows the C++ names, so
/// the request is folded onto those before it is validated.
fn with_canonical_backend(request: &HashRequest) -> Cow<'_, HashRequest> {
    match canonical_backend(&request.backend) {
        Some(canonical) if canonical != request.backend => {
            let mut folded = request.clone();
            canonical.clone_into(&mut folded.backend);
            Cow::Owned(folded)
        }
        _ => Cow::Borrowed(request),
    }
}

/// True when `args` (the whole argv) names a hash-API subcommand.
pub fn is_hash_api_command(args: &[String]) -> bool {
    matches!(
        args.get(1).map(String::as_str),
        Some("hash-one" | "hash-batch" | "hash-benchmark" | "hash-help")
    )
}

/// Runs the hash CLI over the process argv and reports the C++ exit codes: 0 on success,
/// 2 on a failed request or a bad argument, 1 on an unrecognised subcommand.
pub fn run(args: &[String]) -> ExitCode {
    let mut out = std::io::stdout().lock();
    let mut err = std::io::stderr().lock();
    ExitCode::from(run_with(args, &mut out, &mut err))
}

/// The testable entry point: same behaviour as [`run`], with the two streams injected.
pub fn run_with(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> u8 {
    let command = args.get(1).map(String::as_str);
    if command.is_none() || command == Some("hash-help") {
        let _ = out.write_all(USAGE.as_bytes());
        return 0;
    }
    let parsed = parse_args(args);
    let json = parsed.flag("--json");

    match dispatch(command.unwrap_or_default(), &parsed, json, out) {
        Ok(code) => code,
        Err(Failure::Usage) => {
            let _ = out.write_all(USAGE.as_bytes());
            1
        }
        // The C++ catches every exception around the whole command and reports it in the
        // same envelope as a failed request, so a bad `--difficulty-sequence` is a JSON
        // error object rather than a stack trace.
        Err(Failure::Message(message)) => {
            if json {
                let result = HashResult {
                    error: message,
                    ..Default::default()
                };
                let _ = writeln!(out, "{}", to_json(&result));
            } else {
                let _ = writeln!(err, "Hash API error: {message}");
            }
            2
        }
    }
}

enum Failure {
    Usage,
    Message(String),
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Failure::Message(message)
    }
}

fn dispatch(
    command: &str,
    args: &Args,
    json: bool,
    out: &mut dyn Write,
) -> Result<u8, Failure> {
    let mut request = base_request(args)?;
    match command {
        "hash-one" => {
            request.key = args.get("--key", "");
            request.batch_size = 1;
            Ok(print_result(&run_backend(&request), json, out))
        }
        "hash-batch" => {
            request.batch_size = args.size("--batch-size", 1)?;
            if args.flag("--auto-batch-size") {
                request.batch_size = automatic_gpu_batch_size(
                    &request,
                    &[],
                    if args.has("--batch-size") {
                        request.batch_size
                    } else {
                        0
                    },
                )?;
            }
            Ok(print_result(&run_backend(&request), json, out))
        }
        "hash-benchmark" => {
            request.batch_size = args.size("--batch-size", 1)?;
            let seconds = args.uint("--seconds", 30)?;
            let difficulty_sequence = parse_difficulty_sequence(&args.get("--difficulty-sequence", ""))?;
            let batch_size_sequence = parse_batch_size_sequence(&args.get("--batch-size-sequence", ""))?;
            if args.flag("--auto-batch-size") && batch_size_sequence.is_empty() {
                request.batch_size = automatic_gpu_batch_size(
                    &request,
                    &difficulty_sequence,
                    if args.has("--batch-size") {
                        request.batch_size
                    } else {
                        0
                    },
                )?;
            }
            Ok(run_benchmark(
                request,
                seconds,
                json,
                &difficulty_sequence,
                &batch_size_sequence,
                out,
            ))
        }
        _ => Err(Failure::Usage),
    }
}

// ---------------------------------------------------------------------------- arguments

/// The C++ argument map: `--key value` pairs plus valueless flags, starting at argv[2].
/// Anything that does not begin with `--` is ignored, and a repeated flag keeps the last
/// value — both are load-bearing, because the parity harness appends flags blindly.
#[derive(Debug, Default, Clone)]
pub struct Args {
    values: HashMap<String, String>,
}

impl Args {
    pub fn has(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    pub fn flag(&self, key: &str) -> bool {
        self.values.get(key).map(String::as_str) == Some("true")
    }

    pub fn get(&self, key: &str, fallback: &str) -> String {
        self.values
            .get(key)
            .cloned()
            .unwrap_or_else(|| fallback.to_owned())
    }

    pub fn uint(&self, key: &str, fallback: u32) -> Result<u32, String> {
        match self.values.get(key) {
            None => Ok(fallback),
            Some(text) => text
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("{key} must be an unsigned integer")),
        }
    }

    pub fn size(&self, key: &str, fallback: usize) -> Result<usize, String> {
        match self.values.get(key) {
            None => Ok(fallback),
            Some(text) => text
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("{key} must be an unsigned integer")),
        }
    }
}

pub fn parse_args(argv: &[String]) -> Args {
    let mut values = HashMap::new();
    let mut index = 2;
    while index < argv.len() {
        let key = &argv[index];
        if !key.starts_with("--") {
            index += 1;
            continue;
        }
        if BOOL_FLAGS.contains(&key.as_str()) {
            values.insert(key.clone(), "true".to_owned());
            index += 1;
            continue;
        }
        if index + 1 < argv.len() {
            values.insert(key.clone(), argv[index + 1].clone());
            index += 2;
        } else {
            index += 1;
        }
    }
    Args { values }
}

/// Port of `baseRequest`.
pub fn base_request(args: &Args) -> Result<HashRequest, String> {
    let defaults = HashRequest::default();
    let backend = args.get("--backend", "cpu");
    if canonical_backend(&backend).is_none() {
        return Err(format!(
            "unsupported backend: {backend} (valid backends: cpu, gpu)"
        ));
    }
    Ok(HashRequest {
        request_id: args.get("--request-id", ""),
        backend,
        salt_hex: args.get("--salt", ""),
        key: args.get("--key", ""),
        key_prefix: args.get("--prefix", ""),
        target_pattern: args.get("--pattern", "XEN11"),
        difficulty: args.uint("--difficulty", defaults.difficulty)?,
        batch_size: args.size("--batch-size", defaults.batch_size)?,
        device_id: args.uint("--device", 0)? as i32,
        allow_xuni: !args.flag("--no-xuni"),
        detailed_timings: args.flag("--detailed-timings"),
        first_block_workers: args.size("--first-block-workers", 0)?,
        first_block_dynamic_chunk_size: args.size("--first-block-dynamic-chunk-size", 0)?,
        first_block_dynamic_chunk_auto: args.flag("--first-block-dynamic-chunk-auto"),
        gpu_first_blocks: args.flag("--gpu-first-blocks"),
        ..defaults
    })
}

fn parse_sequence(text: &str, label: &str) -> Result<Vec<u64>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut values = Vec::new();
    for token in text.split(',') {
        if token.is_empty() {
            return Err(format!("{label} sequence cannot contain empty values"));
        }
        if !token.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!("{label} sequence values must be unsigned integers"));
        }
        let value: u64 = token
            .parse()
            .map_err(|_| format!("{label} sequence values must be unsigned integers"))?;
        if value == 0 {
            return Err(format!(
                "{label} sequence values must be between 1 and {}",
                if label == "difficulty" {
                    "UINT32_MAX"
                } else {
                    "SIZE_MAX"
                }
            ));
        }
        values.push(value);
    }
    Ok(values)
}

pub fn parse_difficulty_sequence(text: &str) -> Result<Vec<u32>, String> {
    parse_sequence(text, "difficulty")?
        .into_iter()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                "difficulty sequence values must be between 1 and UINT32_MAX".to_owned()
            })
        })
        .collect()
}

pub fn parse_batch_size_sequence(text: &str) -> Result<Vec<usize>, String> {
    parse_sequence(text, "batch-size")?
        .into_iter()
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| "batch-size sequence values must be between 1 and SIZE_MAX".to_owned())
        })
        .collect()
}

// ----------------------------------------------------------------------------- backends

fn error_result(request: &HashRequest, backend: &str, message: String) -> HashResult {
    HashResult {
        request_id: request.request_id.clone(),
        algorithm: request.algorithm.clone(),
        backend: backend.to_owned(),
        device_id: request.device_id,
        batch_size: request.batch_size,
        error: message,
        ..Default::default()
    }
}

/// Port of `runBackend`: one request, one fresh backend.
pub fn run_backend(request: &HashRequest) -> HashResult {
    if is_gpu_backend(&request.backend) {
        if let Err(errors) = tm_argon2::validate_request(&with_canonical_backend(request)) {
            return error_result(request, &request.backend, errors.to_string());
        }
        return match GpuHashBackend::open(request.device_id) {
            Ok(mut backend) => backend.run_batch(request),
            Err(message) => error_result(request, &request.backend, message),
        };
    }
    tm_argon2::CpuHashBackend.run_batch(request)
}

/// Port of `makeReusableBackend`: the benchmark keeps one backend across iterations so the
/// device pool is allocated once.
fn make_reusable_backend(request: &HashRequest) -> Result<Box<dyn HashBackend>, String> {
    if is_gpu_backend(&request.backend) {
        return Ok(Box::new(GpuHashBackend::open(request.device_id)?));
    }
    Ok(Box::new(tm_argon2::CpuHashBackend))
}

/// The `IHashBackend` implementation over `tm-gpu`. Port of `CudaHashBackend.cpp`, which
/// is CUDA in name only here -- the kernels it reaches are HIP.
///
/// The first blocks are filled by `tm_argon2::CpuArgon2Host` unless `--gpu-first-blocks`
/// is set, which is why the CLI is the natural place to check the two against each other.
pub struct GpuHashBackend {
    backend: tm_gpu::GpuHashBackend,
    device_id: i32,
}

impl GpuHashBackend {
    pub fn open(device_id: i32) -> Result<Self, String> {
        let device = tm_gpu::Device::open(device_id).map_err(|error| error.to_string())?;
        Ok(Self {
            backend: tm_gpu::GpuHashBackend::new(tm_gpu::GpuBackend::new(device)),
            device_id,
        })
    }
}

impl HashBackend for GpuHashBackend {
    fn run_batch(&mut self, request: &HashRequest) -> HashResult {
        let total_start = Instant::now();
        let mut result = HashResult {
            request_id: request.request_id.clone(),
            algorithm: request.algorithm.clone(),
            // The requested spelling, echoed back: the parity harness diffs this field
            // against the C++ miner, which only ever sees `cuda`.
            backend: request.backend.clone(),
            device_id: self.device_id,
            batch_size: request.batch_size,
            gpu_first_blocks: request.gpu_first_blocks,
            ..Default::default()
        };

        let validation_start = Instant::now();
        let validation = tm_argon2::validate_request(&with_canonical_backend(request));
        result.timings.validation_ms = millis_since(validation_start);
        if let Err(errors) = validation {
            result.error = errors.to_string();
            result.timings.total_ms = millis_since(total_start);
            return result;
        }
        if !is_gpu_backend(&request.backend) {
            result.error = "GpuHashBackend requires --backend gpu".to_owned();
            result.timings.total_ms = millis_since(total_start);
            return result;
        }

        let start = Instant::now();
        let setup_start = Instant::now();
        let salt = tm_argon2::normalize_hex(&request.salt_hex);
        let prefix = tm_argon2::normalize_hex(&request.key_prefix);
        let fixed_key = tm_argon2::normalize_hex(&request.key);
        let single_key = !fixed_key.is_empty();
        let attempts = if single_key { 1 } else { request.batch_size };

        result.first_block_worker_count =
            first_block_worker_count(attempts, request.first_block_workers);
        result.first_block_dynamic_chunk_auto =
            request.first_block_dynamic_chunk_auto && request.first_block_dynamic_chunk_size == 0;
        let requested_dynamic_chunk_size = if request.first_block_dynamic_chunk_size > 0 {
            request.first_block_dynamic_chunk_size
        } else {
            recommended_first_block_dynamic_chunk_size(
                request.first_block_dynamic_chunk_auto,
                &request.backend,
                single_key,
                request.difficulty,
                attempts,
                result.first_block_worker_count,
            )
        };
        if result.first_block_worker_count > 1 && requested_dynamic_chunk_size > 0 {
            result.first_block_dynamic_chunk_size = attempts.min(requested_dynamic_chunk_size);
        }
        result.first_block_chunk_size = first_block_selected_chunk_size(
            attempts,
            result.first_block_worker_count,
            result.first_block_dynamic_chunk_size,
        );
        result.first_block_dynamic_chunk_size_min = result.first_block_dynamic_chunk_size;
        result.first_block_dynamic_chunk_size_max = result.first_block_dynamic_chunk_size;
        result.first_block_chunk_size_min = result.first_block_chunk_size;
        result.first_block_chunk_size_max = result.first_block_chunk_size;
        result.timings.setup_ms = millis_since(setup_start);

        let input_start = Instant::now();
        let keygen_start = Instant::now();
        let passwords: Vec<String> = if single_key {
            vec![fixed_key]
        } else {
            let mut generator = RandomHexKeyGenerator::new(&prefix, HASH_API_KEY_LENGTH);
            (0..attempts).map(|_| generator.next_random_key()).collect()
        };
        result.timings.keygen_ms = millis_since(keygen_start);

        let host = CpuArgon2Host::new()
            .with_workers(request.first_block_workers)
            .with_dynamic_chunk_size(result.first_block_dynamic_chunk_size);
        let batch = tm_gpu::BatchRequest {
            passwords: &passwords,
            salt_hex: &salt,
            difficulty: request.difficulty,
            target_pattern: &request.target_pattern,
            allow_xuni: request.allow_xuni,
            gpu_first_blocks: request.gpu_first_blocks,
            collect_digests: false,
        };
        let outcome = match self.backend.run_batch(&batch, &host) {
            Ok(outcome) => outcome,
            Err(error) => {
                result.error = error.to_string();
                result.timings.total_ms = millis_since(total_start);
                return result;
            }
        };
        result.timings.input_ms = millis_since(input_start);

        result.gpu_first_blocks = outcome.gpu_first_blocks;
        result.timings.first_block_ms = outcome.timings.first_block_ms;
        result.timings.compute_ms = outcome.timings.compute_ms;
        result.timings.kernel_ms = outcome.timings.kernel_ms;
        result.timings.host_to_device_ms = outcome.timings.host_to_device_ms;
        result.timings.gpu_first_block_ms = outcome.timings.gpu_first_block_ms;
        result.timings.device_to_host_ms = outcome.timings.device_to_host_ms;
        result.timings.finalize_ms = outcome.timings.finalize_ms;
        result.timings.setup_ms += outcome.timings.setup_ms;

        if single_key {
            result.hash = outcome.hash.clone().unwrap_or_default();
        }
        result.matches = outcome
            .matches
            .iter()
            .map(|item| HashMatch {
                key: item.key.clone(),
                hash: item.hash.clone(),
                matched_pattern: item.matched_pattern.clone(),
                attempt_index: item.attempt_index,
                is_superblock: item.is_superblock,
            })
            .collect();

        result.ok = true;
        result.attempts = outcome.attempts;
        result.batch_size = outcome.attempts;
        result.batch_size_min = outcome.attempts;
        result.batch_size_max = outcome.attempts;
        result.elapsed_ms = millis_since(start);
        result.timings.total_ms = millis_since(total_start);
        if result.elapsed_ms > 0.0 && result.attempts > 0 {
            result.hashrate = result.attempts as f64 / (result.elapsed_ms / 1000.0);
        }
        result
    }
}

// ------------------------------------------------------------------------- batch sizing

/// Port of `selectAutomaticCudaBatchSize`, including the difficulty-sequence variant of
/// `HashApiTuning.cpp`: the sequence is sized for its *largest* difficulty, so one pool
/// serves every shape the benchmark cycles through.
fn automatic_gpu_batch_size(
    request: &HashRequest,
    difficulty_sequence: &[u32],
    explicit_max_batch_size: usize,
) -> Result<usize, String> {
    if !is_gpu_backend(&request.backend) {
        return Err("--auto-batch-size is only supported with --backend gpu".to_owned());
    }
    let device = tm_gpu::Device::open(request.device_id).map_err(|error| error.to_string())?;
    let free_memory = device
        .free_memory_bytes()
        .map_err(|error| error.to_string())?;

    let selected = if difficulty_sequence.is_empty() {
        tm_core::select_batch_size(
            RUNTIME,
            free_memory,
            request.difficulty,
            explicit_max_batch_size,
        )
        .selected_batch_size
    } else {
        select_batch_size_for_sequence(free_memory, difficulty_sequence, explicit_max_batch_size)
    };
    if selected == 0 {
        return Err("automatic GPU batch-size selection found no safe batch size".to_owned());
    }
    Ok(selected)
}

fn select_batch_size_for_sequence(
    free_memory_bytes: usize,
    difficulties: &[u32],
    explicit_max_batch_size: usize,
) -> usize {
    let Some(&max_difficulty) = difficulties.iter().max() else {
        return 0;
    };
    let memory_limited = estimate_memory_batch_limit(
        RUNTIME,
        free_memory_bytes,
        max_difficulty,
        DEFAULT_MEMORY_RESERVE_BYTES,
    );
    if memory_limited == 0 {
        return 0;
    }
    if explicit_max_batch_size > 0 {
        return memory_limited.min(explicit_max_batch_size);
    }
    // A tuned ceiling only applies when every difficulty in the sequence has one.
    let mut tuned = 0usize;
    for &difficulty in difficulties {
        let recommended = recommended_batch_size(difficulty);
        if recommended == 0 {
            tuned = 0;
            break;
        }
        tuned = if tuned == 0 {
            recommended
        } else {
            tuned.min(recommended)
        };
    }
    if tuned > 0 {
        return memory_limited.min(tuned);
    }
    memory_limited
}

// -------------------------------------------------------------------------- benchmarking

fn add_timings(target: &mut HashTimings, source: &HashTimings) {
    target.validation_ms += source.validation_ms;
    target.setup_ms += source.setup_ms;
    target.setup_normalize_cpu_ms += source.setup_normalize_cpu_ms;
    target.setup_activate_cpu_ms += source.setup_activate_cpu_ms;
    target.setup_device_info_cpu_ms += source.setup_device_info_cpu_ms;
    target.setup_params_cpu_ms += source.setup_params_cpu_ms;
    target.setup_backend_init_cpu_ms += source.setup_backend_init_cpu_ms;
    target.input_ms += source.input_ms;
    target.keygen_ms += source.keygen_ms;
    target.first_block_ms += source.first_block_ms;
    target.first_block_initial_hash_cpu_ms += source.first_block_initial_hash_cpu_ms;
    target.first_block_digest_cpu_ms += source.first_block_digest_cpu_ms;
    target.first_block_max_worker_ms += source.first_block_max_worker_ms;
    target.first_block_thread_launch_ms += source.first_block_thread_launch_ms;
    target.first_block_max_worker_start_ms += source.first_block_max_worker_start_ms;
    target.first_block_worker_start_span_ms += source.first_block_worker_start_span_ms;
    target.first_block_max_worker_finish_ms += source.first_block_max_worker_finish_ms;
    target.first_block_worker_finish_span_ms += source.first_block_worker_finish_span_ms;
    target.compute_ms += source.compute_ms;
    target.kernel_ms += source.kernel_ms;
    target.host_to_device_ms += source.host_to_device_ms;
    target.gpu_first_block_ms += source.gpu_first_block_ms;
    target.device_to_host_ms += source.device_to_host_ms;
    target.finalize_ms += source.finalize_ms;
    target.finalize_hash_ms += source.finalize_hash_ms;
    target.argon2_finalize_ms += source.argon2_finalize_ms;
    target.base64_ms += source.base64_ms;
    target.match_ms += source.match_ms;
    target.total_ms += source.total_ms;
}

fn run_benchmark(
    mut request: HashRequest,
    seconds: u32,
    json: bool,
    difficulty_sequence: &[u32],
    batch_size_sequence: &[usize],
    out: &mut dyn Write,
) -> u8 {
    let difficulties: Vec<u32> = if difficulty_sequence.is_empty() {
        vec![request.difficulty]
    } else {
        difficulty_sequence.to_vec()
    };
    let batch_sizes: Vec<usize> = if batch_size_sequence.is_empty() {
        vec![request.batch_size]
    } else {
        batch_size_sequence.to_vec()
    };
    if difficulties.len() != batch_sizes.len() && difficulties.len() != 1 && batch_sizes.len() != 1
    {
        let result = error_result(
            &request,
            &request.backend,
            "difficulty sequence and batch-size sequence lengths must match unless one sequence has length 1".to_owned(),
        );
        return print_result(&result, json, out);
    }

    let shape_count = difficulties.len().max(batch_sizes.len());
    for index in 0..shape_count {
        let mut probe = request.clone();
        probe.difficulty = difficulties[if difficulties.len() == 1 { 0 } else { index }];
        probe.batch_size = batch_sizes[if batch_sizes.len() == 1 { 0 } else { index }];
        // Through with_canonical_backend like every other validation site: tm-argon2 knows
        // only the wire name, so a probe built from a `gpu` request would be rejected as an
        // unsupported backend — which is exactly what hash-benchmark did while hash-one
        // worked, because this was the one site that skipped the mapping.
        if let Err(errors) = tm_argon2::validate_request(&with_canonical_backend(&probe)) {
            let message = if difficulty_sequence.is_empty() && batch_size_sequence.is_empty() {
                errors.to_string()
            } else {
                format!("benchmark sequence item {index}: {errors}")
            };
            let mut result = error_result(&request, &request.backend, message);
            result.batch_size = probe.batch_size;
            return print_result(&result, json, out);
        }
    }

    let mut backend = match make_reusable_backend(&request) {
        Ok(backend) => backend,
        Err(message) => {
            let result = error_result(&request, &request.backend, message);
            return print_result(&result, json, out);
        }
    };

    let mut aggregate = HashResult {
        request_id: request.request_id.clone(),
        algorithm: request.algorithm.clone(),
        backend: request.backend.clone(),
        device_id: request.device_id,
        batch_size: request.batch_size,
        ..Default::default()
    };
    let mut batch_size_range_seen = false;
    let mut first_block_range_seen = false;

    let start = Instant::now();
    let deadline = start + Duration::from_secs(u64::from(seconds));
    let mut shape_index = 0usize;
    while Instant::now() < deadline {
        request.difficulty = difficulties[if difficulties.len() == 1 { 0 } else { shape_index }];
        request.batch_size = batch_sizes[if batch_sizes.len() == 1 { 0 } else { shape_index }];
        shape_index = (shape_index + 1) % shape_count;

        let current = backend.run_batch(&request);
        if !current.ok {
            return print_result(&current, json, out);
        }
        aggregate.ok = true;
        aggregate.attempts += current.attempts;
        aggregate.batch_size = current.batch_size;
        if batch_size_range_seen {
            aggregate.batch_size_min = aggregate.batch_size_min.min(current.batch_size);
            aggregate.batch_size_max = aggregate.batch_size_max.max(current.batch_size);
        } else {
            aggregate.batch_size_min = current.batch_size;
            aggregate.batch_size_max = current.batch_size;
            batch_size_range_seen = true;
        }
        aggregate.first_block_dynamic_chunk_size = current.first_block_dynamic_chunk_size;
        aggregate.first_block_dynamic_chunk_auto = current.first_block_dynamic_chunk_auto;
        aggregate.first_block_worker_count = current.first_block_worker_count;
        aggregate.first_block_chunk_size = current.first_block_chunk_size;
        aggregate.gpu_first_blocks = current.gpu_first_blocks;
        if first_block_range_seen {
            aggregate.first_block_dynamic_chunk_size_min = aggregate
                .first_block_dynamic_chunk_size_min
                .min(current.first_block_dynamic_chunk_size);
            aggregate.first_block_dynamic_chunk_size_max = aggregate
                .first_block_dynamic_chunk_size_max
                .max(current.first_block_dynamic_chunk_size);
            aggregate.first_block_chunk_size_min = aggregate
                .first_block_chunk_size_min
                .min(current.first_block_chunk_size);
            aggregate.first_block_chunk_size_max = aggregate
                .first_block_chunk_size_max
                .max(current.first_block_chunk_size);
        } else {
            aggregate.first_block_dynamic_chunk_size_min = current.first_block_dynamic_chunk_size;
            aggregate.first_block_dynamic_chunk_size_max = current.first_block_dynamic_chunk_size;
            aggregate.first_block_chunk_size_min = current.first_block_chunk_size;
            aggregate.first_block_chunk_size_max = current.first_block_chunk_size;
            first_block_range_seen = true;
        }
        add_timings(&mut aggregate.timings, &current.timings);
        if !request.key.is_empty() {
            aggregate.hash.clone_from(&current.hash);
        }
        aggregate.matches.extend(current.matches);
    }

    if request.key.is_empty() {
        aggregate.hash.clear();
    }
    aggregate.elapsed_ms = millis_since(start);
    if aggregate.elapsed_ms > 0.0 {
        aggregate.hashrate = aggregate.attempts as f64 / (aggregate.elapsed_ms / 1000.0);
    }
    print_result(&aggregate, json, out)
}

// ------------------------------------------------------------------------------ output

fn millis_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Port of `hashapi::toJson`. The key order is the C++ order so the two outputs diff
/// line-for-line once pretty-printed, and every key is always present — the parity harness
/// reads `ok`, `error` and `hash` unconditionally.
pub fn to_json(result: &HashResult) -> String {
    serde_json::to_string(result).unwrap_or_else(|error| {
        format!("{{\"ok\":false,\"error\":\"result is not serialisable: {error}\"}}")
    })
}

fn print_result(result: &HashResult, json: bool, out: &mut dyn Write) -> u8 {
    if json {
        let _ = writeln!(out, "{}", to_json(result));
    } else if !result.ok {
        let _ = writeln!(std::io::stderr(), "Hash API error: {}", result.error);
    } else {
        let _ = writeln!(
            out,
            "ok=true backend={} attempts={} hashrate={} matches={}",
            result.backend,
            result.attempts,
            format_double(result.hashrate),
            result.matches.len()
        );
        if !result.hash.is_empty() {
            let _ = writeln!(out, "hash={}", result.hash);
        }
    }
    if result.ok {
        0
    } else {
        2
    }
}

/// `std::ostream`'s default double formatting: six significant digits, no trailing zeros.
fn format_double(value: f64) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    let formatted = format!("{value:.6e}");
    let (mantissa, exponent) = formatted.split_once('e').unwrap_or((formatted.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if (-5..6).contains(&exponent) {
        let decimals = (5 - exponent).max(0) as usize;
        let text = format!("{value:.decimals$}");
        let trimmed = if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            text
        };
        return trimmed;
    }
    let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
    format!("{mantissa}e{}{:02}", if exponent < 0 { "-" } else { "+" }, exponent.abs())
}
