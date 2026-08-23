//! The network-free hash CLI, checked against the C++ binary it has to stay diffable with.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;
use treeminer::hashcli::{
    base_request, parse_args, parse_batch_size_sequence, parse_difficulty_sequence, run_with,
};

/// The C++ miner, used as the oracle for the JSON shape. Absent on a machine that has not
/// built it, in which case those tests skip rather than fail.
const CPP_BINARY: &str = "/tmp/claude-1000/-home-codebam-Documents-tree-miner-amd/083f64a4-9069-40f9-adf6-b87ab41c106e/scratchpad/nixbuild/bin/xenblocksMiner";

fn argv(parts: &[&str]) -> Vec<String> {
    std::iter::once("treeminer".to_owned())
        .chain(parts.iter().map(|part| (*part).to_owned()))
        .collect()
}

fn run(parts: &[&str]) -> (u8, String, String) {
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = run_with(&argv(parts), &mut out, &mut err);
    (
        code,
        String::from_utf8(out).expect("stdout is utf-8"),
        String::from_utf8(err).expect("stderr is utf-8"),
    )
}

fn run_json(parts: &[&str]) -> (u8, Value) {
    let (code, out, _) = run(parts);
    let value = serde_json::from_str(out.trim()).unwrap_or_else(|error| {
        panic!("expected JSON on stdout, got {out:?}: {error}");
    });
    (code, value)
}

#[derive(serde::Deserialize)]
struct Vector {
    salt_hex: String,
    key: String,
    difficulty: u32,
    phc: String,
}

fn load_vectors() -> Vec<Vector> {
    #[derive(serde::Deserialize)]
    struct Fixture {
        vectors: Vec<Vector>,
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/argon2_vectors.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_str::<Fixture>(&text)
        .expect("fixture is valid JSON")
        .vectors
}

// ------------------------------------------------------------------ argument parsing

#[test]
fn hash_one_arguments_parse_into_a_request() {
    let args = parse_args(&argv(&[
        "hash-one",
        "--salt",
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "--key",
        "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f",
        "--backend",
        "cuda",
        "--difficulty",
        "64",
        "--no-xuni",
        "--detailed-timings",
        "--gpu-first-blocks",
        "--json",
    ]));
    assert!(args.flag("--json"));
    assert!(args.flag("--gpu-first-blocks"));
    let request = base_request(&args).expect("request parses");
    assert_eq!(request.backend, "cuda");
    assert_eq!(request.difficulty, 64);
    assert_eq!(request.key.len(), 64);
    assert!(!request.allow_xuni);
    assert!(request.detailed_timings);
    assert!(request.gpu_first_blocks);
}

#[test]
fn hash_batch_arguments_parse_into_a_request() {
    let args = parse_args(&argv(&[
        "hash-batch",
        "--salt",
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "--prefix",
        "abcd",
        "--pattern",
        "XUNI",
        "--batch-size",
        "512",
        "--auto-batch-size",
        "--first-block-workers",
        "6",
        "--first-block-dynamic-chunk-size",
        "16",
        "--first-block-dynamic-chunk-auto",
        "--device",
        "1",
    ]));
    assert!(args.flag("--auto-batch-size"));
    let request = base_request(&args).expect("request parses");
    assert_eq!(request.key_prefix, "abcd");
    assert_eq!(request.target_pattern, "XUNI");
    assert_eq!(request.batch_size, 512);
    assert_eq!(request.device_id, 1);
    assert_eq!(request.first_block_workers, 6);
    assert_eq!(request.first_block_dynamic_chunk_size, 16);
    assert!(request.first_block_dynamic_chunk_auto);
}

#[test]
fn hash_benchmark_sequences_parse() {
    let args = parse_args(&argv(&[
        "hash-benchmark",
        "--salt",
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "--seconds",
        "5",
        "--difficulty-sequence",
        "1,8,64",
        "--batch-size-sequence",
        "2048,1024,512",
    ]));
    assert_eq!(args.uint("--seconds", 30), Ok(5));
    assert_eq!(
        parse_difficulty_sequence(&args.get("--difficulty-sequence", "")),
        Ok(vec![1, 8, 64])
    );
    assert_eq!(
        parse_batch_size_sequence(&args.get("--batch-size-sequence", "")),
        Ok(vec![2048, 1024, 512])
    );
}

#[test]
fn defaults_match_the_cpp_request() {
    let request = base_request(&parse_args(&argv(&["hash-one"]))).expect("request parses");
    assert_eq!(request.algorithm, "argon2id-xen");
    assert_eq!(request.backend, "cpu");
    assert_eq!(request.target_pattern, "XEN11");
    assert_eq!(request.difficulty, 42069);
    assert_eq!(request.batch_size, 1);
    assert!(request.allow_xuni);
}

/// The C++ ignores anything that is not a `--flag`, and a bool flag never eats the next
/// argument. The parity harness relies on both when it appends flags to a fixed command.
#[test]
fn stray_arguments_are_ignored_and_flags_do_not_consume_values() {
    let args = parse_args(&argv(&[
        "hash-one", "junk", "--json", "--difficulty", "8", "--dangling",
    ]));
    assert!(args.flag("--json"));
    assert_eq!(args.uint("--difficulty", 0), Ok(8));
    assert!(!args.has("--dangling"), "a trailing flag with no value is dropped");
}

#[test]
fn malformed_sequences_are_rejected() {
    for text in ["1,,8", "1,x", "0", "1,0"] {
        assert!(
            parse_difficulty_sequence(text).is_err(),
            "difficulty sequence {text:?} should be rejected"
        );
        assert!(
            parse_batch_size_sequence(text).is_err(),
            "batch-size sequence {text:?} should be rejected"
        );
    }
    assert_eq!(parse_difficulty_sequence(""), Ok(Vec::new()));
}

#[test]
fn a_non_numeric_flag_is_an_error_not_a_panic() {
    let (code, value) = run_json(&["hash-batch", "--salt", "abcdabcdabcdabcd", "--batch-size", "many", "--json"]);
    assert_eq!(code, 2);
    assert_eq!(value["ok"], Value::Bool(false));
    assert!(value["error"].as_str().is_some_and(|text| !text.is_empty()));
}

// ------------------------------------------------------------------------ subcommands

#[test]
fn hash_help_prints_usage_and_succeeds() {
    let (code, out, _) = run(&["hash-help"]);
    assert_eq!(code, 0);
    assert!(out.contains("hash-one"));
    assert!(out.contains("hash-benchmark"));
}

#[test]
fn an_unknown_subcommand_prints_usage_and_fails() {
    let (code, out, _) = run(&["hash-nonsense"]);
    assert_eq!(code, 1);
    assert!(out.contains("Hash API commands"));
}

#[test]
fn is_hash_api_command_recognises_exactly_the_four_subcommands() {
    for command in ["hash-one", "hash-batch", "hash-benchmark", "hash-help"] {
        assert!(treeminer::is_hash_api_command(&argv(&[command])), "{command}");
    }
    assert!(!treeminer::is_hash_api_command(&argv(&["--help"])));
    assert!(!treeminer::is_hash_api_command(&argv(&[])));
}

/// The end-to-end check the parity harness performs, run against the fixtures in-process.
#[test]
fn hash_one_on_the_cpu_reproduces_the_fixture_vectors() {
    let vectors = load_vectors();
    assert!(!vectors.is_empty());
    for vector in &vectors {
        let difficulty = vector.difficulty.to_string();
        let (code, value) = run_json(&[
            "hash-one",
            "--salt",
            &vector.salt_hex,
            "--key",
            &vector.key,
            "--backend",
            "cpu",
            "--difficulty",
            &difficulty,
            "--json",
        ]);
        assert_eq!(code, 0, "m={difficulty}: {}", value["error"]);
        assert_eq!(value["ok"], Value::Bool(true));
        assert_eq!(value["hash"], Value::String(vector.phc.clone()));
        assert_eq!(value["attempts"], Value::from(1));
        assert_eq!(value["backend"], Value::String("cpu".to_owned()));
    }
}

#[test]
fn an_invalid_request_reports_the_error_in_the_result() {
    let (code, value) = run_json(&["hash-one", "--salt", "zz", "--json"]);
    assert_eq!(code, 2);
    assert_eq!(value["ok"], Value::Bool(false));
    assert!(value["error"]
        .as_str()
        .expect("error is a string")
        .contains("salt_hex"));
    // Even a failure carries the whole schema, which is what makes the outputs diffable.
    assert!(value.get("timings").is_some());
}

#[test]
fn hash_batch_generates_the_requested_number_of_keys() {
    let (code, value) = run_json(&[
        "hash-batch",
        "--salt",
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "--backend",
        "cpu",
        "--difficulty",
        "8",
        "--batch-size",
        "4",
        "--json",
    ]);
    assert_eq!(code, 0);
    assert_eq!(value["attempts"], Value::from(4));
    assert_eq!(value["batch_size"], Value::from(4));
}

#[test]
fn auto_batch_size_is_rejected_on_the_cpu_backend() {
    let (code, value) = run_json(&[
        "hash-batch",
        "--salt",
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "--backend",
        "cpu",
        "--auto-batch-size",
        "--json",
    ]);
    assert_eq!(code, 2);
    assert!(value["error"]
        .as_str()
        .expect("error is a string")
        .contains("--backend gpu"));
}

// -------------------------------------------------------------- the C++ JSON contract

fn cpp_hash_one(extra: &[&str]) -> Option<Value> {
    if !std::path::Path::new(CPP_BINARY).exists() {
        eprintln!("skipping: {CPP_BINARY} is not built on this machine");
        return None;
    }
    let output = Command::new(CPP_BINARY)
        .args(["hash-one", "--json"])
        .args(extra)
        .output()
        .expect("the C++ miner runs");
    Some(
        serde_json::from_slice(&output.stdout)
            .unwrap_or_else(|error| panic!("C++ JSON: {error}")),
    )
}

/// Key sets, not values: the timings differ every run, but a renamed or dropped field is a
/// port bug that would silently break the differential harness.
fn assert_same_keys(rust: &Value, cpp: &Value, path: &str) {
    let rust_keys: Vec<&String> = rust
        .as_object()
        .unwrap_or_else(|| panic!("{path} is not an object in the Rust output"))
        .keys()
        .collect();
    let cpp_keys: Vec<&String> = cpp
        .as_object()
        .unwrap_or_else(|| panic!("{path} is not an object in the C++ output"))
        .keys()
        .collect();
    assert_eq!(rust_keys, cpp_keys, "{path} keys differ");
}

#[test]
fn the_json_keys_match_the_cpp_binary() {
    let salt = "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc";
    let key = "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f";
    let Some(cpp) = cpp_hash_one(&[
        "--salt", salt, "--key", key, "--backend", "cpu", "--difficulty", "8",
    ]) else {
        return;
    };
    let (code, rust) = run_json(&[
        "hash-one", "--salt", salt, "--key", key, "--backend", "cpu", "--difficulty", "8",
        "--json",
    ]);
    assert_eq!(code, 0);
    assert_same_keys(&rust, &cpp, "result");
    assert_same_keys(&rust["timings"], &cpp["timings"], "result.timings");
    // The digest, and the fields the parity harness reads, must agree exactly.
    assert_eq!(rust["hash"], cpp["hash"]);
    assert_eq!(rust["ok"], cpp["ok"]);
    assert_eq!(rust["error"], cpp["error"]);
    assert_eq!(rust["algorithm"], cpp["algorithm"]);
    assert_eq!(rust["backend"], cpp["backend"]);
    assert_eq!(rust["attempts"], cpp["attempts"]);
}

/// A failed request has to carry the same schema too — the harness parses it the same way.
#[test]
fn the_json_keys_match_the_cpp_binary_on_failure() {
    let Some(cpp) = cpp_hash_one(&["--salt", "zz"]) else {
        return;
    };
    let (code, rust) = run_json(&["hash-one", "--salt", "zz", "--json"]);
    assert_eq!(code, 2);
    assert_same_keys(&rust, &cpp, "result");
    assert_same_keys(&rust["timings"], &cpp["timings"], "result.timings");
    assert_eq!(rust["ok"], cpp["ok"]);
    assert_eq!(rust["error"], cpp["error"]);
}

// ------------------------------------------------------- vendor-neutral backend spelling

/// `gpu` is the spelling the usage text advertises; `cuda` is the C++ miner's name for the
/// same backend and stays accepted, because `tests/parity/run_parity.sh` drives both
/// binaries with one command line.
#[test]
fn both_backend_spellings_parse_into_a_request() {
    for backend in ["gpu", "cuda"] {
        let request = base_request(&parse_args(&argv(&["hash-one", "--backend", backend])))
            .expect("request parses");
        assert_eq!(request.backend, backend);
    }
}

#[test]
fn the_usage_text_advertises_gpu_and_not_cuda() {
    let (code, out, _) = run(&["hash-help"]);
    assert_eq!(code, 0);
    assert!(out.contains("--backend cpu|gpu"));
    assert!(!out.to_lowercase().contains("cuda"), "usage still names CUDA");
}

/// A typo used to fall through to the CPU backend and silently hash for hours on the wrong
/// device. It has to be a hard error that says what the valid values are.
#[test]
fn an_unknown_backend_is_rejected_by_name() {
    let (code, value) = run_json(&[
        "hash-one",
        "--salt",
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "--backend",
        "rocm",
        "--json",
    ]);
    assert_eq!(code, 2);
    let error = value["error"].as_str().expect("error is a string");
    assert!(error.contains("rocm"), "{error}");
    assert!(error.contains("cpu"), "{error}");
    assert!(error.contains("gpu"), "{error}");
    // The failure still carries the whole schema, like every other failed request.
    assert!(value.get("timings").is_some());
}

/// `--auto-batch-size` needs a device; the message must name the flag the operator should
/// have passed, in the spelling the help text uses.
#[test]
fn auto_batch_size_names_the_gpu_backend_in_its_error() {
    let (code, value) = run_json(&[
        "hash-batch",
        "--salt",
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "--backend",
        "cpu",
        "--auto-batch-size",
        "--json",
    ]);
    assert_eq!(code, 2);
    let error = value["error"].as_str().expect("error is a string");
    assert!(error.contains("--backend gpu"), "{error}");
    assert!(!error.contains("cuda"), "{error}");
}

// ---------------------------------------------------------------------- the gpu path

/// The GPU backend end to end through the CLI. Batches stay tiny: the card is shared with
/// a live miner, so this is a correctness check, not a throughput one.
#[test]
fn hash_one_on_the_gpu_reproduces_a_fixture_vector() {
    if !tm_gpu::gpu_available() {
        eprintln!("skipping: no GPU present");
        return;
    }
    let vector = load_vectors()
        .into_iter()
        .find(|vector| vector.difficulty == 8)
        .expect("a difficulty-8 fixture");
    let (code, value) = run_json(&[
        "hash-one",
        "--salt",
        &vector.salt_hex,
        "--key",
        &vector.key,
        "--backend",
        "cuda",
        "--difficulty",
        "8",
        "--json",
    ]);
    assert_eq!(code, 0, "{}", value["error"]);
    assert_eq!(value["backend"], Value::String("cuda".to_owned()));
    // The GPU path reports the bare digest; the CPU path reports the whole PHC string.
    // That asymmetry is the C++ behaviour and the parity harness splits on `$` for it.
    assert!(
        vector.phc.ends_with(value["hash"].as_str().expect("hash is a string")),
        "gpu digest {} is not the tail of {}",
        value["hash"],
        vector.phc
    );
    assert_eq!(value["gpu_first_blocks"], Value::Bool(false));
}

/// A multi-job GPU batch exercises the threaded first-block fill from the CLI's side.
#[test]
fn hash_batch_on_the_gpu_runs_a_small_batch() {
    if !tm_gpu::gpu_available() {
        eprintln!("skipping: no GPU present");
        return;
    }
    let (code, value) = run_json(&[
        "hash-batch",
        "--salt",
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "--backend",
        "cuda",
        "--difficulty",
        "8",
        "--batch-size",
        "8",
        "--first-block-workers",
        "4",
        "--json",
    ]);
    assert_eq!(code, 0, "{}", value["error"]);
    assert_eq!(value["attempts"], Value::from(8));
    assert_eq!(value["first_block_worker_count"], Value::from(4));
    assert_eq!(value["first_block_chunk_size"], Value::from(2));
}

/// The JSON `backend` field echoes the spelling that was asked for: the parity harness
/// diffs it against the C++ miner, which only knows `cuda`.
#[test]
fn the_json_backend_field_echoes_the_requested_spelling() {
    if !tm_gpu::gpu_available() {
        eprintln!("skipping: no GPU present");
        return;
    }
    let vector = load_vectors()
        .into_iter()
        .find(|vector| vector.difficulty == 8)
        .expect("a difficulty-8 fixture");
    let mut digests = Vec::new();
    for backend in ["gpu", "cuda"] {
        let (code, value) = run_json(&[
            "hash-one",
            "--salt",
            &vector.salt_hex,
            "--key",
            &vector.key,
            "--backend",
            backend,
            "--difficulty",
            "8",
            "--json",
        ]);
        assert_eq!(code, 0, "{}", value["error"]);
        assert_eq!(value["backend"], Value::String(backend.to_owned()));
        digests.push(value["hash"].clone());
    }
    assert_eq!(digests[0], digests[1], "the alias took a different path");
}

/// Without a device the two spellings still have to fail identically, and each has to say
/// which backend was asked for.
#[test]
fn both_gpu_spellings_report_the_same_failure_shape() {
    if tm_gpu::gpu_available() {
        eprintln!("skipping: a GPU is present, so this path succeeds");
        return;
    }
    for backend in ["gpu", "cuda"] {
        let (code, value) = run_json(&[
            "hash-one",
            "--salt",
            "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
            "--key",
            "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f",
            "--backend",
            backend,
            "--difficulty",
            "8",
            "--json",
        ]);
        assert_eq!(code, 2);
        assert_eq!(value["backend"], Value::String(backend.to_owned()));
    }
}

/// Regression: `hash-benchmark --backend gpu` failed with "unsupported backend: gpu" while
/// `hash-one --backend gpu` worked, because the benchmark's per-shape validation probe was
/// the one site that skipped the canonical mapping. Found on the first NVIDIA test run,
/// after the whole correctness suite had passed — a hole no digest test can see.
#[test]
fn every_subcommand_accepts_both_backend_spellings() {
    for subcommand in ["hash-one", "hash-batch", "hash-benchmark"] {
        for backend in ["gpu", "cuda", "cpu"] {
            let mut args = vec![
                "treeminer".to_owned(),
                subcommand.to_owned(),
                "--salt".to_owned(),
                "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc".to_owned(),
                "--backend".to_owned(),
                backend.to_owned(),
                "--difficulty".to_owned(),
                "8".to_owned(),
                "--seconds".to_owned(),
                "0".to_owned(),
                "--json".to_owned(),
            ];
            if subcommand == "hash-one" {
                args.push("--key".to_owned());
                args.push(
                    "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f".to_owned(),
                );
            }
            let mut out = Vec::new();
            let mut err = Vec::new();
            run_with(&args, &mut out, &mut err);
            let text = String::from_utf8_lossy(&out).into_owned()
                + &String::from_utf8_lossy(&err);
            assert!(
                !text.contains("unsupported backend"),
                "{subcommand} rejected --backend {backend}: {text}"
            );
        }
    }
}
