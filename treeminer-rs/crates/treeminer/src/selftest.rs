//! Startup Argon2 CPU/GPU self-test. Port of `src/hashapi/HashApiSelfTest.cpp` and the
//! self-test block in `src/main.cpp`.
//!
//! This runs before journals, network threads, dashboards or mining exist, and it is the
//! only thing standing between the miner and submitting invalid blocks: a miscompiled or
//! broken kernel produces digests that look exactly like real ones locally and are rejected
//! by the server after the work has been paid for (commit 12e241c). A deterministic
//! reference comparison catches that even when the bad output happens to contain `XEN11`.
//!
//! Two verdicts come out of it, per device:
//!   * whether the device may mine at all — a mismatch means it is skipped;
//!   * whether it may derive Argon2's first blocks on the device. That path is the one that
//!     was miscompiled, so it is probed separately and a device that fails only the probe
//!     keeps its first blocks on the CPU rather than being dropped.

use crate::state::MiningState;

/// The known-good vector: `salt`, `key`, `m`, and the digest they must produce.
pub const SELF_TEST_SALT: &str = "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc";
pub const SELF_TEST_KEY: &str =
    "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f";
pub const SELF_TEST_DIFFICULTY: u32 = 8;
/// A pattern no digest can contain, so the self-test never manufactures a "find".
pub const SELF_TEST_PATTERN: &str = "SELFTEST-NO-MATCH";

/// The GPU backend's name in operator-facing text (`TREEMINER_GPU_BACKEND_NAME`).
pub const BACKEND_NAME: &str = "HIP";

/// A digest, or why one could not be produced. The error text is operator-facing.
pub type ProbeOutcome = Result<String, String>;

/// Where one self-test line belongs. The C++ sends failures to stderr and progress to
/// stdout; scripts in the field grep both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfTestLine {
    pub stream: Stream,
    pub text: String,
}

impl SelfTestLine {
    fn out(text: String) -> Self {
        Self {
            stream: Stream::Out,
            text,
        }
    }

    fn err(text: String) -> Self {
        Self {
            stream: Stream::Err,
            text,
        }
    }
}

/// What the self-test decided about one device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDecision {
    pub device_index: i32,
    /// False means the device is skipped entirely.
    pub mine: bool,
    /// Only meaningful when `mine`; false keeps first blocks on the CPU.
    pub gpu_first_blocks: bool,
    pub lines: Vec<SelfTestLine>,
}

/// Compare one GPU digest against the CPU reference, producing the C++ error text.
pub fn compare(expected: &str, actual: &ProbeOutcome, gpu_first_blocks: bool) -> Result<(), String> {
    let digest = match actual {
        Ok(digest) => digest,
        Err(error) => return Err(format!("{BACKEND_NAME} computation failed: {error}")),
    };
    if digest.is_empty() {
        return Err(format!("{BACKEND_NAME} computation returned an empty digest"));
    }
    if digest != expected {
        return Err(format!(
            "CPU/{BACKEND_NAME} Argon2 digest mismatch (gpu_first_blocks={gpu_first_blocks})"
        ));
    }
    Ok(())
}

/// The decision for one device, given the two probe results.
///
/// `first_blocks` is the second probe (device-side first blocks) and is only consulted when
/// the base probe passed; `None` means the base probe already exercised that path, which is
/// how a default-on backend behaves.
pub fn decide_device(
    device_index: i32,
    expected: &str,
    base: &ProbeOutcome,
    base_used_gpu_first_blocks: bool,
    first_blocks: Option<&ProbeOutcome>,
) -> DeviceDecision {
    if let Err(error) = compare(expected, base, base_used_gpu_first_blocks) {
        return DeviceDecision {
            device_index,
            mine: false,
            gpu_first_blocks: false,
            lines: vec![SelfTestLine::err(format!(
                "WARN: GPU #{device_index} failed startup Argon2 CPU/{BACKEND_NAME} self-test: \
                 {error} — skipping this device."
            ))],
        };
    }

    let mut lines = vec![SelfTestLine::out(format!(
        "GPU #{device_index} Argon2 CPU/{BACKEND_NAME} self-test passed."
    ))];

    let gpu_first_blocks = match first_blocks {
        // The base probe already ran the GPU first-blocks path and matched.
        None => base_used_gpu_first_blocks,
        Some(probe) => match compare(expected, probe, true) {
            Ok(()) => {
                lines.push(SelfTestLine::out(format!(
                    "GPU #{device_index} GPU-first-blocks probe matched the CPU reference; \
                     enabling GPU first blocks for this device."
                )));
                true
            }
            Err(error) => {
                lines.push(SelfTestLine::out(format!(
                    "GPU #{device_index} GPU-first-blocks probe still mismatches ({error}); \
                     mining stays on CPU first-blocks."
                )));
                false
            }
        },
    };

    DeviceDecision {
        device_index,
        mine: true,
        gpu_first_blocks,
        lines,
    }
}

/// The source of digests the self-test compares. Implemented by the real GPU path in
/// [`crate::backend`]; the tests substitute a fake so the decision logic is exercised
/// without hardware.
pub trait SelfTestProbe {
    /// The CPU reference digest for the known-good vector.
    fn cpu_reference(&mut self) -> ProbeOutcome;

    /// One device's digest for the same vector. An error covers both "the device could not
    /// be opened" and "the batch failed"; the C++ treats them identically.
    fn gpu_digest(&mut self, device_index: i32, gpu_first_blocks: bool) -> ProbeOutcome;
}

/// Every device's verdict, plus the lines to print.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfTestReport {
    pub decisions: Vec<DeviceDecision>,
    /// Set when the CPU reference itself could not be produced: nothing can be compared, so
    /// the whole self-test is fatal rather than per-device.
    pub reference_error: Option<String>,
}

impl SelfTestReport {
    pub fn mining_devices(&self) -> Vec<i32> {
        self.decisions
            .iter()
            .filter(|decision| decision.mine)
            .map(|decision| decision.device_index)
            .collect()
    }

    /// The C++ fatal condition: not one device survived.
    pub fn is_fatal(&self) -> bool {
        self.reference_error.is_some() || self.mining_devices().is_empty()
    }

    pub fn fatal_message(&self) -> String {
        match &self.reference_error {
            Some(error) => format!(
                "FATAL: startup Argon2 CPU/{BACKEND_NAME} self-test could not run: {error}. \
                 Mining was not started."
            ),
            None => format!(
                "FATAL: no GPU passed the startup Argon2 CPU/{BACKEND_NAME} self-test. \
                 Mining was not started."
            ),
        }
    }

    /// Print every line to the stream the C++ sent it to.
    pub fn emit(&self) {
        for line in self.decisions.iter().flat_map(|decision| &decision.lines) {
            match line.stream {
                Stream::Out => println!("{}", line.text),
                Stream::Err => eprintln!("{}", line.text),
            }
        }
    }

    /// Record each device's first-blocks verdict where the mining loop reads it.
    pub fn apply(&self, state: &MiningState) {
        for decision in &self.decisions {
            if decision.mine {
                state.set_gpu_first_blocks_verified(decision.device_index, decision.gpu_first_blocks);
            }
        }
    }
}

/// Run the self-test over `devices`.
///
/// `default_gpu_first_blocks` mirrors `hashapi::kGpuFirstBlocksEnabled`: when it is already
/// on, the base probe exercised the device path and no second probe is needed; when it is
/// off (the HIP default) every device that passed still gets probed, because a device that
/// matches on the fast path should use it.
pub fn run_self_test(
    devices: &[i32],
    default_gpu_first_blocks: bool,
    probe: &mut dyn SelfTestProbe,
) -> SelfTestReport {
    let expected = match probe.cpu_reference() {
        Ok(digest) if !digest.is_empty() => digest,
        Ok(_) => {
            return SelfTestReport {
                decisions: Vec::new(),
                reference_error: Some(
                    "CPU reference returned an invalid PHC string".to_owned(),
                ),
            }
        }
        Err(error) => {
            return SelfTestReport {
                decisions: Vec::new(),
                reference_error: Some(format!("CPU reference failed: {error}")),
            }
        }
    };

    let decisions = devices
        .iter()
        .map(|&index| {
            let base = probe.gpu_digest(index, default_gpu_first_blocks);
            // A device that already failed the base comparison is skipped outright, so the
            // second probe would only cost time on a device that will not mine anyway.
            let base_matched = compare(&expected, &base, default_gpu_first_blocks).is_ok();
            let first_blocks = if default_gpu_first_blocks || !base_matched {
                None
            } else {
                Some(probe.gpu_digest(index, true))
            };
            decide_device(
                index,
                &expected,
                &base,
                default_gpu_first_blocks,
                first_blocks.as_ref(),
            )
        })
        .collect();

    SelfTestReport {
        decisions,
        reference_error: None,
    }
}

/// The CPU reference digest for the known-good vector — the value every device is compared
/// against, and the same code path the mining loop's CPU sidecar uses.
pub fn cpu_reference_digest() -> ProbeOutcome {
    let phc = tm_argon2::argon2id_phc(SELF_TEST_SALT, SELF_TEST_KEY, SELF_TEST_DIFFICULTY)
        .map_err(|error| error.to_string())?;
    tm_core::phc_digest(&phc)
        .map(str::to_owned)
        .ok_or_else(|| "CPU reference returned an invalid PHC string".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = "digest-that-matches";

    struct FakeProbe {
        reference: ProbeOutcome,
        /// `(device, gpu_first_blocks) -> outcome`
        answers: Vec<((i32, bool), ProbeOutcome)>,
        calls: Vec<(i32, bool)>,
    }

    impl FakeProbe {
        fn new(answers: Vec<((i32, bool), ProbeOutcome)>) -> Self {
            Self {
                reference: Ok(GOOD.to_owned()),
                answers,
                calls: Vec::new(),
            }
        }
    }

    impl SelfTestProbe for FakeProbe {
        fn cpu_reference(&mut self) -> ProbeOutcome {
            self.reference.clone()
        }

        fn gpu_digest(&mut self, device_index: i32, gpu_first_blocks: bool) -> ProbeOutcome {
            self.calls.push((device_index, gpu_first_blocks));
            self.answers
                .iter()
                .find(|(key, _)| *key == (device_index, gpu_first_blocks))
                .map(|(_, outcome)| outcome.clone())
                .unwrap_or_else(|| Err("no fake answer".to_owned()))
        }
    }

    #[test]
    fn a_matching_device_mines_with_gpu_first_blocks() {
        let mut probe = FakeProbe::new(vec![
            ((0, false), Ok(GOOD.to_owned())),
            ((0, true), Ok(GOOD.to_owned())),
        ]);
        let report = run_self_test(&[0], false, &mut probe);

        assert!(!report.is_fatal());
        assert_eq!(report.mining_devices(), vec![0]);
        assert!(report.decisions[0].gpu_first_blocks);
        assert_eq!(probe.calls, vec![(0, false), (0, true)]);
    }

    #[test]
    fn a_device_failing_only_the_first_blocks_probe_still_mines_on_cpu_first_blocks() {
        let mut probe = FakeProbe::new(vec![
            ((0, false), Ok(GOOD.to_owned())),
            ((0, true), Ok("something-else".to_owned())),
        ]);
        let report = run_self_test(&[0], false, &mut probe);

        assert!(!report.is_fatal());
        assert_eq!(report.mining_devices(), vec![0]);
        assert!(!report.decisions[0].gpu_first_blocks);
        assert!(report.decisions[0]
            .lines
            .iter()
            .any(|line| line.text.contains("mining stays on CPU first-blocks")));
    }

    #[test]
    fn a_device_failing_the_base_self_test_is_skipped_and_never_probed_further() {
        let mut probe = FakeProbe::new(vec![
            ((0, false), Ok("wrong".to_owned())),
            ((1, false), Ok(GOOD.to_owned())),
            ((1, true), Ok(GOOD.to_owned())),
        ]);
        let report = run_self_test(&[0, 1], false, &mut probe);

        assert_eq!(report.mining_devices(), vec![1]);
        assert!(!report.is_fatal());
        assert!(!probe.calls.contains(&(0, true)));
        assert!(report.decisions[0].lines[0]
            .text
            .contains("skipping this device"));
        assert_eq!(report.decisions[0].lines[0].stream, Stream::Err);
    }

    #[test]
    fn a_device_that_cannot_be_opened_is_skipped_like_a_mismatch() {
        let mut probe = FakeProbe::new(vec![((0, false), Err("no such device".to_owned()))]);
        let report = run_self_test(&[0], false, &mut probe);

        assert!(report.is_fatal());
        assert!(report.decisions[0].lines[0].text.contains("no such device"));
    }

    #[test]
    fn every_device_failing_is_fatal() {
        let mut probe = FakeProbe::new(vec![
            ((0, false), Ok("wrong".to_owned())),
            ((1, false), Ok("also wrong".to_owned())),
        ]);
        let report = run_self_test(&[0, 1], false, &mut probe);

        assert!(report.is_fatal());
        assert!(report.mining_devices().is_empty());
        assert!(report.fatal_message().contains("no GPU passed"));
    }

    #[test]
    fn a_broken_cpu_reference_is_fatal_before_any_device_is_touched() {
        let mut probe = FakeProbe::new(Vec::new());
        probe.reference = Err("argon2 unavailable".to_owned());
        let report = run_self_test(&[0, 1], false, &mut probe);

        assert!(report.is_fatal());
        assert!(probe.calls.is_empty());
        assert!(report.fatal_message().contains("could not run"));
    }

    #[test]
    fn a_default_on_backend_trusts_the_base_probe_without_a_second_one() {
        let mut probe = FakeProbe::new(vec![((0, true), Ok(GOOD.to_owned()))]);
        let report = run_self_test(&[0], true, &mut probe);

        assert!(report.decisions[0].gpu_first_blocks);
        assert_eq!(probe.calls, vec![(0, true)]);
    }

    #[test]
    fn verdicts_land_where_the_mining_loop_reads_them() {
        let mut probe = FakeProbe::new(vec![
            ((0, false), Ok(GOOD.to_owned())),
            ((0, true), Ok(GOOD.to_owned())),
            ((1, false), Ok(GOOD.to_owned())),
            ((1, true), Ok("wrong".to_owned())),
        ]);
        let report = run_self_test(&[0, 1], false, &mut probe);
        let state = MiningState::for_test(1000);
        report.apply(&state);

        assert!(state.gpu_first_blocks_verified(0));
        assert!(!state.gpu_first_blocks_verified(1));
    }

    #[test]
    fn the_cpu_reference_reproduces_the_known_good_vector() {
        const EXPECTED: &str = "2PKfnaEX2s+Yf/Drzi92D8HJ+B6K+FppyT7g5glp2knIMlFGWhnyOb9r1QIPf0GaVUEw8KumqQZ/pK2dkNTDxA";
        assert_eq!(cpu_reference_digest().as_deref(), Ok(EXPECTED));
    }
}
