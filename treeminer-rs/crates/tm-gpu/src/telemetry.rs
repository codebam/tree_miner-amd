//! GPU power and utilisation via ROCm SMI. Port of `src/gpu/GpuTelemetry.cpp`.
//!
//! The library is loaded at runtime, not linked: a driver without ROCm SMI installed still
//! mines, it just reports nothing. Every failure path here degrades to "unavailable" —
//! telemetry must never be able to stop the miner.

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::{Path, PathBuf};
use std::ptr;

/// What one device reports. Absent fields mean the sensor (or the library) is unavailable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceTelemetry {
    pub power_milliwatts: Option<u32>,
    pub utilization_percent: Option<u32>,
}

const RSMI_STATUS_SUCCESS: c_int = 0;

type RsmiInit = unsafe extern "C" fn(u64) -> c_int;
type RsmiShutDown = unsafe extern "C" fn() -> c_int;
type RsmiNumMonitorDevices = unsafe extern "C" fn(*mut u32) -> c_int;
type RsmiDevPciIdGet = unsafe extern "C" fn(u32, *mut u64) -> c_int;
type RsmiDevPowerAveGet = unsafe extern "C" fn(u32, u32, *mut u64) -> c_int;
type RsmiDevBusyPercentGet = unsafe extern "C" fn(u32, *mut u32) -> c_int;

struct Symbols {
    handle: *mut c_void,
    shut_down: RsmiShutDown,
    num_monitor_devices: RsmiNumMonitorDevices,
    pci_id_get: RsmiDevPciIdGet,
    power_ave_get: RsmiDevPowerAveGet,
    busy_percent_get: RsmiDevBusyPercentGet,
}

/// Holds ROCm SMI open for the duration of a reporting pass, as the C++ session does.
///
/// Not `Sync`: ROCm SMI is not documented as thread safe, so one session belongs to one
/// reporting thread.
pub struct TelemetrySession {
    symbols: Option<Symbols>,
}

impl std::fmt::Debug for TelemetrySession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TelemetrySession")
            .field("available", &self.available())
            .finish()
    }
}

impl TelemetrySession {
    /// Loads and initialises ROCm SMI. Never fails: an unavailable library yields a session
    /// that reports nothing.
    pub fn new() -> Self {
        Self {
            symbols: load_symbols(),
        }
    }

    pub fn available(&self) -> bool {
        self.symbols.is_some()
    }

    /// `"ROCm SMI"` when loaded, `"none"` otherwise — the string the logs print.
    pub fn source_name(&self) -> &'static str {
        if self.available() {
            "ROCm SMI"
        } else {
            "none"
        }
    }

    /// Telemetry for the HIP device at `device_index`, whose PCI bus byte is `bus_id`.
    ///
    /// ROCm SMI enumerates devices in its own order, so the bus id is what actually
    /// identifies the card; the HIP index is only the fallback.
    pub fn query(&self, device_index: i32, bus_id: i32) -> DeviceTelemetry {
        let mut telemetry = DeviceTelemetry::default();
        let Some(symbols) = self.symbols.as_ref() else {
            return telemetry;
        };
        if device_index < 0 {
            return telemetry;
        }
        let mut smi_index = device_index as u32;
        if bus_id >= 0 {
            if let Some(resolved) = find_smi_index_for_bus(symbols, bus_id) {
                smi_index = resolved;
            }
        }

        let mut microwatts = 0u64;
        // SAFETY: the pointers are live out-parameters and the symbols came from the
        // library handle this session keeps open for its whole lifetime.
        //
        // rsmi_dev_power_ave_get is deprecated in ROCm 6 in favour of rsmi_dev_power_get,
        // but the average-power sensor is what every supported release exposes.
        if unsafe { (symbols.power_ave_get)(smi_index, 0, &mut microwatts) }
            == RSMI_STATUS_SUCCESS
        {
            telemetry.power_milliwatts = u32::try_from(microwatts / 1000).ok();
        }
        let mut busy = 0u32;
        // SAFETY: as above.
        if unsafe { (symbols.busy_percent_get)(smi_index, &mut busy) } == RSMI_STATUS_SUCCESS {
            telemetry.utilization_percent = Some(busy);
        }
        telemetry
    }
}

impl Default for TelemetrySession {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TelemetrySession {
    fn drop(&mut self) {
        if let Some(symbols) = self.symbols.take() {
            // SAFETY: the handle and symbol were produced together by `load_symbols` and
            // are used exactly once, here, before the library is closed.
            unsafe {
                (symbols.shut_down)();
                libc::dlclose(symbols.handle);
            }
        }
    }
}

/// `rsmi_dev_pci_id_get` packs the BDF as `(domain << 32) | (bus << 8) | (device << 3) | fn`.
fn find_smi_index_for_bus(symbols: &Symbols, bus_id: i32) -> Option<u32> {
    let mut count = 0u32;
    // SAFETY: live out-parameter; symbol belongs to the open library.
    if unsafe { (symbols.num_monitor_devices)(&mut count) } != RSMI_STATUS_SUCCESS {
        return None;
    }
    (0..count).find(|index| {
        let mut pci_id = 0u64;
        // SAFETY: live out-parameter; symbol belongs to the open library.
        let ok = unsafe { (symbols.pci_id_get)(*index, &mut pci_id) } == RSMI_STATUS_SUCCESS;
        ok && ((pci_id >> 8) & 0xff) as i32 == bus_id
    })
}

/// Candidate paths for librocm_smi64, most specific first. `TM_ROCM_SMI_PATH` may name the
/// library itself or the ROCm prefix that contains it.
fn candidate_paths() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for variable in ["TM_ROCM_SMI_PATH", "ROCM_PATH"] {
        let Ok(value) = std::env::var(variable) else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let path = Path::new(&value);
        if path.is_file() {
            candidates.push(path.to_path_buf());
            continue;
        }
        candidates.push(path.join("lib").join("librocm_smi64.so"));
        candidates.push(path.join("lib").join("librocm_smi64.so.1"));
    }
    candidates.push(PathBuf::from("librocm_smi64.so.1"));
    candidates.push(PathBuf::from("librocm_smi64.so"));
    candidates
}

fn load_symbols() -> Option<Symbols> {
    for candidate in candidate_paths() {
        let Ok(name) = CString::new(candidate.as_os_str().to_string_lossy().as_bytes()) else {
            continue;
        };
        // SAFETY: `name` is a valid NUL-terminated path; dlopen returns null on failure
        // rather than trapping. Loading a shared library runs its initialisers, which is
        // why only ROCm's own library is ever named here.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if handle.is_null() {
            continue;
        }
        match bind(handle) {
            Some(symbols) => return Some(symbols),
            None => {
                // SAFETY: the handle came from the dlopen just above and is closed once.
                unsafe { libc::dlclose(handle) };
            }
        }
    }
    None
}

/// Resolves every symbol the session needs and runs `rsmi_init`. Returns `None` — leaving
/// the caller to close the handle — if anything is missing or initialisation fails.
fn bind(handle: *mut c_void) -> Option<Symbols> {
    // SAFETY of every transmute below: each symbol is looked up by the exact name whose
    // C signature the matching type alias reproduces, from the ROCm SMI ABI. A missing
    // symbol yields null and is rejected before it is ever called.
    let init: RsmiInit = unsafe { symbol(handle, b"rsmi_init\0")? };
    let shut_down: RsmiShutDown = unsafe { symbol(handle, b"rsmi_shut_down\0")? };
    let num_monitor_devices: RsmiNumMonitorDevices =
        unsafe { symbol(handle, b"rsmi_num_monitor_devices\0")? };
    let pci_id_get: RsmiDevPciIdGet = unsafe { symbol(handle, b"rsmi_dev_pci_id_get\0")? };
    let power_ave_get: RsmiDevPowerAveGet =
        unsafe { symbol(handle, b"rsmi_dev_power_ave_get\0")? };
    let busy_percent_get: RsmiDevBusyPercentGet =
        unsafe { symbol(handle, b"rsmi_dev_busy_percent_get\0")? };

    // SAFETY: `init` is the resolved rsmi_init, which takes an init-flags word.
    if unsafe { init(0) } != RSMI_STATUS_SUCCESS {
        return None;
    }
    Some(Symbols {
        handle,
        shut_down,
        num_monitor_devices,
        pci_id_get,
        power_ave_get,
        busy_percent_get,
    })
}

/// # Safety
/// `T` must be the function-pointer type matching the C signature of `name` in the library
/// behind `handle`.
unsafe fn symbol<T: Copy>(handle: *mut c_void, name: &[u8]) -> Option<T> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    let pointer = libc::dlsym(handle, name.as_ptr().cast::<c_char>());
    if pointer.is_null() {
        return None;
    }
    Some(ptr::read(ptr::addr_of!(pointer).cast::<T>()))
}

#[cfg(test)]
mod tests {
    use super::{candidate_paths, TelemetrySession};

    #[test]
    fn an_unavailable_library_degrades_instead_of_failing() {
        // Whatever this box has, constructing and querying a session must not panic and an
        // unavailable session must report nothing.
        let session = TelemetrySession::new();
        let telemetry = session.query(0, -1);
        if !session.available() {
            assert_eq!(session.source_name(), "none");
            assert_eq!(telemetry.power_milliwatts, None);
            assert_eq!(telemetry.utilization_percent, None);
        }
    }

    #[test]
    fn candidates_always_end_with_the_bare_soname() {
        let candidates = candidate_paths();
        assert!(candidates
            .iter()
            .any(|path| path.as_os_str() == "librocm_smi64.so"));
    }
}
