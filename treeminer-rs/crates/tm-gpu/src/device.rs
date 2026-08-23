//! Device enumeration and selection. Port of `src/CudaDevice.cpp` and the device half of
//! `src/CudaBackend.cpp`.

use crate::error::{GpuError, Result};
use crate::driver;

/// One GPU, identified the way the dashboard and the telemetry session need it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    index: i32,
    name: String,
    pci_bus_id: String,
    bus_id: i32,
    total_memory_bytes: usize,
}

impl Device {
    /// Opens one device by its runtime index.
    pub fn open(index: i32) -> Result<Self> {
        let count = driver::device_count()?;
        if index < 0 || index >= count {
            return Err(GpuError::NoSuchDevice(index));
        }
        let pci_bus_id = driver::device_pci_bus_id(index)?;
        Ok(Self {
            index,
            name: driver::device_name(index)?,
            bus_id: parse_bus_id(&pci_bus_id).unwrap_or(-1),
            pci_bus_id,
            total_memory_bytes: driver::device_total_memory(index)?,
        })
    }

    /// Every device the runtime reports. An absent or broken driver yields an error, not a
    /// panic, so the caller can fall back to CPU mining.
    pub fn enumerate() -> Result<Vec<Self>> {
        let count = driver::device_count()?;
        (0..count).map(Self::open).collect()
    }

    pub fn index(&self) -> i32 {
        self.index
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full `0000:03:00.0` form.
    pub fn pci_bus_id(&self) -> &str {
        &self.pci_bus_id
    }

    /// Just the bus byte, which is what ROCm SMI matches devices on; -1 when unknown.
    pub fn bus_id(&self) -> i32 {
        self.bus_id
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.total_memory_bytes
    }

    /// `"<name> | <n> GB"`, the string the C++ miner logs.
    pub fn full_name(&self) -> String {
        let gigabytes =
            (self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)).round() as i64;
        format!("{} | {} GB", self.name, gigabytes)
    }

    /// Binds this device to the calling thread. Must be called before any allocation on it.
    pub fn activate(&self) -> Result<()> {
        driver::set_device(self.index)
    }

    /// Free VRAM on this device. Requires the device to be active on this thread, so it is
    /// activated first — the free-memory query always reports the current device.
    pub fn free_memory_bytes(&self) -> Result<usize> {
        self.activate()?;
        let (free, _total) = driver::mem_get_info()?;
        Ok(free)
    }
}

/// `0000:03:00.0` -> 3. Anything else yields `None`.
fn parse_bus_id(pci_bus_id: &str) -> Option<i32> {
    let mut fields = pci_bus_id.split(':');
    let _domain = fields.next()?;
    let bus = fields.next()?;
    i32::from_str_radix(bus, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::parse_bus_id;

    #[test]
    fn parses_the_bus_byte_out_of_a_pci_id() {
        assert_eq!(parse_bus_id("0000:03:00.0"), Some(3));
        assert_eq!(parse_bus_id("0000:C1:00.0"), Some(0xc1));
        assert_eq!(parse_bus_id("garbage"), None);
    }
}
