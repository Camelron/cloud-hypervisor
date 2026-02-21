// Copyright © 2025 Cloud Hypervisor Contributors
//
// SPDX-License-Identifier: Apache-2.0

//! Virtio RTC device implementation.
//!
//! This module implements the virtio-rtc device (device type 44) matching the
//! upstream Linux `virtio_rtc.h` UAPI and `virtio_rtc_driver.c` wire protocol.
//! It provides high-resolution clock readings to the guest via a virtqueue-based
//! request/response protocol, enabling PTP-based time synchronization.
//!
//! The device exposes two clocks:
//! - Clock 0: UTC wall-clock (backed by `CLOCK_REALTIME`)
//! - Clock 1: Monotonic (backed by `CLOCK_MONOTONIC`)
//!
//! On x86_64, cross-timestamping with the TSC is supported for both clocks,
//! enabling high-accuracy PTP synchronization similar to KVM PTP.

use std::io;
use std::mem;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Barrier};

use anyhow::anyhow;
use seccompiler::SeccompAction;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use virtio_queue::{Queue, QueueT};
use vm_memory::{ByteValued, Bytes, GuestAddressSpace, GuestMemoryAtomic, Le16, Le32, Le64};
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};
use vm_virtio::{AccessPlatform, Translatable};
use vmm_sys_util::eventfd::EventFd;

use super::{
    ActivateResult, EPOLL_HELPER_EVENT_LAST, EpollHelper, EpollHelperError, EpollHelperHandler,
    Error as DeviceError, VIRTIO_F_IOMMU_PLATFORM, VIRTIO_F_VERSION_1, VirtioCommon, VirtioDevice,
    VirtioDeviceType,
};
use crate::seccomp_filters::Thread;
use crate::thread_helper::spawn_virtio_thread;
use crate::{GuestMemoryMmap, VirtioInterrupt, VirtioInterruptType};

// ── Virtio RTC spec constants (matching linux/virtio_rtc.h UAPI) ─────────────

// Read request message types
const VIRTIO_RTC_REQ_READ: u16 = 0x0001;
const VIRTIO_RTC_REQ_READ_CROSS: u16 = 0x0002;

// Control request message types
const VIRTIO_RTC_REQ_CFG: u16 = 0x1000;
const VIRTIO_RTC_REQ_CLOCK_CAP: u16 = 0x1001;
const VIRTIO_RTC_REQ_CROSS_CAP: u16 = 0x1002;

// Status codes (struct virtio_rtc_resp_head)
const VIRTIO_RTC_S_OK: u8 = 0;
const VIRTIO_RTC_S_EOPNOTSUPP: u8 = 2;
const VIRTIO_RTC_S_ENODEV: u8 = 3;
#[allow(dead_code)]
const VIRTIO_RTC_S_EINVAL: u8 = 4;
const VIRTIO_RTC_S_EIO: u8 = 5;

// Clock types (struct virtio_rtc_resp_clock_cap.type)
const VIRTIO_RTC_CLOCK_UTC: u8 = 0;
#[allow(dead_code)]
const VIRTIO_RTC_CLOCK_TAI: u8 = 1;
const VIRTIO_RTC_CLOCK_MONOTONIC: u8 = 2;

// HW counter types
#[cfg(target_arch = "x86_64")]
const VIRTIO_RTC_COUNTER_X86_TSC: u8 = 1;

// Cross-cap flags
const VIRTIO_RTC_FLAG_CROSS_CAP: u8 = 1 << 0;

// Number of clocks we expose
const NUM_CLOCKS: u16 = 2;

// ── Virtqueue settings ───────────────────────────────────────────────────────

const QUEUE_SIZE: u16 = 64;
const QUEUE_SIZES: &[u16] = &[QUEUE_SIZE];

// Epoll event for new descriptors on the requestq
const QUEUE_AVAIL_EVENT: u16 = EPOLL_HELPER_EVENT_LAST + 1;

// ── Wire-format structures (matching linux/virtio_rtc.h exactly) ─────────────

/// Common request header (8 bytes).
///
/// ```text
/// struct virtio_rtc_req_head {
///     __le16 msg_type;
///     __u8   reserved[6];
/// };
/// ```
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcReqHead {
    msg_type: Le16,
    _reserved: [u8; 6],
}

// SAFETY: repr(C, packed) struct of plain data fields with no padding/pointers.
unsafe impl ByteValued for VirtioRtcReqHead {}

/// Common response header (8 bytes).
///
/// ```text
/// struct virtio_rtc_resp_head {
///     __u8 status;
///     __u8 reserved[7];
/// };
/// ```
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcRespHead {
    status: u8,
    _reserved: [u8; 7],
}

unsafe impl ByteValued for VirtioRtcRespHead {}

// ── VIRTIO_RTC_REQ_READ (0x0001) ─────────────────────────────────────────────

/// REQ_READ request (16 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcReqRead {
    head: VirtioRtcReqHead,
    clock_id: Le16,
    _reserved: [u8; 6],
}

unsafe impl ByteValued for VirtioRtcReqRead {}

/// REQ_READ response (16 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcRespRead {
    head: VirtioRtcRespHead,
    clock_reading: Le64,
}

unsafe impl ByteValued for VirtioRtcRespRead {}

// ── VIRTIO_RTC_REQ_READ_CROSS (0x0002) ───────────────────────────────────────

/// REQ_READ_CROSS request (16 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcReqReadCross {
    head: VirtioRtcReqHead,
    clock_id: Le16,
    hw_counter: u8,
    _reserved: [u8; 5],
}

unsafe impl ByteValued for VirtioRtcReqReadCross {}

/// REQ_READ_CROSS response (24 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcRespReadCross {
    head: VirtioRtcRespHead,
    clock_reading: Le64,
    counter_cycles: Le64,
}

unsafe impl ByteValued for VirtioRtcRespReadCross {}

// ── VIRTIO_RTC_REQ_CFG (0x1000) ──────────────────────────────────────────────

// REQ_CFG request is just the 8-byte header (no extra fields).

/// REQ_CFG response (16 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcRespCfg {
    head: VirtioRtcRespHead,
    num_clocks: Le16,
    _reserved: [u8; 6],
}

unsafe impl ByteValued for VirtioRtcRespCfg {}

// ── VIRTIO_RTC_REQ_CLOCK_CAP (0x1001) ────────────────────────────────────────

/// REQ_CLOCK_CAP request (16 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcReqClockCap {
    head: VirtioRtcReqHead,
    clock_id: Le16,
    _reserved: [u8; 6],
}

unsafe impl ByteValued for VirtioRtcReqClockCap {}

/// REQ_CLOCK_CAP response (16 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcRespClockCap {
    head: VirtioRtcRespHead,
    clock_type: u8,
    leap_second_smearing: u8,
    flags: u8,
    _reserved: [u8; 5],
}

unsafe impl ByteValued for VirtioRtcRespClockCap {}

// ── VIRTIO_RTC_REQ_CROSS_CAP (0x1002) ────────────────────────────────────────

/// REQ_CROSS_CAP request (16 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcReqCrossCap {
    head: VirtioRtcReqHead,
    clock_id: Le16,
    hw_counter: u8,
    _reserved: [u8; 5],
}

unsafe impl ByteValued for VirtioRtcReqCrossCap {}

/// REQ_CROSS_CAP response (16 bytes).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcRespCrossCap {
    head: VirtioRtcRespHead,
    flags: u8,
    _reserved: [u8; 7],
}

unsafe impl ByteValued for VirtioRtcRespCrossCap {}

// ── Config space ─────────────────────────────────────────────────────────────

/// Config space exposed to the guest (optional; the driver uses REQ_CFG instead).
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcConfig {
    num_clocks: Le32,
}

unsafe impl ByteValued for VirtioRtcConfig {}

// ── Helper: read host clock ──────────────────────────────────────────────────

/// Read the given POSIX clock and return nanoseconds since epoch / boot.
fn clock_gettime_ns(clockid: libc::clockid_t) -> Result<u64, ()> {
    let mut ts: libc::timespec = unsafe { mem::zeroed() };
    // SAFETY: ts is a valid pointer to a timespec struct on the stack.
    let ret = unsafe { libc::clock_gettime(clockid, &mut ts) };
    if ret != 0 {
        return Err(());
    }
    Ok(ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
}

/// Read the given POSIX clock and simultaneously capture the x86 TSC.
/// Returns `(clock_ns, tsc_cycles)`. The TSC value is the midpoint of two
/// reads sandwiching the `clock_gettime` call for best accuracy.
#[cfg(target_arch = "x86_64")]
fn clock_gettime_with_tsc(clockid: libc::clockid_t) -> Result<(u64, u64), ()> {
    // SAFETY: _rdtsc reads the timestamp counter; always safe on x86_64.
    let tsc1 = unsafe { std::arch::x86_64::_rdtsc() };
    let ns = clock_gettime_ns(clockid)?;
    let tsc2 = unsafe { std::arch::x86_64::_rdtsc() };
    Ok((ns, tsc1.wrapping_add(tsc2) / 2))
}

/// Map a virtio-rtc clock_id to a POSIX clockid.
fn clock_id_to_posix(clock_id: u16) -> Option<libc::clockid_t> {
    match clock_id {
        0 => Some(libc::CLOCK_REALTIME),  // Clock 0 = UTC
        1 => Some(libc::CLOCK_MONOTONIC), // Clock 1 = Monotonic
        _ => None,
    }
}

/// Return the virtio-rtc clock type constant for a given clock_id.
fn clock_id_to_type(clock_id: u16) -> Option<u8> {
    match clock_id {
        0 => Some(VIRTIO_RTC_CLOCK_UTC),
        1 => Some(VIRTIO_RTC_CLOCK_MONOTONIC),
        _ => None,
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
enum Error {
    #[error("Descriptor chain too short")]
    DescriptorChainTooShort,
    #[error("Failed to read request from guest memory")]
    GuestMemoryRead(#[source] vm_memory::guest_memory::Error),
    #[error("Failed to write response to guest memory")]
    GuestMemoryWrite(#[source] vm_memory::guest_memory::Error),
    #[error("Failed adding used index")]
    QueueAddUsed(#[source] virtio_queue::Error),
    #[error("Invalid descriptor (expected {expected})")]
    InvalidDescriptor { expected: &'static str },
}

// ── Response builders ────────────────────────────────────────────────────────

fn resp_head(status: u8) -> VirtioRtcRespHead {
    VirtioRtcRespHead {
        status,
        _reserved: [0u8; 7],
    }
}

fn handle_cfg() -> VirtioRtcRespCfg {
    VirtioRtcRespCfg {
        head: resp_head(VIRTIO_RTC_S_OK),
        num_clocks: Le16::from(NUM_CLOCKS),
        _reserved: [0u8; 6],
    }
}

fn handle_clock_cap(clock_id: u16) -> VirtioRtcRespClockCap {
    match clock_id_to_type(clock_id) {
        Some(clock_type) => VirtioRtcRespClockCap {
            head: resp_head(VIRTIO_RTC_S_OK),
            clock_type,
            leap_second_smearing: 0, // VIRTIO_RTC_SMEAR_UNSPECIFIED
            flags: 0,                // no alarm capability
            _reserved: [0u8; 5],
        },
        None => VirtioRtcRespClockCap {
            head: resp_head(VIRTIO_RTC_S_ENODEV),
            clock_type: 0,
            leap_second_smearing: 0,
            flags: 0,
            _reserved: [0u8; 5],
        },
    }
}

#[cfg(target_arch = "x86_64")]
fn handle_cross_cap(clock_id: u16, hw_counter: u8) -> VirtioRtcRespCrossCap {
    let supported =
        clock_id_to_posix(clock_id).is_some() && hw_counter == VIRTIO_RTC_COUNTER_X86_TSC;
    VirtioRtcRespCrossCap {
        head: resp_head(VIRTIO_RTC_S_OK),
        flags: if supported {
            VIRTIO_RTC_FLAG_CROSS_CAP
        } else {
            0
        },
        _reserved: [0u8; 7],
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn handle_cross_cap(_clock_id: u16, _hw_counter: u8) -> VirtioRtcRespCrossCap {
    VirtioRtcRespCrossCap {
        head: resp_head(VIRTIO_RTC_S_OK),
        flags: 0,
        _reserved: [0u8; 7],
    }
}

fn handle_read(clock_id: u16) -> VirtioRtcRespRead {
    let clockid = match clock_id_to_posix(clock_id) {
        Some(c) => c,
        None => {
            return VirtioRtcRespRead {
                head: resp_head(VIRTIO_RTC_S_ENODEV),
                clock_reading: Le64::from(0u64),
            };
        }
    };
    match clock_gettime_ns(clockid) {
        Ok(ns) => VirtioRtcRespRead {
            head: resp_head(VIRTIO_RTC_S_OK),
            clock_reading: Le64::from(ns),
        },
        Err(()) => {
            error!("virtio-rtc: clock_gettime failed for clock_id {clock_id}");
            VirtioRtcRespRead {
                head: resp_head(VIRTIO_RTC_S_EIO),
                clock_reading: Le64::from(0u64),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn handle_read_cross(clock_id: u16, hw_counter: u8) -> VirtioRtcRespReadCross {
    if hw_counter != VIRTIO_RTC_COUNTER_X86_TSC {
        return VirtioRtcRespReadCross {
            head: resp_head(VIRTIO_RTC_S_EOPNOTSUPP),
            clock_reading: Le64::from(0u64),
            counter_cycles: Le64::from(0u64),
        };
    }
    let clockid = match clock_id_to_posix(clock_id) {
        Some(c) => c,
        None => {
            return VirtioRtcRespReadCross {
                head: resp_head(VIRTIO_RTC_S_ENODEV),
                clock_reading: Le64::from(0u64),
                counter_cycles: Le64::from(0u64),
            };
        }
    };
    match clock_gettime_with_tsc(clockid) {
        Ok((ns, tsc)) => VirtioRtcRespReadCross {
            head: resp_head(VIRTIO_RTC_S_OK),
            clock_reading: Le64::from(ns),
            counter_cycles: Le64::from(tsc),
        },
        Err(()) => {
            error!("virtio-rtc: clock_gettime failed for READ_CROSS clock_id {clock_id}");
            VirtioRtcRespReadCross {
                head: resp_head(VIRTIO_RTC_S_EIO),
                clock_reading: Le64::from(0u64),
                counter_cycles: Le64::from(0u64),
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn handle_read_cross(_clock_id: u16, _hw_counter: u8) -> VirtioRtcRespReadCross {
    VirtioRtcRespReadCross {
        head: resp_head(VIRTIO_RTC_S_EOPNOTSUPP),
        clock_reading: Le64::from(0u64),
        counter_cycles: Le64::from(0u64),
    }
}

// ── Epoll handler ────────────────────────────────────────────────────────────

struct RtcEpollHandler {
    mem: GuestMemoryAtomic<GuestMemoryMmap>,
    queue: Queue,
    interrupt_cb: Arc<dyn VirtioInterrupt>,
    queue_evt: EventFd,
    kill_evt: EventFd,
    pause_evt: EventFd,
    access_platform: Option<Arc<dyn AccessPlatform>>,
}

impl RtcEpollHandler {
    /// Process all available descriptors on the requestq.
    ///
    /// Each descriptor chain has:
    ///   - One device-readable descriptor containing the request
    ///   - One device-writable descriptor for the response
    ///
    /// We read the 8-byte request header to determine `msg_type`, then dispatch
    /// to the appropriate handler which reads any additional request fields and
    /// builds the correctly-typed response.
    fn process_queue(&mut self) -> Result<bool, Error> {
        let queue = &mut self.queue;
        let mut used_descs = false;

        while let Some(mut desc_chain) = queue.pop_descriptor_chain(self.mem.memory()) {
            // ── 1. Get the device-readable request descriptor ─────────────
            let req_desc = desc_chain.next().ok_or(Error::DescriptorChainTooShort)?;

            if req_desc.is_write_only() {
                return Err(Error::InvalidDescriptor {
                    expected: "readable request",
                });
            }

            let req_addr = req_desc
                .addr()
                .translate_gva(self.access_platform.as_ref(), req_desc.len() as usize);

            // ── 2. Get the device-writable response descriptor ────────────
            let resp_desc = desc_chain.next().ok_or(Error::DescriptorChainTooShort)?;

            if !resp_desc.is_write_only() {
                return Err(Error::InvalidDescriptor {
                    expected: "writable response",
                });
            }

            let resp_addr = resp_desc
                .addr()
                .translate_gva(self.access_platform.as_ref(), resp_desc.len() as usize);

            let memory = desc_chain.memory();

            // ── 3. Read request header to determine message type ──────────
            let req_head: VirtioRtcReqHead = memory
                .read_obj(req_addr)
                .map_err(Error::GuestMemoryRead)?;
            let msg_type: u16 = req_head.msg_type.into();

            // ── 4. Dispatch on message type ───────────────────────────────
            let resp_len: u32 = match msg_type {
                VIRTIO_RTC_REQ_CFG => {
                    let resp = handle_cfg();
                    memory
                        .write_obj(resp, resp_addr)
                        .map_err(Error::GuestMemoryWrite)?;
                    mem::size_of::<VirtioRtcRespCfg>() as u32
                }

                VIRTIO_RTC_REQ_CLOCK_CAP => {
                    let req: VirtioRtcReqClockCap = memory
                        .read_obj(req_addr)
                        .map_err(Error::GuestMemoryRead)?;
                    let resp = handle_clock_cap(req.clock_id.into());
                    memory
                        .write_obj(resp, resp_addr)
                        .map_err(Error::GuestMemoryWrite)?;
                    mem::size_of::<VirtioRtcRespClockCap>() as u32
                }

                VIRTIO_RTC_REQ_CROSS_CAP => {
                    let req: VirtioRtcReqCrossCap = memory
                        .read_obj(req_addr)
                        .map_err(Error::GuestMemoryRead)?;
                    let resp = handle_cross_cap(req.clock_id.into(), req.hw_counter);
                    memory
                        .write_obj(resp, resp_addr)
                        .map_err(Error::GuestMemoryWrite)?;
                    mem::size_of::<VirtioRtcRespCrossCap>() as u32
                }

                VIRTIO_RTC_REQ_READ => {
                    let req: VirtioRtcReqRead = memory
                        .read_obj(req_addr)
                        .map_err(Error::GuestMemoryRead)?;
                    let resp = handle_read(req.clock_id.into());
                    memory
                        .write_obj(resp, resp_addr)
                        .map_err(Error::GuestMemoryWrite)?;
                    mem::size_of::<VirtioRtcRespRead>() as u32
                }

                VIRTIO_RTC_REQ_READ_CROSS => {
                    let req: VirtioRtcReqReadCross = memory
                        .read_obj(req_addr)
                        .map_err(Error::GuestMemoryRead)?;
                    let resp = handle_read_cross(req.clock_id.into(), req.hw_counter);
                    memory
                        .write_obj(resp, resp_addr)
                        .map_err(Error::GuestMemoryWrite)?;
                    mem::size_of::<VirtioRtcRespReadCross>() as u32
                }

                _ => {
                    warn!("virtio-rtc: unsupported message type 0x{msg_type:04x}");
                    let resp = resp_head(VIRTIO_RTC_S_EOPNOTSUPP);
                    memory
                        .write_obj(resp, resp_addr)
                        .map_err(Error::GuestMemoryWrite)?;
                    mem::size_of::<VirtioRtcRespHead>() as u32
                }
            };

            queue
                .add_used(desc_chain.memory(), desc_chain.head_index(), resp_len)
                .map_err(Error::QueueAddUsed)?;

            used_descs = true;
        }

        Ok(used_descs)
    }

    fn signal_used_queue(&self) -> Result<(), DeviceError> {
        self.interrupt_cb
            .trigger(VirtioInterruptType::Queue(0))
            .map_err(|e| {
                error!("Failed to signal used queue: {:?}", e);
                DeviceError::FailedSignalingUsedQueue(e)
            })
    }

    fn run(
        &mut self,
        paused: Arc<AtomicBool>,
        paused_sync: Arc<Barrier>,
    ) -> Result<(), EpollHelperError> {
        let mut helper = EpollHelper::new(&self.kill_evt, &self.pause_evt)?;
        helper.add_event(self.queue_evt.as_raw_fd(), QUEUE_AVAIL_EVENT)?;
        helper.run(paused, paused_sync, self)?;
        Ok(())
    }
}

impl EpollHelperHandler for RtcEpollHandler {
    fn handle_event(
        &mut self,
        _helper: &mut EpollHelper,
        event: &epoll::Event,
    ) -> Result<(), EpollHelperError> {
        let ev_type = event.data as u16;
        match ev_type {
            QUEUE_AVAIL_EVENT => {
                self.queue_evt.read().map_err(|e| {
                    EpollHelperError::HandleEvent(anyhow!("Failed to get queue event: {:?}", e))
                })?;
                let needs_notification = self.process_queue().map_err(|e| {
                    EpollHelperError::HandleEvent(anyhow!("Failed to process queue: {:?}", e))
                })?;
                if needs_notification {
                    self.signal_used_queue().map_err(|e| {
                        EpollHelperError::HandleEvent(anyhow!(
                            "Failed to signal used queue: {:?}",
                            e
                        ))
                    })?;
                }
            }
            _ => {
                return Err(EpollHelperError::HandleEvent(anyhow!(
                    "Unexpected event: {}",
                    ev_type
                )));
            }
        }
        Ok(())
    }
}

// ── Public device struct ─────────────────────────────────────────────────────

/// Virtio RTC device exposing high-resolution host clocks to the guest.
pub struct Rtc {
    common: VirtioCommon,
    id: String,
    seccomp_action: SeccompAction,
    exit_evt: EventFd,
}

#[derive(Deserialize, Serialize)]
pub struct RtcState {
    pub avail_features: u64,
    pub acked_features: u64,
}

impl Rtc {
    /// Create a new virtio-rtc device.
    pub fn new(
        id: String,
        iommu: bool,
        seccomp_action: SeccompAction,
        exit_evt: EventFd,
        state: Option<RtcState>,
    ) -> io::Result<Rtc> {
        let (avail_features, acked_features, paused) = if let Some(state) = state {
            info!("Restoring virtio-rtc {}", id);
            (state.avail_features, state.acked_features, true)
        } else {
            // Do NOT set bit 0 — upstream defines that as VIRTIO_RTC_F_ALARM,
            // and we do not support alarms. Only advertise VIRTIO_F_VERSION_1.
            let mut avail_features = 1u64 << VIRTIO_F_VERSION_1;

            if iommu {
                avail_features |= 1u64 << VIRTIO_F_IOMMU_PLATFORM;
            }

            (avail_features, 0, false)
        };

        Ok(Rtc {
            common: VirtioCommon {
                device_type: VirtioDeviceType::Rtc as u32,
                queue_sizes: QUEUE_SIZES.to_vec(),
                paused_sync: Some(Arc::new(Barrier::new(2))),
                avail_features,
                acked_features,
                min_queues: 1,
                paused: Arc::new(AtomicBool::new(paused)),
                ..Default::default()
            },
            id,
            seccomp_action,
            exit_evt,
        })
    }

    fn state(&self) -> RtcState {
        RtcState {
            avail_features: self.common.avail_features,
            acked_features: self.common.acked_features,
        }
    }

    #[cfg(fuzzing)]
    pub fn wait_for_epoll_threads(&mut self) {
        self.common.wait_for_epoll_threads();
    }
}

impl Drop for Rtc {
    fn drop(&mut self) {
        if let Some(kill_evt) = self.common.kill_evt.take() {
            let _ = kill_evt.write(1);
        }
        self.common.wait_for_epoll_threads();
    }
}

impl VirtioDevice for Rtc {
    fn device_type(&self) -> u32 {
        self.common.device_type
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &self.common.queue_sizes
    }

    fn features(&self) -> u64 {
        self.common.avail_features
    }

    fn ack_features(&mut self, value: u64) {
        self.common.ack_features(value)
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        let config = VirtioRtcConfig {
            num_clocks: Le32::from(NUM_CLOCKS as u32),
        };
        let config_bytes = config.as_slice();
        let config_len = config_bytes.len() as u64;

        if offset >= config_len {
            error!("virtio-rtc: config read past end (offset={offset})");
            return;
        }

        let end = std::cmp::min(offset + data.len() as u64, config_len) as usize;
        let src = &config_bytes[offset as usize..end];
        data[..src.len()].copy_from_slice(src);
    }

    fn activate(
        &mut self,
        mem: GuestMemoryAtomic<GuestMemoryMmap>,
        interrupt_cb: Arc<dyn VirtioInterrupt>,
        mut queues: Vec<(usize, Queue, EventFd)>,
    ) -> ActivateResult {
        self.common.activate(&queues, &interrupt_cb)?;
        let (kill_evt, pause_evt) = self.common.dup_eventfds();

        let (_, queue, queue_evt) = queues.remove(0);

        let mut handler = RtcEpollHandler {
            mem,
            queue,
            interrupt_cb,
            queue_evt,
            kill_evt,
            pause_evt,
            access_platform: self.common.access_platform.clone(),
        };

        let paused = self.common.paused.clone();
        let paused_sync = self.common.paused_sync.clone();
        let mut epoll_threads = Vec::new();
        spawn_virtio_thread(
            &self.id,
            &self.seccomp_action,
            Thread::VirtioRtc,
            &mut epoll_threads,
            &self.exit_evt,
            move || handler.run(paused, paused_sync.unwrap()),
        )?;

        self.common.epoll_threads = Some(epoll_threads);

        event!("virtio-device", "activated", "id", &self.id);
        Ok(())
    }

    fn reset(&mut self) -> Option<Arc<dyn VirtioInterrupt>> {
        let result = self.common.reset();
        event!("virtio-device", "reset", "id", &self.id);
        result
    }

    fn set_access_platform(&mut self, access_platform: Arc<dyn AccessPlatform>) {
        self.common.set_access_platform(access_platform)
    }
}

impl Pausable for Rtc {
    fn pause(&mut self) -> Result<(), MigratableError> {
        self.common.pause()
    }

    fn resume(&mut self) -> Result<(), MigratableError> {
        self.common.resume()
    }
}

impl Snapshottable for Rtc {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn snapshot(&mut self) -> Result<Snapshot, MigratableError> {
        Snapshot::new_from_state(&self.state())
    }
}

impl Transportable for Rtc {}
impl Migratable for Rtc {}
