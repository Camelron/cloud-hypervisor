// Copyright © 2025 Cloud Hypervisor Contributors
//
// SPDX-License-Identifier: Apache-2.0

//! Virtio RTC device implementation.
//!
//! This module implements the virtio-rtc device (device type 44) as specified in
//! the virtio RTC specification. It provides high-resolution clock readings to
//! the guest via a virtqueue-based request/response protocol, enabling
//! PTP-based time synchronization between host and guest.
//!
//! The device exposes two clocks:
//! - Clock 0: UTC (wall-clock time from `CLOCK_REALTIME`)
//! - Clock 1: Monotonic (from `CLOCK_MONOTONIC`)

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

// ── Virtio RTC spec constants ────────────────────────────────────────────────

/// Request type: read a clock
const VIRTIO_RTC_REQ_READ_CLOCK: u16 = 1;

/// Status: success
const VIRTIO_RTC_S_OK: u8 = 0;
/// Status: I/O error
const VIRTIO_RTC_S_IOERR: u8 = 1;
/// Status: unsupported request
const VIRTIO_RTC_S_UNSUPP: u8 = 2;

/// Clock type: UTC wall‑clock (backed by CLOCK_REALTIME)
const VIRTIO_RTC_CLOCK_UTC: u16 = 0;
/// Clock type: monotonic (backed by CLOCK_MONOTONIC)
const VIRTIO_RTC_CLOCK_MONOTONIC: u16 = 1;

/// Number of clocks we expose
const NUM_CLOCKS: u32 = 2;

// ── Virtio feature bits ──────────────────────────────────────────────────────

/// The device can provide UTC time
#[allow(dead_code)]
const VIRTIO_RTC_F_UTC: u64 = 0;

// ── Virtqueue settings ───────────────────────────────────────────────────────

const QUEUE_SIZE: u16 = 64;
const QUEUE_SIZES: &[u16] = &[QUEUE_SIZE];

// Epoll event for new descriptors on the requestq
const QUEUE_AVAIL_EVENT: u16 = EPOLL_HELPER_EVENT_LAST + 1;

// ── Wire‑format structures ───────────────────────────────────────────────────

/// Request header sent by the guest (device‑readable descriptor).
///
/// ```text
/// struct virtio_rtc_req_read_clock {
///     le16 msg_type;     // VIRTIO_RTC_REQ_READ_CLOCK = 1
///     le16 clock_id;     // 0 = UTC, 1 = monotonic
///     le32 flags;        // reserved, must be 0
/// };
/// ```
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcReqReadClock {
    msg_type: Le16,
    clock_id: Le16,
    flags: Le32,
}

// SAFETY: VirtioRtcReqReadClock is a repr(C, packed) struct of plain data fields
// with no padding, pointers, or references — it is safe to transmute from a byte slice.
unsafe impl ByteValued for VirtioRtcReqReadClock {}

/// Response returned to the guest (device‑writable descriptor).
///
/// ```text
/// struct virtio_rtc_resp_read_clock {
///     le64 clock_ns;   // nanoseconds (since epoch for UTC, arbitrary for mono)
///     u8   status;     // VIRTIO_RTC_S_OK, etc.
///     u8   padding[7];
/// };
/// ```
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcRespReadClock {
    clock_ns: Le64,
    status: u8,
    _padding: [u8; 7],
}

// SAFETY: Same rationale as above.
unsafe impl ByteValued for VirtioRtcRespReadClock {}

/// Config space exposed to the guest.
///
/// ```text
/// struct virtio_rtc_config {
///     le32 num_clocks;
/// };
/// ```
#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
struct VirtioRtcConfig {
    num_clocks: Le32,
}

// SAFETY: Same rationale as above.
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
    fn process_queue(&mut self) -> Result<bool, Error> {
        let queue = &mut self.queue;
        let mut used_descs = false;

        while let Some(mut desc_chain) = queue.pop_descriptor_chain(self.mem.memory()) {
            // ── 1. Read the device-readable request descriptor ────────────
            let req_desc = desc_chain.next().ok_or(Error::DescriptorChainTooShort)?;

            if req_desc.is_write_only() {
                return Err(Error::InvalidDescriptor {
                    expected: "readable request",
                });
            }

            let req_addr = req_desc
                .addr()
                .translate_gva(self.access_platform.as_ref(), req_desc.len() as usize);

            let req: VirtioRtcReqReadClock = desc_chain
                .memory()
                .read_obj(req_addr)
                .map_err(Error::GuestMemoryRead)?;

            // ── 2. Build the response ─────────────────────────────────────
            let resp = handle_request(&req);

            // ── 3. Write to the device-writable response descriptor ───────
            let resp_desc = desc_chain.next().ok_or(Error::DescriptorChainTooShort)?;

            if !resp_desc.is_write_only() {
                return Err(Error::InvalidDescriptor {
                    expected: "writable response",
                });
            }

            let resp_addr = resp_desc
                .addr()
                .translate_gva(self.access_platform.as_ref(), resp_desc.len() as usize);

            desc_chain
                .memory()
                .write_obj(resp, resp_addr)
                .map_err(Error::GuestMemoryWrite)?;

            let resp_len = mem::size_of::<VirtioRtcRespReadClock>() as u32;
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

/// Dispatch a single request and return the appropriate response.
fn handle_request(req: &VirtioRtcReqReadClock) -> VirtioRtcRespReadClock {
    let msg_type: u16 = req.msg_type.into();
    let clock_id: u16 = req.clock_id.into();

    if msg_type != VIRTIO_RTC_REQ_READ_CLOCK {
        warn!("virtio-rtc: unsupported request type {msg_type}");
        return VirtioRtcRespReadClock {
            clock_ns: Le64::from(0u64),
            status: VIRTIO_RTC_S_UNSUPP,
            _padding: [0u8; 7],
        };
    }

    let clockid = match clock_id {
        VIRTIO_RTC_CLOCK_UTC => libc::CLOCK_REALTIME,
        VIRTIO_RTC_CLOCK_MONOTONIC => libc::CLOCK_MONOTONIC,
        _ => {
            warn!("virtio-rtc: unsupported clock_id {clock_id}");
            return VirtioRtcRespReadClock {
                clock_ns: Le64::from(0u64),
                status: VIRTIO_RTC_S_UNSUPP,
                _padding: [0u8; 7],
            };
        }
    };

    match clock_gettime_ns(clockid) {
        Ok(ns) => VirtioRtcRespReadClock {
            clock_ns: Le64::from(ns),
            status: VIRTIO_RTC_S_OK,
            _padding: [0u8; 7],
        },
        Err(()) => {
            error!("virtio-rtc: clock_gettime failed for clock_id {clock_id}");
            VirtioRtcRespReadClock {
                clock_ns: Le64::from(0u64),
                status: VIRTIO_RTC_S_IOERR,
                _padding: [0u8; 7],
            }
        }
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
            let mut avail_features = 1u64 << VIRTIO_F_VERSION_1 | 1u64 << VIRTIO_RTC_F_UTC;

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
            num_clocks: Le32::from(NUM_CLOCKS),
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
