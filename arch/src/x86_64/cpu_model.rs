// SPDX-License-Identifier: Apache-2.0

//! Stable guest CPUID profiles for named CPU models.
//!
//! Model names and architectural feature sets are based on the corresponding
//! libvirt x86 CPU maps for QEMU models. They are translated into CPUID masks
//! here and into MSHV partition feature masks in `hypervisor::mshv`. Keep both
//! representations synchronized when changing a profile. These profiles are
//! not a bit-for-bit QEMU ABI without versioned differential test coverage.

use hypervisor::arch::x86::{
    CPUID_FLAG_EXACT_EAX, CPUID_FLAG_EXACT_EBX, CPUID_FLAG_EXACT_ECX, CPUID_FLAG_EXACT_EDX,
    CpuIdEntry,
};
use hypervisor::{CpuModel, CpuVendor};

use super::Error;

macro_rules! bits {
    ($($bit:expr),* $(,)?) => {
        0_u32 $(| (1_u32 << $bit))*
    };
}

const LEAF_1_EDX: u32 = bits!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 23, 24, 25, 26
);
const INTEL_LEAF_1_ECX: u32 = bits!(
    0, 1, 9, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31
);
const AMD_LEAF_1_ECX: u32 = bits!(0, 1, 9, 12, 13, 19, 20, 21, 22, 23, 25, 26, 28, 29, 30, 31);

const INTEL_EXTENDED_ECX: u32 = bits!(0, 5, 8);
const INTEL_EXTENDED_EDX: u32 = bits!(11, 20, 26, 27, 29);
const AMD_EXTENDED_EDX: u32 = bits!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 29
);

#[derive(Clone, Copy)]
struct CpuModelProfile {
    vendor: CpuVendor,
    signature: u32,
    max_extended_leaf: u32,
    brand: &'static str,
    leaf_1_ecx: u32,
    leaf_7_ebx: u32,
    leaf_7_ecx: u32,
    leaf_7_edx: u32,
    extended_1_ecx: u32,
    extended_1_edx: u32,
    extended_8_ebx: u32,
    extended_a_edx: u32,
    xcr0: u32,
    xsave_1_eax: u32,
}

impl CpuModelProfile {
    fn for_model(model: CpuModel, nested: bool) -> Self {
        let intel_leaf_1_ecx = INTEL_LEAF_1_ECX | if nested { bits!(5) } else { 0 };
        let amd_leaf_1_ecx = AMD_LEAF_1_ECX
            | if matches!(model, CpuModel::EpycMilan) {
                bits!(17)
            } else {
                0
            };
        let amd_extended_1_ecx = bits!(0, 4, 5, 6, 7, 8, 9) | if nested { bits!(2) } else { 0 };

        match model {
            CpuModel::SkylakeServerIbrs => {
                let mut profile = Self::for_model(CpuModel::SkylakeServer, nested);
                profile.leaf_7_edx = bits!(26);
                profile
            }
            CpuModel::SkylakeServerNoTsxIbrs => {
                let mut profile = Self::for_model(CpuModel::SkylakeServerIbrs, nested);
                profile.leaf_7_ebx &= !bits!(4, 11);
                profile
            }
            CpuModel::CascadelakeServerNoTsx => {
                let mut profile = Self::for_model(CpuModel::CascadelakeServer, nested);
                profile.leaf_7_ebx &= !bits!(4, 11);
                profile
            }
            CpuModel::IcelakeServerNoTsx => {
                let mut profile = Self::for_model(CpuModel::IcelakeServer, nested);
                profile.leaf_7_ebx &= !bits!(4, 11);
                profile
            }
            CpuModel::SkylakeServer => Self {
                vendor: CpuVendor::Intel,
                signature: 0x0005_0654,
                max_extended_leaf: 0x8000_0008,
                brand: "Intel Xeon Processor (Skylake)",
                leaf_1_ecx: intel_leaf_1_ecx,
                leaf_7_ebx: bits!(
                    0, 3, 4, 5, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 24, 28, 30, 31
                ),
                leaf_7_ecx: 0,
                leaf_7_edx: 0,
                extended_1_ecx: INTEL_EXTENDED_ECX,
                extended_1_edx: INTEL_EXTENDED_EDX,
                extended_8_ebx: 0,
                extended_a_edx: 0,
                xcr0: bits!(0, 1, 2, 5, 6, 7),
                xsave_1_eax: bits!(0, 1, 2),
            },
            CpuModel::CascadelakeServer => Self {
                vendor: CpuVendor::Intel,
                signature: 0x0005_0657,
                max_extended_leaf: 0x8000_0008,
                brand: "Intel Xeon Processor (Cascade Lake)",
                leaf_1_ecx: intel_leaf_1_ecx,
                leaf_7_ebx: bits!(
                    0, 3, 4, 5, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 23, 24, 28, 30, 31
                ),
                leaf_7_ecx: bits!(11),
                leaf_7_edx: bits!(26, 31),
                extended_1_ecx: INTEL_EXTENDED_ECX,
                extended_1_edx: INTEL_EXTENDED_EDX,
                extended_8_ebx: 0,
                extended_a_edx: 0,
                xcr0: bits!(0, 1, 2, 5, 6, 7),
                xsave_1_eax: bits!(0, 1, 2),
            },
            CpuModel::IcelakeServer => Self {
                vendor: CpuVendor::Intel,
                signature: 0x0006_06a6,
                max_extended_leaf: 0x8000_0008,
                brand: "Intel Xeon Processor (Ice Lake)",
                leaf_1_ecx: intel_leaf_1_ecx,
                leaf_7_ebx: bits!(
                    0, 3, 4, 5, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 23, 24, 28, 30, 31
                ),
                leaf_7_ecx: bits!(1, 2, 3, 6, 8, 9, 10, 11, 12, 14, 16),
                leaf_7_edx: bits!(26, 31),
                extended_1_ecx: INTEL_EXTENDED_ECX,
                extended_1_edx: INTEL_EXTENDED_EDX,
                extended_8_ebx: bits!(9),
                extended_a_edx: 0,
                xcr0: bits!(0, 1, 2, 5, 6, 7, 9),
                xsave_1_eax: bits!(0, 1, 2),
            },
            CpuModel::Epyc => Self {
                vendor: CpuVendor::AMD,
                signature: 0x0080_0f12,
                max_extended_leaf: 0x8000_001e,
                brand: "AMD EPYC Processor",
                leaf_1_ecx: amd_leaf_1_ecx,
                leaf_7_ebx: bits!(0, 3, 5, 7, 8, 18, 19, 20, 23, 29),
                leaf_7_ecx: 0,
                leaf_7_edx: 0,
                extended_1_ecx: amd_extended_1_ecx,
                extended_1_edx: AMD_EXTENDED_EDX,
                extended_8_ebx: 0,
                extended_a_edx: 0,
                xcr0: bits!(0, 1, 2),
                xsave_1_eax: bits!(0, 1, 2),
            },
            CpuModel::EpycRome => Self {
                vendor: CpuVendor::AMD,
                signature: 0x0083_0f10,
                max_extended_leaf: 0x8000_001e,
                brand: "AMD EPYC Rome Processor",
                leaf_1_ecx: amd_leaf_1_ecx,
                leaf_7_ebx: bits!(0, 3, 5, 7, 8, 18, 19, 20, 23, 24, 29),
                leaf_7_ecx: bits!(2, 22),
                leaf_7_edx: 0,
                extended_1_ecx: amd_extended_1_ecx | bits!(23),
                extended_1_edx: AMD_EXTENDED_EDX,
                extended_8_ebx: bits!(0, 2, 9, 12, 15),
                extended_a_edx: if nested { bits!(0, 3) } else { 0 },
                xcr0: bits!(0, 1, 2),
                xsave_1_eax: bits!(0, 1, 2),
            },
            CpuModel::EpycMilan => Self {
                vendor: CpuVendor::AMD,
                signature: 0x00a0_0f11,
                max_extended_leaf: 0x8000_001e,
                brand: "AMD EPYC Milan Processor",
                leaf_1_ecx: amd_leaf_1_ecx,
                leaf_7_ebx: bits!(0, 3, 5, 7, 8, 9, 10, 18, 19, 20, 23, 24, 29),
                leaf_7_ecx: bits!(2, 3, 22),
                leaf_7_edx: bits!(4),
                extended_1_ecx: amd_extended_1_ecx | bits!(23),
                extended_1_edx: AMD_EXTENDED_EDX,
                extended_8_ebx: bits!(0, 2, 9, 12, 14, 15, 24),
                extended_a_edx: if nested { bits!(0, 3, 28) } else { 0 },
                xcr0: bits!(0, 1, 2, 9),
                xsave_1_eax: bits!(0, 1, 2, 3),
            },
        }
    }
}

#[derive(Clone, Copy)]
enum Register {
    Eax,
    Ebx,
    Ecx,
    Edx,
}

fn apply_register(
    cpuid: &mut [CpuIdEntry],
    model: CpuModel,
    function: u32,
    index: u32,
    register: Register,
    value: u32,
) -> Result<(), Error> {
    let entry = cpuid
        .iter_mut()
        .find(|entry| entry.function == function && entry.index == index)
        .ok_or_else(|| Error::CpuModelNotSupported {
            model: model.to_string(),
            reason: format!("CPUID leaf 0x{function:x}, subleaf 0x{index:x} is unavailable"),
        })?;

    let (host_value, exact_flag) = match register {
        Register::Eax => (&mut entry.eax, CPUID_FLAG_EXACT_EAX),
        Register::Ebx => (&mut entry.ebx, CPUID_FLAG_EXACT_EBX),
        Register::Ecx => (&mut entry.ecx, CPUID_FLAG_EXACT_ECX),
        Register::Edx => (&mut entry.edx, CPUID_FLAG_EXACT_EDX),
    };

    if *host_value & value != value {
        return Err(Error::CpuModelNotSupported {
            model: model.to_string(),
            reason: format!(
                "CPUID leaf 0x{function:x}, subleaf 0x{index:x} lacks required bits 0x{:08x}",
                value & !*host_value
            ),
        });
    }

    *host_value = value;
    entry.flags |= exact_flag;
    Ok(())
}

fn set_signature(cpuid: &mut [CpuIdEntry], model: CpuModel, signature: u32) -> Result<(), Error> {
    let entry = cpuid
        .iter_mut()
        .find(|entry| entry.function == 1 && entry.index == 0)
        .ok_or_else(|| Error::CpuModelNotSupported {
            model: model.to_string(),
            reason: "CPUID leaf 0x1 is unavailable".to_string(),
        })?;
    entry.eax = signature;
    entry.flags |= CPUID_FLAG_EXACT_EAX;
    Ok(())
}

fn set_max_leaf(
    cpuid: &mut [CpuIdEntry],
    model: CpuModel,
    function: u32,
    value: u32,
) -> Result<(), Error> {
    let entry = cpuid
        .iter_mut()
        .find(|entry| entry.function == function && entry.index == 0)
        .ok_or_else(|| Error::CpuModelNotSupported {
            model: model.to_string(),
            reason: format!("CPUID leaf 0x{function:x} is unavailable"),
        })?;
    if entry.eax < value {
        return Err(Error::CpuModelNotSupported {
            model: model.to_string(),
            reason: format!(
                "CPUID leaf 0x{function:x} reports maximum 0x{:x}, but the model requires 0x{value:x}",
                entry.eax
            ),
        });
    }
    entry.eax = value;
    entry.flags |= CPUID_FLAG_EXACT_EAX;
    Ok(())
}

fn set_brand_string(cpuid: &mut Vec<CpuIdEntry>, brand: &str) {
    let mut bytes = [b' '; 48];
    let brand = brand.as_bytes();
    bytes[..brand.len()].copy_from_slice(brand);

    for (offset, function) in (0x8000_0002..=0x8000_0004).enumerate() {
        cpuid.retain(|entry| entry.function != function);
        let words: [u32; 4] = std::array::from_fn(|index| {
            let start = offset * 16 + index * 4;
            u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
        });
        cpuid.push(CpuIdEntry {
            function,
            flags: CPUID_FLAG_EXACT_EAX
                | CPUID_FLAG_EXACT_EBX
                | CPUID_FLAG_EXACT_ECX
                | CPUID_FLAG_EXACT_EDX,
            eax: words[0],
            ebx: words[1],
            ecx: words[2],
            edx: words[3],
            ..Default::default()
        });
    }
}

pub(super) fn apply_cpu_model(
    cpuid: &mut Vec<CpuIdEntry>,
    model: CpuModel,
    host_vendor: CpuVendor,
    nested: bool,
) -> Result<(), Error> {
    let profile = CpuModelProfile::for_model(model, nested);
    if host_vendor != profile.vendor {
        return Err(Error::CpuModelNotSupported {
            model: model.to_string(),
            reason: format!(
                "the model requires a {:?} host, but this host is {:?}",
                profile.vendor, host_vendor
            ),
        });
    }

    set_max_leaf(cpuid, model, 0, 0xd)?;
    set_max_leaf(cpuid, model, 0x8000_0000, profile.max_extended_leaf)?;
    set_signature(cpuid, model, profile.signature)?;
    for (function, index, register, value) in [
        (1, 0, Register::Ebx, 8 << 8),
        (1, 0, Register::Ecx, profile.leaf_1_ecx),
        (1, 0, Register::Edx, LEAF_1_EDX),
        (6, 0, Register::Eax, bits!(2)),
        (7, 0, Register::Eax, 0),
        (7, 0, Register::Ebx, profile.leaf_7_ebx),
        (7, 0, Register::Ecx, profile.leaf_7_ecx),
        (7, 0, Register::Edx, profile.leaf_7_edx),
        (0xd, 0, Register::Eax, profile.xcr0),
        (0xd, 0, Register::Edx, 0),
        (0xd, 1, Register::Eax, profile.xsave_1_eax),
        (0xd, 1, Register::Ecx, 0),
        (0xd, 1, Register::Edx, 0),
        (0x8000_0001, 0, Register::Ecx, profile.extended_1_ecx),
        (0x8000_0001, 0, Register::Edx, profile.extended_1_edx),
        (0x8000_0007, 0, Register::Edx, 0),
        (0x8000_0008, 0, Register::Ebx, profile.extended_8_ebx),
    ] {
        apply_register(cpuid, model, function, index, register, value)?;
    }

    if matches!(profile.vendor, CpuVendor::AMD) {
        apply_register(
            cpuid,
            model,
            0x8000_000a,
            0,
            Register::Edx,
            profile.extended_a_edx,
        )?;
    }

    set_brand_string(cpuid, profile.brand);
    Ok(())
}

#[cfg(test)]
mod tests {
    use hypervisor::arch::x86::CPUID_FLAG_VALID_INDEX;

    use super::*;

    fn full_cpuid() -> Vec<CpuIdEntry> {
        [
            (0, 0),
            (1, 0),
            (6, 0),
            (7, 0),
            (0xd, 0),
            (0xd, 1),
            (0x8000_0000, 0),
            (0x8000_0001, 0),
            (0x8000_0007, 0),
            (0x8000_0008, 0),
            (0x8000_000a, 0),
        ]
        .into_iter()
        .map(|(function, index)| CpuIdEntry {
            function,
            index,
            flags: if index == 0 {
                0
            } else {
                CPUID_FLAG_VALID_INDEX
            },
            eax: u32::MAX,
            ebx: u32::MAX,
            ecx: u32::MAX,
            edx: u32::MAX,
        })
        .collect()
    }

    #[test]
    fn model_masks_newer_cpu_features() {
        let mut cpuid = full_cpuid();
        apply_cpu_model(&mut cpuid, CpuModel::SkylakeServer, CpuVendor::Intel, false).unwrap();

        let leaf_1 = cpuid.iter().find(|entry| entry.function == 1).unwrap();
        assert_eq!(leaf_1.eax, 0x0005_0654);
        assert_eq!(leaf_1.ebx, 8 << 8);
        assert_eq!(leaf_1.ecx, INTEL_LEAF_1_ECX);
        assert_ne!(leaf_1.flags & CPUID_FLAG_EXACT_ECX, 0);

        let leaf_7 = cpuid.iter().find(|entry| entry.function == 7).unwrap();
        assert_eq!(leaf_7.ecx, 0);
        assert_eq!(leaf_7.edx, 0);
    }

    #[test]
    fn model_rejects_missing_feature() {
        let mut cpuid = full_cpuid();
        cpuid
            .iter_mut()
            .find(|entry| entry.function == 7)
            .unwrap()
            .ebx &= !bits!(5);

        assert!(matches!(
            apply_cpu_model(&mut cpuid, CpuModel::SkylakeServer, CpuVendor::Intel, false,),
            Err(Error::CpuModelNotSupported { .. })
        ));
    }

    #[test]
    fn model_rejects_wrong_vendor() {
        let mut cpuid = full_cpuid();
        assert!(matches!(
            apply_cpu_model(&mut cpuid, CpuModel::EpycMilan, CpuVendor::Intel, false,),
            Err(Error::CpuModelNotSupported { .. })
        ));
    }

    #[test]
    fn no_tsx_model_clears_tsx_features() {
        let mut cpuid = full_cpuid();
        apply_cpu_model(
            &mut cpuid,
            CpuModel::SkylakeServerNoTsxIbrs,
            CpuVendor::Intel,
            false,
        )
        .unwrap();

        let leaf_7 = cpuid.iter().find(|entry| entry.function == 7).unwrap();
        assert_eq!(leaf_7.ebx & bits!(4, 11), 0);
        assert_ne!(leaf_7.edx & bits!(26), 0);
    }
}
