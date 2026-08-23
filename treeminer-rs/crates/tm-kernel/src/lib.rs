//! TreeMiner's Argon2 device kernels, in Rust, compiled for `amdgcn-amd-amdhsa`.
//!
//! This crate is never built for the host. `tm-gpu`'s `build.rs` compiles it to an AMDGPU
//! code object which the runtime loads with `hipModuleLoad`; see PORT.md, "Rust GPU
//! kernels", for the toolchain and why it is staged behind the `rust-kernel` feature.
//!
//! Port of the `__device__` / `__global__` half of `../tm-gpu/kernel/argon2_kernel.hip`.
//! Behaviour must match that file bit for bit: the two are cross-checked against each other
//! by `first_blocks_differential` in `tm-gpu/src/runner.rs`, and a silent divergence costs
//! real submissions (commit 12e241c).
//!
//! Both kernels are ported. `argon2_first_blocks_kernel` (stage 1) is plain per-work-item
//! arithmetic — no wavefront shuffles, no shared memory, one Argon2 job per thread.
//! `argon2_kernel_oneshot` (stage 2) is the hot loop: one workgroup of 32 work-items per
//! job, the block spread across the wavefront, and `ds_bpermute` shuffles in place of the
//! C++'s per-thread LDS scratch.

#![no_std]
#![feature(abi_gpu_kernel, link_llvm_intrinsics)]
// `link_llvm_intrinsics` is the only way to reach the AMDGPU workitem/workgroup ids from
// Rust; there is no stable equivalent.
#![allow(internal_features)]
// The kernel entry point is `unsafe extern "gpu-kernel"`; every pointer it touches is a
// device allocation the host sized, so the whole file is unsafe by nature and marking each
// individual dereference would only add noise.
#![allow(clippy::missing_safety_doc)]

use core::panic::PanicInfo;

/// There is no unwinding and nothing to report from a work-item. A panic can only come from
/// a bounds check that the surrounding code makes unreachable, so parking the wavefront is
/// as good as it gets — and it is loud, because the launch never completes.
#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

extern "C" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn workitem_id_x() -> u32;
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn workgroup_id_x() -> u32;
}

const ARGON2_BLOCK_SIZE: usize = 1024;
const ARGON2_SYNC_POINTS: u32 = 4;
const ARGON2_PREHASH_DIGEST_LENGTH: usize = 64;
const ARGON2_PREHASH_SEED_LENGTH: usize = 72;
const BLAKE2B_BLOCK_BYTES: usize = 128;
const BLAKE2B_OUT_BYTES: usize = 64;

const BLAKE2B_IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

const BLAKE2B_SIGMA: [[u8; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// `device_store32`.
#[inline(always)]
unsafe fn store32(dst: *mut u8, value: u32) {
    let bytes = value.to_le_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        dst.add(index).write(*byte);
    }
}

/// `device_store64`.
#[inline(always)]
unsafe fn store64(dst: *mut u8, value: u64) {
    let bytes = value.to_le_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        dst.add(index).write(*byte);
    }
}

/// `device_load64`. Unaligned by construction — the input may be any byte offset into a
/// message block, which is why the C++ builds the word a byte at a time too.
#[inline(always)]
unsafe fn load64(src: *const u8) -> u64 {
    let mut value = 0u64;
    let mut index = 0;
    while index < 8 {
        value |= (src.add(index).read() as u64) << (8 * index);
        index += 1;
    }
    value
}

#[inline(always)]
fn rotr64(x: u64, n: u32) -> u64 {
    x.rotate_right(n)
}

/// `Blake2bDeviceState`. Lives entirely in registers/private scratch, one per work-item.
struct Blake2b {
    h: [u64; 8],
    t: [u64; 2],
    buf: [u8; BLAKE2B_BLOCK_BYTES],
    buf_len: u32,
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn blake_g(m: &[u64; 16], r: usize, i: usize, v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize) {
    v[a] = v[a]
        .wrapping_add(v[b])
        .wrapping_add(m[BLAKE2B_SIGMA[r][2 * i] as usize]);
    v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] = v[a]
        .wrapping_add(v[b])
        .wrapping_add(m[BLAKE2B_SIGMA[r][2 * i + 1] as usize]);
    v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = rotr64(v[b] ^ v[c], 63);
}

#[inline(always)]
fn blake_round(m: &[u64; 16], v: &mut [u64; 16], r: usize) {
    blake_g(m, r, 0, v, 0, 4, 8, 12);
    blake_g(m, r, 1, v, 1, 5, 9, 13);
    blake_g(m, r, 2, v, 2, 6, 10, 14);
    blake_g(m, r, 3, v, 3, 7, 11, 15);
    blake_g(m, r, 4, v, 0, 5, 10, 15);
    blake_g(m, r, 5, v, 1, 6, 11, 12);
    blake_g(m, r, 6, v, 2, 7, 8, 13);
    blake_g(m, r, 7, v, 3, 4, 9, 14);
}

impl Blake2b {
    /// `device_blake2b_init`. The parameter block is the Argon2 one: no key, fanout 1,
    /// depth 1, which is what the two `1 <<` terms encode.
    fn init(out_len: u32) -> Self {
        let mut h = BLAKE2B_IV;
        h[0] ^= (out_len as u64) | (1u64 << 16) | (1u64 << 24);
        Self {
            h,
            t: [0, 0],
            buf: [0u8; BLAKE2B_BLOCK_BYTES],
            buf_len: 0,
        }
    }

    #[inline(always)]
    fn increment_counter(&mut self, inc: u64) {
        self.t[0] = self.t[0].wrapping_add(inc);
        self.t[1] = self.t[1].wrapping_add(u64::from(self.t[0] < inc));
    }

    /// `device_blake2b_compress`.
    unsafe fn compress(&mut self, block: *const u8, f0: u64) {
        let mut m = [0u64; 16];
        for (index, word) in m.iter_mut().enumerate() {
            *word = load64(block.add(index * 8));
        }

        let mut v = [0u64; 16];
        for index in 0..8 {
            v[index] = self.h[index];
            v[index + 8] = BLAKE2B_IV[index];
        }
        v[12] ^= self.t[0];
        v[13] ^= self.t[1];
        v[14] ^= f0;

        for r in 0..12 {
            blake_round(&m, &mut v, r);
        }

        for index in 0..8 {
            self.h[index] ^= v[index] ^ v[index + 8];
        }
    }

    /// `device_blake2b_update`. Note the strict `>` comparisons: the last full block is
    /// deliberately left buffered so `final` can apply the finalisation flag to it.
    unsafe fn update(&mut self, input: *const u8, input_len: u32) {
        let mut input = input;
        let mut input_len = input_len;
        if self.buf_len as usize + input_len as usize > BLAKE2B_BLOCK_BYTES {
            let have = self.buf_len as usize;
            let left = BLAKE2B_BLOCK_BYTES - have;
            for index in 0..left {
                self.buf[have + index] = input.add(index).read();
            }

            self.increment_counter(BLAKE2B_BLOCK_BYTES as u64);
            let buf = self.buf.as_ptr();
            self.compress(buf, 0);

            self.buf_len = 0;
            input_len -= left as u32;
            input = input.add(left);

            while input_len as usize > BLAKE2B_BLOCK_BYTES {
                self.increment_counter(BLAKE2B_BLOCK_BYTES as u64);
                self.compress(input, 0);
                input_len -= BLAKE2B_BLOCK_BYTES as u32;
                input = input.add(BLAKE2B_BLOCK_BYTES);
            }
        }
        for index in 0..input_len as usize {
            self.buf[self.buf_len as usize + index] = input.add(index).read();
        }
        self.buf_len += input_len;
    }

    /// `device_blake2b_final`. Consumes the state, matching the C++'s one-shot use.
    unsafe fn finalize(mut self, out: *mut u8, out_len: u32) {
        self.increment_counter(u64::from(self.buf_len));
        for index in self.buf_len as usize..BLAKE2B_BLOCK_BYTES {
            self.buf[index] = 0;
        }
        let buf = self.buf.as_ptr();
        self.compress(buf, u64::MAX);

        let mut buffer = [0u8; BLAKE2B_OUT_BYTES];
        for index in 0..8 {
            store64(buffer.as_mut_ptr().add(index * 8), self.h[index]);
        }
        for index in 0..out_len as usize {
            out.add(index).write(buffer[index]);
        }
    }
}

/// `device_digest_long` — Argon2's variable-length hash H'.
unsafe fn digest_long(out: *mut u8, out_len: u32, input: *const u8, input_len: u32) {
    let mut output = out;
    let mut out_len_bytes = [0u8; 4];
    store32(out_len_bytes.as_mut_ptr(), out_len);
    let len_ptr = out_len_bytes.as_ptr();

    if out_len as usize <= BLAKE2B_OUT_BYTES {
        let mut blake = Blake2b::init(out_len);
        blake.update(len_ptr, 4);
        blake.update(input, input_len);
        blake.finalize(out, out_len);
        return;
    }

    let mut out_buffer = [0u8; BLAKE2B_OUT_BYTES];
    let mut blake = Blake2b::init(BLAKE2B_OUT_BYTES as u32);
    blake.update(len_ptr, 4);
    blake.update(input, input_len);
    blake.finalize(out_buffer.as_mut_ptr(), BLAKE2B_OUT_BYTES as u32);

    let half = BLAKE2B_OUT_BYTES / 2;
    for byte in out_buffer.iter().take(half) {
        output.write(*byte);
        output = output.add(1);
    }

    let mut to_produce = out_len - half as u32;
    while to_produce as usize > BLAKE2B_OUT_BYTES {
        let previous = out_buffer;
        let mut blake = Blake2b::init(BLAKE2B_OUT_BYTES as u32);
        blake.update(previous.as_ptr(), BLAKE2B_OUT_BYTES as u32);
        blake.finalize(out_buffer.as_mut_ptr(), BLAKE2B_OUT_BYTES as u32);

        for byte in out_buffer.iter().take(half) {
            output.write(*byte);
            output = output.add(1);
        }
        to_produce -= half as u32;
    }

    let previous = out_buffer;
    let mut blake = Blake2b::init(to_produce);
    blake.update(previous.as_ptr(), BLAKE2B_OUT_BYTES as u32);
    blake.finalize(output, to_produce);
}

/// `device_initial_hash` — Argon2's H0.
#[allow(clippy::too_many_arguments)]
unsafe fn initial_hash(
    out: *mut u8,
    password: *const u8,
    password_len: u32,
    salt: *const u8,
    salt_len: u32,
    output_len: u32,
    memory_cost: u32,
    time_cost: u32,
    version: u32,
    type_: u32,
    lanes: u32,
) {
    let mut blake = Blake2b::init(ARGON2_PREHASH_DIGEST_LENGTH as u32);

    let mut header = [0u8; 7 * 4];
    let fields = [
        lanes,
        output_len,
        memory_cost,
        time_cost,
        version,
        type_,
        password_len,
    ];
    for (index, field) in fields.iter().enumerate() {
        store32(header.as_mut_ptr().add(index * 4), *field);
    }
    blake.update(header.as_ptr(), header.len() as u32);
    blake.update(password, password_len);

    let mut value = [0u8; 4];
    store32(value.as_mut_ptr(), salt_len);
    blake.update(value.as_ptr(), 4);
    blake.update(salt, salt_len);

    // Argon2id with no secret and no associated data: two zero lengths, no bytes.
    store32(value.as_mut_ptr(), 0);
    blake.update(value.as_ptr(), 4);
    blake.update(value.as_ptr(), 4);

    blake.finalize(out, ARGON2_PREHASH_DIGEST_LENGTH as u32);
}

/// `argon2_first_blocks_kernel`: derives blocks 0 and 1 of every job in the batch.
///
/// The argument order is *not* the C++ kernel's. It is regrouped — pointers, then the 64-bit
/// integer, then the 32-bit ones — so that the AMDGPU kernarg segment has no interior
/// padding and the host's `#[repr(C)]` mirror in `tm-gpu/src/module.rs` cannot disagree with
/// it. `threads_per_block` is passed explicitly because the workgroup size is only otherwise
/// reachable through the HSA dispatch packet.
///
/// # Safety
/// `memory` must hold `batch_size` jobs of `4 * segment_blocks` 1 KiB blocks, `keys`
/// `batch_size * key_length` bytes and `salt` `salt_length` bytes, all device-resident.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "gpu-kernel" fn argon2_first_blocks_kernel(
    memory: *mut u8,
    keys: *const u8,
    salt: *const u8,
    batch_size: u64,
    key_length: u32,
    salt_length: u32,
    output_length: u32,
    memory_cost: u32,
    time_cost: u32,
    version: u32,
    type_: u32,
    lanes: u32,
    segment_blocks: u32,
    threads_per_block: u32,
) {
    let job_id = u64::from(workgroup_id_x()) * u64::from(threads_per_block)
        + u64::from(workitem_id_x());
    if job_id >= batch_size {
        return;
    }

    let password = keys.add((job_id * u64::from(key_length)) as usize);
    let mut init_hash = [0u8; ARGON2_PREHASH_SEED_LENGTH];
    initial_hash(
        init_hash.as_mut_ptr(),
        password,
        key_length,
        salt,
        salt_length,
        output_length,
        memory_cost,
        time_cost,
        version,
        type_,
        lanes,
    );

    let lane_blocks = u64::from(ARGON2_SYNC_POINTS * segment_blocks);
    let output = memory.add((job_id * lane_blocks * ARGON2_BLOCK_SIZE as u64) as usize);

    let counter = init_hash.as_mut_ptr().add(ARGON2_PREHASH_DIGEST_LENGTH);
    store32(counter, 0);
    store32(counter.add(4), 0);
    digest_long(
        output,
        ARGON2_BLOCK_SIZE as u32,
        init_hash.as_ptr(),
        ARGON2_PREHASH_SEED_LENGTH as u32,
    );

    let counter = init_hash.as_mut_ptr().add(ARGON2_PREHASH_DIGEST_LENGTH);
    store32(counter, 1);
    store32(counter.add(4), 0);
    digest_long(
        output.add(ARGON2_BLOCK_SIZE),
        ARGON2_BLOCK_SIZE as u32,
        init_hash.as_ptr(),
        ARGON2_PREHASH_SEED_LENGTH as u32,
    );
}

// ---------------------------------------------------------------- stage 2: the one-shot
//
// Port of `argon2_kernel_oneshot` and everything it calls. One workgroup of
// `THREADS_PER_LANE` work-items per Argon2 job; the 1 KiB block is spread across the
// wavefront four `u64` per work-item, and the Argon2 permutation is carried out with
// wavefront shuffles instead of shared memory.
//
// The C++'s `extern __shared__ struct block_l` is *not* shared between threads:
// `block_l_store` writes slots `[i * 32 + thread]` and `block_l_load_xor` reads back the
// same thread's slots. It only exists to spill four registers on NVIDIA. Here that
// scratch is a `BlockTh` in registers, which is the shape `move_block`/`xor_block` already
// have, and rustc's amdgcn target cannot express `addrspace(3)` globals anyway.

const THREADS_PER_LANE: u32 = 32;
const QWORDS_IN_BLOCK: usize = ARGON2_BLOCK_SIZE / 8;

extern "C" {
    #[link_name = "llvm.amdgcn.mbcnt.lo"]
    fn mbcnt_lo(mask: u32, base: u32) -> u32;
    #[link_name = "llvm.amdgcn.mbcnt.hi"]
    fn mbcnt_hi(mask: u32, base: u32) -> u32;
    #[link_name = "llvm.amdgcn.ds.bpermute"]
    fn ds_bpermute(byte_index: u32, src: u32) -> u32;
}

/// This work-item's index inside its wavefront.
#[inline(always)]
fn lane_id() -> u32 {
    // SAFETY: both intrinsics are pure reads of the lane-mask registers.
    unsafe { mbcnt_hi(!0, mbcnt_lo(!0, 0)) }
}

/// `TM_SHFL` for a 32-bit value: read `value` from lane `src` of this work-item's *group of
/// 32*. `ds_bpermute` addresses the whole wavefront, so the group base is put back — on a
/// wave64 part lanes 32..63 must read from their own half, exactly as HIP's `__shfl(...,
/// THREADS_PER_LANE)` does.
#[inline(always)]
fn shfl32(value: u32, src: u32) -> u32 {
    let lane = (lane_id() & !(THREADS_PER_LANE - 1)) | (src & (THREADS_PER_LANE - 1));
    // SAFETY: `ds_bpermute` is a cross-lane register read; the byte index is a lane number
    // scaled by 4, which is the intrinsic's documented encoding.
    unsafe { ds_bpermute(lane << 2, value) }
}

/// `u64_shuffle`: two 32-bit shuffles, like the C++.
#[inline(always)]
fn shfl64(value: u64, src: u32) -> u64 {
    let lo = shfl32(value as u32, src);
    let hi = shfl32((value >> 32) as u32, src);
    (u64::from(hi) << 32) | u64::from(lo)
}

/// `TM_SHFL_XOR`. For masks below 32 — the only ones used — this is the same lane the HIP
/// builtin picks, because the group base is restored by [`shfl32`].
#[inline(always)]
fn shfl_xor64(value: u64, mask: u32) -> u64 {
    shfl64(value, lane_id() ^ mask)
}

/// `struct block_th`: this work-item's quarter of the Argon2 block.
#[derive(Clone, Copy)]
struct BlockTh {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

/// `cmpeq_mask`: all-ones when the indices match. Branchless on purpose — a `match` on a
/// divergent index compiles to an indexed local array, which on AMDGPU means scratch.
#[inline(always)]
fn cmpeq_mask(test: u32, reference: u32) -> u64 {
    // `0u64.wrapping_sub(1)` is the 64-bit form of the C++'s `-(test == ref)` broadcast.
    0u64.wrapping_sub(u64::from(test == reference))
}

impl BlockTh {
    /// `block_th_get`, for an index that may differ between work-items.
    #[inline(always)]
    fn get(&self, index: u32) -> u64 {
        (cmpeq_mask(index, 0) & self.a)
            ^ (cmpeq_mask(index, 1) & self.b)
            ^ (cmpeq_mask(index, 2) & self.c)
            ^ (cmpeq_mask(index, 3) & self.d)
    }

    /// `block_th_get_uniform`: the caller knows the index is the same in every work-item,
    /// so the select can be a real branch on a scalar register.
    #[inline(always)]
    fn get_uniform(&self, index: u32) -> u64 {
        match index {
            0 => self.a,
            1 => self.b,
            2 => self.c,
            _ => self.d,
        }
    }

    /// `block_th_set`, branchless for the same reason as [`Self::get`].
    #[inline(always)]
    fn set(&mut self, index: u32, value: u64) {
        self.a ^= cmpeq_mask(index, 0) & (value ^ self.a);
        self.b ^= cmpeq_mask(index, 1) & (value ^ self.b);
        self.c ^= cmpeq_mask(index, 2) & (value ^ self.c);
        self.d ^= cmpeq_mask(index, 3) & (value ^ self.d);
    }

    /// `xor_block`.
    #[inline(always)]
    fn xor(&mut self, other: &BlockTh) {
        self.a ^= other.a;
        self.b ^= other.b;
        self.c ^= other.c;
        self.d ^= other.d;
    }

    /// `load_block`.
    #[inline(always)]
    unsafe fn load(block: *const u64, thread: u32) -> Self {
        let thread = thread as usize;
        let lane = THREADS_PER_LANE as usize;
        Self {
            a: block.add(thread).read(),
            b: block.add(lane + thread).read(),
            c: block.add(2 * lane + thread).read(),
            d: block.add(3 * lane + thread).read(),
        }
    }

    /// `load_block_xor`.
    #[inline(always)]
    unsafe fn load_xor(&mut self, block: *const u64, thread: u32) {
        let other = Self::load(block, thread);
        self.xor(&other);
    }

    /// `store_block`.
    #[inline(always)]
    unsafe fn store(&self, block: *mut u64, thread: u32) {
        let thread = thread as usize;
        let lane = THREADS_PER_LANE as usize;
        block.add(thread).write(self.a);
        block.add(lane + thread).write(self.b);
        block.add(2 * lane + thread).write(self.c);
        block.add(3 * lane + thread).write(self.d);
    }
}

/// `f`: the Argon2 mixing addition. `u64_build(__umulhi(xlo, ylo), xlo * ylo)` is just the
/// widening 32x32 product, which is one instruction on every target.
#[inline(always)]
fn f(x: u64, y: u64) -> u64 {
    let product = u64::from(x as u32).wrapping_mul(u64::from(y as u32));
    x.wrapping_add(y).wrapping_add(product.wrapping_mul(2))
}

/// `g1` — the Argon2 G function. The C++'s other `g()` is hand-written PTX for nvcc and has
/// no bearing here; on AMD the C++ path is what `TREEMINER_GPU_HIP` selects too.
#[inline(always)]
fn g(block: &mut BlockTh) {
    let mut a = block.a;
    let mut b = block.b;
    let mut c = block.c;
    let mut d = block.d;

    a = f(a, b);
    d = (d ^ a).rotate_right(32);
    c = f(c, d);
    b = (b ^ c).rotate_right(24);
    a = f(a, b);
    d = (d ^ a).rotate_right(16);
    c = f(c, d);
    b = (b ^ c).rotate_right(63);

    block.a = a;
    block.b = b;
    block.c = c;
    block.d = d;
}

/// `transpose`: the branchless four-way rotation of the `(a, b, c, d)` roles that turns a
/// row-wise block layout into a column-wise one.
#[inline(always)]
fn transpose(block: &mut BlockTh, thread: u32) {
    let g1 = thread & 0x4 != 0;
    let g2 = thread & 0x8 != 0;

    let mut x1 = if g2 {
        if g1 { block.c } else { block.d }
    } else if g1 {
        block.a
    } else {
        block.b
    };
    let mut x2 = if g2 {
        if g1 { block.b } else { block.a }
    } else if g1 {
        block.d
    } else {
        block.c
    };
    let mut x3 = if g2 {
        if g1 { block.a } else { block.b }
    } else if g1 {
        block.c
    } else {
        block.d
    };

    x1 = shfl_xor64(x1, 0x4);
    x2 = shfl_xor64(x2, 0x8);
    x3 = shfl_xor64(x3, 0xC);

    let a = if g2 {
        if g1 { x3 } else { x2 }
    } else if g1 {
        x1
    } else {
        block.a
    };
    let b = if g2 {
        if g1 { x2 } else { x3 }
    } else if g1 {
        block.b
    } else {
        x1
    };
    let c = if g2 {
        if g1 { x1 } else { block.c }
    } else if g1 {
        x3
    } else {
        x2
    };
    let d = if g2 {
        if g1 { block.d } else { x1 }
    } else if g1 {
        x2
    } else {
        x3
    };

    block.a = a;
    block.b = b;
    block.c = c;
    block.d = d;
}

/// `shift1_shuffle`.
#[inline(always)]
fn shift1_shuffle(block: &mut BlockTh, thread: u32) {
    let src_b = (thread & 0x1c) | ((thread + 1) & 0x3);
    let src_d = (thread & 0x1c) | ((thread + 3) & 0x3);

    block.b = shfl64(block.b, src_b);
    block.c = shfl_xor64(block.c, 0x2);
    block.d = shfl64(block.d, src_d);
}

/// `unshift1_shuffle`: the same with the `b` and `d` offsets swapped.
#[inline(always)]
fn unshift1_shuffle(block: &mut BlockTh, thread: u32) {
    let src_b = (thread & 0x1c) | ((thread + 3) & 0x3);
    let src_d = (thread & 0x1c) | ((thread + 1) & 0x3);

    block.b = shfl64(block.b, src_b);
    block.c = shfl_xor64(block.c, 0x2);
    block.d = shfl64(block.d, src_d);
}

/// `shift2_shuffle`.
#[inline(always)]
fn shift2_shuffle(block: &mut BlockTh, thread: u32) {
    let lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    let src_b = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);
    let src_d = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);

    block.b = shfl64(block.b, src_b);
    block.c = shfl_xor64(block.c, 0x10);
    block.d = shfl64(block.d, src_d);
}

/// `unshift2_shuffle`.
#[inline(always)]
fn unshift2_shuffle(block: &mut BlockTh, thread: u32) {
    let lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    let src_b = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);
    let src_d = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);

    block.b = shfl64(block.b, src_b);
    block.c = shfl_xor64(block.c, 0x10);
    block.d = shfl64(block.d, src_d);
}

/// `shuffle_block`: the full Argon2 block permutation — two rounds along rows, two along
/// columns, with the wavefront shuffles standing in for the data movement.
#[inline(always)]
fn shuffle_block(block: &mut BlockTh, thread: u32) {
    transpose(block, thread);
    g(block);
    shift1_shuffle(block, thread);
    g(block);
    unshift1_shuffle(block, thread);
    transpose(block, thread);
    g(block);
    shift2_shuffle(block, thread);
    g(block);
    unshift2_shuffle(block, thread);
}

/// `next_addresses1`: derives the next block of reference indices for the data-independent
/// slices. `tmp` is a register copy here, not the C++'s `block_l` scratch.
#[inline(always)]
fn next_addresses(addr: &mut BlockTh, thread_input: u32, thread: u32) {
    addr.a = u64::from(thread_input);
    addr.b = 0;
    addr.c = 0;
    addr.d = 0;

    shuffle_block(addr, thread);

    addr.a ^= u64::from(thread_input);
    let tmp = *addr;

    shuffle_block(addr, thread);

    addr.xor(&tmp);
}

/// `__umulhi`.
#[inline(always)]
fn umulhi(x: u32, y: u32) -> u32 {
    ((u64::from(x) * u64::from(y)) >> 32) as u32
}

/// `compute_ref_pos`: maps a pseudo-random word onto a block in the already-computed area,
/// with Argon2's quadratic bias toward recent blocks.
#[inline(always)]
fn compute_ref_pos(segment_blocks: u32, slice: u32, offset: u32, ref_index: u32) -> u32 {
    let ref_area_size = (slice.wrapping_mul(segment_blocks))
        .wrapping_add(offset)
        .wrapping_sub(1);
    let index = umulhi(ref_index, ref_index);
    ref_area_size
        .wrapping_sub(1)
        .wrapping_sub(umulhi(ref_area_size, index))
}

/// `argon2_core`: the compression function G — XOR in the referenced block, permute, XOR
/// the pre-permutation value back in, store.
///
/// # Safety
/// `memory` must be this job's lane and `ref_index` a block inside it; `mem_curr` must be
/// a writable block of the same lane.
#[inline(always)]
unsafe fn argon2_core(
    memory: *const u64,
    mem_curr: *mut u64,
    prev: &mut BlockTh,
    thread: u32,
    ref_index: u32,
) {
    let mem_ref = memory.add(ref_index as usize * QWORDS_IN_BLOCK);

    prev.load_xor(mem_ref, thread);
    let tmp = *prev;

    shuffle_block(prev, thread);

    prev.xor(&tmp);
    prev.store(mem_curr, thread);
}

/// `argon2_step_indexed`: one block of a data-*independent* slice — the reference index
/// comes from the address block, refreshed every 128 offsets.
///
/// # Safety
/// As for [`argon2_core`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn argon2_step_indexed(
    memory: *const u64,
    mem_curr: *mut u64,
    prev: &mut BlockTh,
    addr: &mut BlockTh,
    segment_blocks: u32,
    thread: u32,
    thread_input: &mut u32,
    slice: u32,
    offset: u32,
) {
    let addr_index = offset % QWORDS_IN_BLOCK as u32;
    if addr_index == 0 {
        if thread == 6 {
            *thread_input += 1;
        }
        next_addresses(addr, *thread_input, thread);
    }

    // `offset` is the same in every work-item, so both of these are uniform.
    let thr = addr_index % THREADS_PER_LANE;
    let idx = addr_index / THREADS_PER_LANE;

    let value = shfl64(addr.get_uniform(idx), thr);
    let ref_index = compute_ref_pos(segment_blocks, slice, offset, value as u32);

    argon2_core(memory, mem_curr, prev, thread, ref_index);
}

/// `argon2_step_dependent`: one block of a data-*dependent* slice — the reference index is
/// the previous block's first word, broadcast from work-item 0.
///
/// # Safety
/// As for [`argon2_core`].
#[inline(always)]
unsafe fn argon2_step_dependent(
    memory: *const u64,
    mem_curr: *mut u64,
    prev: &mut BlockTh,
    segment_blocks: u32,
    thread: u32,
    slice: u32,
    offset: u32,
) {
    let value = shfl64(prev.a, 0);
    let ref_index = compute_ref_pos(segment_blocks, slice, offset, value as u32);

    argon2_core(memory, mem_curr, prev, thread, ref_index);
}

/// `argon2_kernel_oneshot`: fills blocks 2.. of every job's lane, one pass, one lane.
///
/// Launched as one workgroup of `THREADS_PER_LANE` work-items per job, exactly like the HIP
/// shim, so the work-item id *is* the Argon2 thread index and no workgroup size has to be
/// passed in.
///
/// # Safety
/// `memory` must hold one job per workgroup of `4 * segment_blocks` 1 KiB blocks, with
/// blocks 0 and 1 of each already filled.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn argon2_kernel_oneshot(memory: *mut u64, segment_blocks: u32) {
    let job_id = workgroup_id_x() as usize;
    let thread = workitem_id_x();

    let lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;
    let mem_lane = memory.add(job_id * lane_blocks as usize * QWORDS_IN_BLOCK);

    let mut thread_input = u32::from(thread == 3) * lane_blocks
        + u32::from(thread == 4)
        + u32::from(thread == 5) * 2
        + u32::from(thread == 6);

    let mut addr = BlockTh {
        a: 0,
        b: 0,
        c: 0,
        d: 0,
    };
    next_addresses(&mut addr, thread_input, thread);

    let mut prev = BlockTh::load(mem_lane.add(QWORDS_IN_BLOCK).cast_const(), thread);
    let mut mem_curr = mem_lane.add(2 * QWORDS_IN_BLOCK);

    for offset in 2..segment_blocks {
        argon2_step_indexed(
            mem_lane.cast_const(),
            mem_curr,
            &mut prev,
            &mut addr,
            segment_blocks,
            thread,
            &mut thread_input,
            0,
            offset,
        );
        mem_curr = mem_curr.add(QWORDS_IN_BLOCK);
    }

    // Slice 1 restarts the address counter; the C++ bumps work-item 2's word and clears
    // work-item 6's so `next_addresses` sees slice 1 with a fresh counter.
    if thread == 2 {
        thread_input += 1;
    }
    if thread == 6 {
        thread_input = 0;
    }

    for offset in 0..segment_blocks {
        argon2_step_indexed(
            mem_lane.cast_const(),
            mem_curr,
            &mut prev,
            &mut addr,
            segment_blocks,
            thread,
            &mut thread_input,
            1,
            offset,
        );
        mem_curr = mem_curr.add(QWORDS_IN_BLOCK);
    }

    for slice in 2..ARGON2_SYNC_POINTS {
        for offset in 0..segment_blocks {
            argon2_step_dependent(
                mem_lane.cast_const(),
                mem_curr,
                &mut prev,
                segment_blocks,
                thread,
                slice,
                offset,
            );
            mem_curr = mem_curr.add(QWORDS_IN_BLOCK);
        }
    }
}
