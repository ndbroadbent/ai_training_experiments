//! Backend selection.
//!
//! Default build = `ndarray` (CPU): fast to compile, great for iterating on
//! architecture and small sanity runs. Build with `--features gpu` to swap in
//! the `wgpu` backend, which runs on Metal on Apple Silicon, for real training.
//!
//! Everything downstream uses the type alias [`B`] (an `Autodiff`-wrapped
//! backend so gradients work) and the [`device`] helper, so switching backends
//! never touches model or training code.

use burn::backend::Autodiff;

#[cfg(not(feature = "gpu"))]
mod sel {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type Raw = NdArray;
    pub const NAME: &str = "ndarray (CPU)";
    pub fn raw_device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }
}

#[cfg(feature = "gpu")]
mod sel {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type Raw = Wgpu;
    pub const NAME: &str = "wgpu (Metal GPU)";
    pub fn raw_device() -> WgpuDevice {
        WgpuDevice::default()
    }
}

/// Human-readable name of the active backend.
pub use sel::NAME;

/// The raw (inference-only) backend currently selected at compile time.
pub type Raw = sel::Raw;

/// The differentiable backend used everywhere for training.
pub type B = Autodiff<Raw>;

/// The default compute device for the active backend.
pub fn device() -> <Raw as burn::tensor::backend::Backend>::Device {
    sel::raw_device()
}
