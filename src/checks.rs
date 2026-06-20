//! Minimal sanity checks that the chosen backend computes and differentiates.

use crate::backend::{device, B, NAME};
use burn::tensor::Tensor;

/// Compute f(x) = sum(x^2) and its gradient (df/dx = 2x) to confirm that the
/// backend, tensor ops, and autodiff are all wired up correctly.
pub fn run_hello() {
    let device = device();
    println!("Backend: {NAME}");

    let x = Tensor::<B, 1>::from_floats([2.0, 3.0, 4.0], &device).require_grad();
    let y = x.clone().powf_scalar(2.0).sum();

    let grads = y.backward();
    let dx = x.grad(&grads).expect("gradient should exist for a require_grad tensor");

    println!("x           = {x}");
    println!("y = sum(x^2) = {y}");
    println!("dy/dx        = {dx}   (expected 2x = [4, 6, 8])");
}
