//! Tools the model can call. For now just a calculator, exposed via the
//! `>>tool:calculate(...)` convention. Used both to generate correct training
//! traces and to execute tool calls at inference time.

pub mod calculator;
