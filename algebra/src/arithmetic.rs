use super::*;

pub trait Additive: Copy + Add<Output = Self> + AddAssign + PartialEq + Eq {}

pub trait Multiplicative: Copy + Mul<Output = Self> + MulAssign + PartialEq + Eq {}
