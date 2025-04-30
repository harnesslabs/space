pub use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use num_traits::{One, Zero};

pub trait Additive: Copy + Add<Output = Self> + AddAssign + PartialEq + Eq {}

pub trait Multiplicative: Copy + Mul<Output = Self> + MulAssign + PartialEq + Eq {}
