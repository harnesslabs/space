use std::ops::{Add, Mul, Neg, Sub};

use num::{One, Zero};

pub trait Ring: Zero + One + Add + Neg + Sub + Mul {}
