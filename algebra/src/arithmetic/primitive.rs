use super::*;

// Implement Additive for all primitive numeric types
impl Additive for u8 {}
impl Additive for u16 {}
impl Additive for u32 {}
impl Additive for u64 {}
impl Additive for u128 {}
impl Additive for usize {}

impl Additive for i8 {}
impl Additive for i16 {}
impl Additive for i32 {}
impl Additive for i64 {}
impl Additive for i128 {}
impl Additive for isize {}

impl Additive for f32 {}
impl Additive for f64 {}

// Implement Multiplicative for all primitive numeric types
impl Multiplicative for u8 {}
impl Multiplicative for u16 {}
impl Multiplicative for u32 {}
impl Multiplicative for u64 {}
impl Multiplicative for u128 {}
impl Multiplicative for usize {}

impl Multiplicative for i8 {}
impl Multiplicative for i16 {}
impl Multiplicative for i32 {}
impl Multiplicative for i64 {}
impl Multiplicative for i128 {}
impl Multiplicative for isize {}

impl Multiplicative for f32 {}
impl Multiplicative for f64 {}

// Implement Infinity for float primitive numeric types
impl Infinity for f32 {
  const INFINITY: Self = f32::INFINITY;
}

impl Infinity for f64 {
  const INFINITY: Self = f64::INFINITY;
}
