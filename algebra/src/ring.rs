use super::*;

pub trait Ring: Zero + One + Add + Neg + Sub + Mul + AddAssign + SubAssign + MulAssign {}

pub trait Field: Ring + Div + DivAssign {}
