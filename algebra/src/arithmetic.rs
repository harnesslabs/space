use super::*;

pub trait Additive:
  Copy
  + Zero
  + Add<Output = Self>
  + Neg<Output = Self>
  + Sub<Output = Self>
  + AddAssign
  + SubAssign
  + PartialEq
  + Eq
{
}

pub trait Multiplicative:
  Copy + One + Mul<Output = Self> + Div<Output = Self> + MulAssign + DivAssign + PartialEq + Eq
{
}
