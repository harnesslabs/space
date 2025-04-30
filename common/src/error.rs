#[derive(Clone, Debug)]
pub enum MathError {
    DimensionMismatch,
    CWUninitialized,
    InvalidCellIdx,
    InvalidDataIdx,
    NullMatrix,
    NoPointFound,
    NotAdjacent,
    NoRestrictionDefined,
    BadField,
    ImproperBases,
    BadCochain,
    NoCellsofDimensionK,
    NotInvertible,
}