// SPDX-License-Identifier: GPL-3.0-only

type ByteIndex = usize;
type BitIndex = usize;

static BYTE_MASKS: &[u8] = &[
    0b1000_0000,
    0b0100_0000,
    0b0010_0000,
    0b0001_0000,
    0b0000_1000,
    0b0000_0100,
    0b0000_0010,
    0b0000_0001,
];

pub trait BitManip {
    type Type;

    fn extract_bit(self, pos: usize) -> Option<bool>;
    fn embed_bit(self, bit: bool, pos: usize) -> Option<Self::Type>;
}

impl BitManip for u8 {
    type Type = u8;

    fn extract_bit(self, pos: usize) -> Option<bool> {
        let mask = BYTE_MASKS.get(pos)?;

        Some(self & mask != 0)
    }

    fn embed_bit(self, bit: bool, pos: usize) -> Option<u8> {
        let mask = BYTE_MASKS.get(pos)?;

        Some((self & !mask) + (bit as u8) * mask)
    }
}

pub trait FromBits {
    type Type;

    fn from_bits(bits: &[bool]) -> Option<Self::Type>;
}

impl FromBits for u8 {
    type Type = u8;

    fn from_bits(bits: &[bool]) -> Option<u8> {
        if bits.len() > 8 {
            None
        } else {
            let mut byte = 0;

            for (index, bit) in bits.iter().copied().rev().enumerate() {
                byte += (bit as u8) << index;
            }

            Some(byte)
        }
    }
}

impl FromBits for u64 {
    type Type = u64;

    fn from_bits(bits: &[bool]) -> Option<u64> {
        if bits.len() > 64 {
            None
        } else {
            let mut number = 0;

            for (index, &bit) in bits.iter().rev().enumerate() {
                number += (bit as u64) << index;
            }

            Some(number)
        }
    }
}

pub struct BitIter<'msg> {
    message: &'msg [u8],
    next: Option<(ByteIndex, BitIndex)>,
}

impl<'msg> BitIter<'msg> {
    pub fn new(message: &'msg [u8]) -> Self {
        let next = if message.is_empty() {
            None
        } else {
            Some((0, 0))
        };

        BitIter { message, next }
    }
}

impl Iterator for BitIter<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let (byte_pos, bit_pos) = self.next?;

        match self
            .message
            .get(byte_pos)
            .and_then(|byte| byte.extract_bit(bit_pos))
        {
            x @ Some(_) => {
                self.next = if bit_pos < 7 {
                    Some((byte_pos, bit_pos + 1))
                } else {
                    Some((byte_pos + 1, 0))
                };

                x
            }
            None => {
                self.next = None;

                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_extract_bit() {
        assert_eq!(u8::extract_bit(0, 7), Some(false));
        assert_eq!(u8::extract_bit(1, 7), Some(true));
        assert_eq!(u8::extract_bit(2, 7), Some(false));

        assert_eq!(u8::extract_bit(0, 6), Some(false));
        assert_eq!(u8::extract_bit(1, 6), Some(false));
        assert_eq!(u8::extract_bit(2, 6), Some(true));
    }

    #[test]
    fn u8_from_bits() {
        assert_eq!(Some(0), u8::from_bits(&[false]));
        assert_eq!(Some(1), u8::from_bits(&[true]));
        assert_eq!(Some(2), u8::from_bits(&[true, false]));
        assert_eq!(Some(3), u8::from_bits(&[true, true]));

        assert_eq!(Some(0), u8::from_bits(&[false; 8]));
        assert_eq!(None, u8::from_bits(&[false; 9]));
    }

    #[test]
    fn u64_from_bits() {
        assert_eq!(Some(0), u64::from_bits(&[false]));
        assert_eq!(Some(1), u64::from_bits(&[true]));
        assert_eq!(Some(2), u64::from_bits(&[true, false]));
        assert_eq!(Some(3), u64::from_bits(&[true, true]));

        assert_eq!(Some(0), u64::from_bits(&[false; 64]));
        assert_eq!(None, u64::from_bits(&[false; 65]));
    }

    #[test]
    fn bit_iter_for_all_zeros() {
        assert_eq!(
            &BitIter::new(&[0, 0]).collect::<Vec<_>>(),
            &[false; 16]
        );
    }

    #[test]
    fn bit_iter_for_all_ones() {
        assert_eq!(
            &BitIter::new(&[1, 1]).collect::<Vec<_>>(),
            &[
                false, false, false, false, false, false, false, true,
                false, false, false, false, false, false, false, true
            ]
        );
    }
}
