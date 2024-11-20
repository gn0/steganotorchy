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

#[derive(Debug, PartialEq)]
pub struct Ternary {
    data: Vec<bool>,
}

impl Ternary {
    pub fn bits(&self) -> &[bool] {
        &self.data
    }
}

impl TryFrom<Vec<bool>> for Ternary {
    type Error = anyhow::Error;

    fn try_from(data: Vec<bool>) -> anyhow::Result<Ternary> {
        anyhow::ensure!(data.len() % 2 == 0);
        anyhow::ensure!(data.len() >= 2);
        anyhow::ensure!(data[data.len() - 1] == true);
        anyhow::ensure!(data[data.len() - 2] == true);

        Ok(Ternary { data })
    }
}

impl From<usize> for Ternary {
    fn from(number: usize) -> Self {
        let n_trits =
            usize::try_from(number.checked_ilog(3).unwrap_or(0))
                .expect("usize should be at least 32 bits wide.");

        let mut ternary = Vec::with_capacity(n_trits + 1);

        ternary.push(vec![true, true]); // End-of-record marker.

        let mut remainder = number;

        while remainder > 0 {
            match remainder % 3 {
                0 => ternary.push(vec![false, false]),
                1 => ternary.push(vec![false, true]),
                2 => ternary.push(vec![true, false]),

                // The exhaustiveness checker is getting confused here
                // so we help it out.
                3.. => unreachable!(),
            }

            remainder /= 3;
        }

        ternary.reverse();

        Ternary {
            data: ternary.into_iter().flatten().collect(),
        }
    }
}

impl TryFrom<&Ternary> for usize {
    type Error = anyhow::Error;

    fn try_from(ternary: &Ternary) -> anyhow::Result<usize> {
        anyhow::ensure!(ternary.data.len() % 2 == 0);

        let mut number = 0;

        let a_iter = ternary
            .data
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 2 == 0)
            .map(|(_, x)| x);
        let b_iter = ternary
            .data
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 2 == 1)
            .map(|(_, x)| x);

        for (a, b) in a_iter.zip(b_iter) {
            let trit = match (a, b) {
                (false, false) => 0,
                (false, true) => 1,
                (true, false) => 2,
                (true, true) => break, // End-of-record marker.
            };

            anyhow::ensure!(number <= (usize::MAX - trit) / 3);

            number *= 3;
            number += trit;
        }

        Ok(number)
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

    #[test]
    fn ternary_conversion_roundtrip_preserves_value() {
        for number in 0..1024 {
            assert_eq!(
                usize::try_from(&Ternary::from(number)).unwrap(),
                number
            );
        }
    }

    #[test]
    fn ternary_from_is_correct() {
        for (number, ternary) in [
            (0, vec![true, true]),
            (1, vec![false, true, true, true]),
            (2, vec![true, false, true, true]),
            (3, vec![false, true, false, false, true, true]),
            (4, vec![false, true, false, true, true, true]),
            (5, vec![false, true, true, false, true, true]),
            (6, vec![true, false, false, false, true, true]),
        ] {
            assert_eq!(Ternary::from(number).data, ternary);
        }
    }

    #[test]
    fn ternary_catches_overflow() {
        let mut ternary_too_large: Ternary = usize::into(usize::MAX);

        assert!(usize::try_from(&ternary_too_large).is_ok());

        // Add 1.
        let n_ternary_too_large = ternary_too_large.data.len();
        ternary_too_large.data[n_ternary_too_large - 3] = true;

        assert!(usize::try_from(&ternary_too_large).is_err());
    }
}
