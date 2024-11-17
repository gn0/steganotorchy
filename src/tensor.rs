// SPDX-License-Identifier: GPL-3.0-only

use safetensors::tensor::Dtype;
use safetensors::{SafeTensors, View};
use std::borrow::Cow;
use std::fmt;
use std::io;
use std::io::{Read, Write};
use std::path::Path;
use std::str::FromStr;

use crate::bits::{BitIter, BitManip, FromBits};

pub struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl OwnedTensor {
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: Vec<u8>) -> Self {
        OwnedTensor { dtype, shape, data }
    }
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        (&self.data).into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

pub struct OwnedSafeTensors {
    tensors: Vec<(String, OwnedTensor, Option<usize>)>,
    bits_per_byte: usize,

    /// The total number of F16/BF16/F32/F64 parameters in `tensors`
    /// that can be used to store a header and a message.
    capacity: usize,

    /// The index of the next parameter that can be used to encode a
    /// message.  The three dimensions index the following:
    ///
    /// 1. Tensors in `tensors`.
    /// 2. Elements in `tensors.data`.
    /// 3. Bits in the last byte of an element.
    next_pos: Option<(usize, usize, usize)>,
}

impl OwnedSafeTensors {
    pub fn from_file<P>(
        path: P,
        bits_per_byte: usize,
    ) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        let content = std::fs::read(&path)?;
        let model = SafeTensors::deserialize(&content)?;

        let mut labels: Vec<_> =
            model.iter().map(|(label, _)| label).collect();

        labels.sort();

        let tensors: Vec<_> = labels
            .into_iter()
            .map(|label| {
                let view = model.tensor(label).unwrap();
                let owned_tensor = OwnedTensor::new(
                    view.dtype(),
                    view.shape().to_vec(),
                    view.data().to_vec(),
                );
                let dtype = owned_tensor.dtype;

                (label.to_string(), owned_tensor, dtype_to_size(dtype))
            })
            .collect();

        let capacity = tensors
            .iter()
            .map(|(_, owned_tensor, dtype_size)| match dtype_size {
                Some(x) => owned_tensor.data.len() / x,
                None => 0,
            })
            .sum();

        let next_pos = if capacity > 0 {
            Some((0, 0, 8 - bits_per_byte))
        } else {
            None
        };

        Ok(OwnedSafeTensors {
            tensors,
            bits_per_byte,
            capacity,
            next_pos,
        })
    }

    pub fn to_file<P>(self, path: P) -> anyhow::Result<()>
    where
        P: AsRef<Path>,
    {
        safetensors::tensor::serialize_to_file(
            self.tensors
                .into_iter()
                .map(|(label, owned_tensor, _)| (label, owned_tensor)),
            &None,
            path.as_ref(),
        )?;

        Ok(())
    }

    pub fn tensors(&self) -> &[(String, OwnedTensor, Option<usize>)] {
        &self.tensors
    }

    pub fn reset_pos(&mut self) {
        self.next_pos = Some((0, 0, 8 - self.bits_per_byte));
    }
}

macro_rules! bump_to_full_byte {
    ($next_pos:expr, $bits_per_byte:expr) => {
        let &(tensor_pos, element_pos, bit_pos) = $next_pos
            .as_ref()
            .expect("Next position should always be set.");

        if bit_pos > 8 - $bits_per_byte {
            // The contents that were read from or written into `buffer`
            // did not exhaust the last byte.  We advance to the next
            // byte.
            //
            $next_pos =
                Some((tensor_pos, element_pos + 1, 8 - $bits_per_byte));
        }
    };
}

impl io::Read for OwnedSafeTensors {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let mut length = 0;
        let mut bits = Vec::with_capacity(8);

        while length < buffer.len() {
            let Some(&(tensor_pos, element_pos, bit_pos)) =
                self.next_pos.as_ref()
            else {
                // No next position.
                //
                break;
            };

            let Some((_, owned_tensor, dtype_size)) =
                self.tensors.get(tensor_pos)
            else {
                // Next position points to invalid owned tensor.
                //
                self.next_pos = None;
                break;
            };

            let Some(dtype_size) = dtype_size else {
                // Next position points to unsuitable owned tensor.  We
                // advance to the next owned tensor.
                //
                self.next_pos =
                    Some((tensor_pos + 1, 0, 8 - self.bits_per_byte));
                continue;
            };

            let byte_pos = (element_pos + 1) * *dtype_size - 1;

            let Some(byte) = owned_tensor.data.get(byte_pos) else {
                // Next position points to invalid byte.  We advance to
                // the next owned tensor.
                //
                self.next_pos =
                    Some((tensor_pos + 1, 0, 8 - self.bits_per_byte));

                continue;
            };

            let bit = byte
                .extract_bit(bit_pos)
                .expect("Bit position should be valid.");

            bits.push(bit);

            if bits.len() == 8 {
                buffer[length] = u8::from_bits(&bits).unwrap();

                length += 1;

                bits.clear();
            }

            if bit_pos < 7 {
                // We advance to the next bit.
                //
                self.next_pos =
                    Some((tensor_pos, element_pos, bit_pos + 1));
            } else {
                // Current byte is now exhausted.  We advance to the
                // next byte.
                //
                self.next_pos = Some((
                    tensor_pos,
                    element_pos + 1,
                    8 - self.bits_per_byte,
                ));
            }
        }

        if !bits.is_empty() {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Content is incomplete, {} bits are unprocessed.",
                    bits.len()
                ),
            ))
        } else {
            bump_to_full_byte!(self.next_pos, self.bits_per_byte);

            Ok(length)
        }
    }
}

impl io::Write for OwnedSafeTensors {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let mut error = None;

        'outer: for bit in BitIter::new(buffer) {
            loop {
                let Some(&(tensor_pos, element_pos, bit_pos)) =
                    self.next_pos.as_ref()
                else {
                    // No next position.
                    //
                    error = Some(io::ErrorKind::StorageFull);
                    break 'outer;
                };

                let Some((_, owned_tensor, dtype_size)) =
                    self.tensors.get_mut(tensor_pos)
                else {
                    // Next position points to invalid owned tensor.
                    //
                    self.next_pos = None;
                    error = Some(io::ErrorKind::StorageFull);
                    break 'outer;
                };

                let Some(dtype_size) = dtype_size else {
                    // Next position points to unsuitable owned tensor.
                    // We advance to the next owned tensor.
                    //
                    self.next_pos = Some((
                        tensor_pos + 1,
                        0,
                        8 - self.bits_per_byte,
                    ));
                    continue;
                };

                let byte_pos = (element_pos + 1) * *dtype_size - 1;

                let Some(byte) = owned_tensor.data.get_mut(byte_pos)
                else {
                    // Next position points to invalid byte.  We advance
                    // to the next owned tensor.
                    //
                    self.next_pos = Some((
                        tensor_pos + 1,
                        0,
                        8 - self.bits_per_byte,
                    ));
                    continue;
                };

                let Some(modified_byte) = byte.embed_bit(bit, bit_pos)
                else {
                    // Couldn't embed the bit.
                    //
                    self.next_pos = None;
                    error = Some(io::ErrorKind::StorageFull);
                    break 'outer;
                };

                *byte = modified_byte;

                if bit_pos < 7 {
                    // We advance to the next bit.
                    //
                    self.next_pos =
                        Some((tensor_pos, element_pos, bit_pos + 1));
                } else {
                    // Current byte is now exhausted.  We advance to the
                    // next byte.
                    //
                    self.next_pos = Some((
                        tensor_pos,
                        element_pos + 1,
                        8 - self.bits_per_byte,
                    ));
                }

                break;
            }
        }

        match error {
            Some(err) => Err(io::Error::from(err)),
            None => {
                bump_to_full_byte!(self.next_pos, self.bits_per_byte);

                Ok(buffer.len())
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub struct ModelInfo {
    capacity: usize,
    bits_per_byte: usize,
    length: u32,
    truncated: Vec<u8>,
}

impl ModelInfo {
    pub fn from_owned_safe_tensors(
        owned_safe_tensors: &mut OwnedSafeTensors,
    ) -> anyhow::Result<Self> {
        let mut header = [0; 4];
        owned_safe_tensors.read_exact(&mut header)?;

        let length = u32::from_be_bytes(header);

        let mut truncated = vec![0; std::cmp::min(length as usize, 21)];
        owned_safe_tensors.read_exact(&mut truncated)?;

        Ok(Self {
            capacity: owned_safe_tensors.capacity,
            bits_per_byte: owned_safe_tensors.bits_per_byte,
            length,
            truncated,
        })
    }
}

impl fmt::Display for ModelInfo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "  Capacity:         {} bits",
            self.capacity * self.bits_per_byte
        )?;
        writeln!(
            formatter,
            "  Message length:   {} bytes",
            self.length
        )?;
        write!(formatter, "  Message content:  \"")?;

        for (index, &byte) in self.truncated.iter().enumerate() {
            if index == 20 {
                write!(formatter, "...")?;
                break;
            } else if byte as char == '"' {
                write!(formatter, r#"\""#)?;
            } else {
                let repr = &format!("{:?}", byte as char);
                let repr = repr.strip_prefix("'").unwrap_or(repr);
                let repr = repr.strip_suffix("'").unwrap_or(repr);

                write!(formatter, "{}", repr)?;
            }
        }

        writeln!(formatter, r#"""#)?;

        Ok(())
    }
}

pub struct Message {
    content: Vec<u8>,
}

impl Message {
    pub fn new(content: &[u8]) -> Self {
        Message {
            content: content.to_vec(),
        }
    }

    pub fn load_from<P>(path: P) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        Ok(Self::new(&std::fs::read(&path)?))
    }

    pub fn get_bit(
        &self,
        byte_pos: usize,
        bit_pos: usize,
    ) -> Option<bool> {
        let byte = self.content.get(byte_pos)?;

        byte.extract_bit(bit_pos)
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.content
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Extract a message from the bits of `owned_safe_tensors`.
    ///
    /// The message is embedded in two sections:
    ///
    /// 1. HEADER    [64 bits long]
    /// 2. CONTENT   [its length in bytes equals HEADER]
    ///
    /// For example, with `bits_per_byte` == 1:
    ///
    /// - If the content is only a single byte long, then it can be
    ///   embedded using 72 elements in the input data,
    ///   `owned_safe_tensors`.  The length of the content (1 byte) is
    ///   encoded in the first 64 elements, and the content takes up
    ///   eight additional elements.
    ///
    /// - If the content is 1 KB long, then it can be embedded using
    ///   8,256 elements.  The length of the content (1,024 bytes) is
    ///   encoded in the first 64 elements, and the content takes up
    ///   another 8,192 elements.
    pub fn from_owned_safe_tensors(
        owned_safe_tensors: &mut OwnedSafeTensors,
    ) -> anyhow::Result<Self> {
        let mut header = [0; 4];
        owned_safe_tensors.read_exact(&mut header)?;

        let length = u32::from_be_bytes(header);

        let mut content = vec![0; length as usize];
        owned_safe_tensors.read_exact(&mut content)?;

        Ok(Self::new(&content))
    }

    /// Embed the message into the bits of `owned_safe_tensors`.
    pub fn embed(
        &self,
        owned_safe_tensors: &mut OwnedSafeTensors,
    ) -> anyhow::Result<()> {
        let bits_per_byte = owned_safe_tensors.bits_per_byte;

        let params_for_header =
            (8 * std::mem::size_of::<u32>()).div_ceil(bits_per_byte);
        let params_for_content =
            (8 * self.content.len()).div_ceil(bits_per_byte);
        let params_needed = params_for_header + params_for_content;

        let params_available = owned_safe_tensors.capacity;

        anyhow::ensure!(
            params_needed <= params_available,
            "Message needs at least {} parameters to be embedded \
             but tensors have only {}.",
            params_needed,
            params_available
        );

        let header = u32::try_from(self.content.len())?.to_be_bytes();

        owned_safe_tensors.write_all(&header)?;
        owned_safe_tensors.write_all(&self.content)?;

        Ok(())
    }
}

impl FromStr for Message {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> anyhow::Result<Message> {
        let content = value.as_bytes().to_vec();

        Ok(Message { content })
    }
}

impl<'a> TryFrom<&'a Message> for &'a str {
    type Error = anyhow::Error;

    fn try_from(message: &'a Message) -> anyhow::Result<&'a str> {
        Ok(std::str::from_utf8(&message.content)?)
    }
}

fn dtype_to_size(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::F16 | Dtype::BF16 => Some(2),
        Dtype::F32 => Some(4),
        Dtype::F64 => Some(8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_BYTE_STRINGS: &[&[u8]] =
        &[b"", b"foo", b"foo\r\nbar\r\n", b"\x12\x13\x14"];

    static TEST_STRS: &[&str] =
        &["", "foo", "foo\r\nbar\r\n", "\x12\x13\x14"];

    #[test]
    fn as_bytes_works() {
        // Test that &[u8] -> Message -> &[u8] == &[u8].
        //
        for &x in TEST_BYTE_STRINGS {
            assert_eq!(Message::new(x).as_bytes(), x);
        }
    }

    #[test]
    fn from_str_works() {
        // Test that &str -> Message -> &[u8] == &str -> &[u8].
        //
        for &x in TEST_STRS {
            assert_eq!(
                Message::from_str(x).unwrap().as_bytes(),
                x.as_bytes()
            );
        }
    }

    #[test]
    fn str_try_from_works() {
        // Test that &[u8] -> Message -> &str == &[u8] -> &str.
        //
        for &x in TEST_BYTE_STRINGS {
            assert_eq!(
                <&str>::try_from(&Message::new(x)).unwrap(),
                std::str::from_utf8(x).unwrap()
            );
        }
    }

    #[rustfmt::skip]
    fn make_owned_safe_tensors(
        bits_per_byte: usize
    ) -> OwnedSafeTensors {
        // Convenient values for testing bit writes and reads:
        //
        // 128 == 0b1000_0000 (parameter value)
        // 255 == 0b1111_1111 (payload to embed in parameters)
        //
        // Values of parameters that have been written to:
        //
        // 129 == 0b1000_0001 -> lowest bit is written to
        // 144 == 0b1001_0000 -> 5th lowest bit is written to
        // 156 == 0b1001_1100 -> ...
        // 159 == 0b1001_1111 -> the 5 lowest bits are written to
        //

        let tensor_1 = OwnedTensor::new(
            Dtype::F32,
            vec![2, 3],
            vec![128, 128, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128, 128, 128, 128, 128],
        );

        let tensor_2 = OwnedTensor::new(
            Dtype::U32,
            vec![2, 3],
            vec![128, 128, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128, 128, 128, 128, 128],
        );

        let tensor_3 = OwnedTensor::new(
            Dtype::F16,
            vec![6, 3],
            vec![128, 128, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128],
        );

        OwnedSafeTensors {
            tensors: vec![
                ("a".to_string(), tensor_1, Some(4)),
                ("b".to_string(), tensor_2, None),
                ("c".to_string(), tensor_3, Some(2)),
            ],
            bits_per_byte,
            capacity: 24,
            next_pos: Some((0, 0, 8 - bits_per_byte)),
        }
    }

    #[test]
    #[rustfmt::skip]
    fn owned_safe_tensors_write_1_bit() {
        let mut input = make_owned_safe_tensors(1);

        assert!(
            input.write_all(&[255, 255])  // 255 == 0b1111_1111
                .is_ok()
        );

        assert_eq!(
            &*input.tensors()[0].1.data(),
            &[128, 128, 128, 129,         // 129 == 0b1000_0001
              128, 128, 128, 129,
              128, 128, 128, 129,
              128, 128, 128, 129,
              128, 128, 128, 129,
              128, 128, 128, 129]
        );

        assert!(input.tensors()[1].1.data().iter().all(|x| *x == 128));

        assert_eq!(
            &*input.tensors()[2].1.data(),
            &[128, 129, 128, 129,
              128, 129, 128, 129,
              128, 129, 128, 129,
              128, 129, 128, 129,
              128, 129, 128, 129,
              128, 128, 128, 128,
              128, 128, 128, 128,
              128, 128, 128, 128,
              128, 128, 128, 128]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn owned_safe_tensors_write_5_bit() {
        let mut input = make_owned_safe_tensors(5);

        assert!(input.write_all(&[255, 255]).is_ok());

        assert_eq!(
            &*input.tensors()[0].1.data(),
            &[128, 128, 128, 159,  // 159 == 0b1001_1111
              128, 128, 128, 159,
              128, 128, 128, 159,
              128, 128, 128, 144,  // 144 == 0b1001_0000
              128, 128, 128, 128,
              128, 128, 128, 128]
        );

        assert!(input.tensors()[1].1.data().iter().all(|x| *x == 128));
        assert!(input.tensors()[2].1.data().iter().all(|x| *x == 128));
    }

    #[test]
    #[rustfmt::skip]
    fn owned_safe_tensors_write_inserts_padding() {
        let mut input = make_owned_safe_tensors(5);

        assert!(input.write_all(&[255]).is_ok());
        assert!(input.write_all(&[255]).is_ok());

        assert_eq!(
            &*input.tensors()[0].1.data(),
            &[128, 128, 128, 159,  // 159 == 0b1001_1111
              128, 128, 128, 156,  // 156 == 0b1001_1100
              128, 128, 128, 159,
              128, 128, 128, 156,
              128, 128, 128, 128,
              128, 128, 128, 128]
        );

        assert!(input.tensors()[1].1.data().iter().all(|x| *x == 128));
        assert!(input.tensors()[2].1.data().iter().all(|x| *x == 128));
    }

    #[test]
    fn owned_safe_tensors_read() {
        for bits_per_byte in 1..=8 {
            let mut input = make_owned_safe_tensors(bits_per_byte);
            let mut buffer = [0; 2];

            assert!(input.write_all(&[255, 255]).is_ok());

            input.reset_pos();

            assert!(input.read_exact(&mut buffer).is_ok());

            assert_eq!(buffer, [255, 255]);
        }
    }

    #[test]
    fn owned_safe_tensors_read_expects_padding() {
        for bits_per_byte in 3..=7 {
            let mut input = make_owned_safe_tensors(bits_per_byte);
            let mut buffer_1 = [0; 1];
            let mut buffer_2 = [0; 1];

            assert!(input.write_all(&[255]).is_ok());
            assert!(input.write_all(&[255]).is_ok());

            input.reset_pos();

            assert!(input.read_exact(&mut buffer_1).is_ok());
            assert!(input.read_exact(&mut buffer_2).is_ok());

            assert_eq!(buffer_1, [255]);
            assert_eq!(buffer_2, [255]);
        }
    }
}
