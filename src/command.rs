// SPDX-License-Identifier: GPL-3.0-only

use crate::tensor;
use std::ffi::OsStr;

pub fn embed(
    model_filename: &OsStr,
    message_filename: &OsStr,
    output_filename: &OsStr,
    bits_per_byte: usize,
) -> anyhow::Result<()> {
    let message = tensor::Message::load_from(message_filename)?;
    let mut owned_safe_tensors = tensor::OwnedSafeTensors::from_file(
        model_filename,
        bits_per_byte,
    )?;

    message.embed(&mut owned_safe_tensors)?;

    owned_safe_tensors.to_file(output_filename)?;

    println!(
        "Embedded {} bytes into {:?}.",
        message.len(),
        output_filename
    );

    Ok(())
}

pub fn extract(
    model_filename: &OsStr,
    output_filename: &OsStr,
    bits_per_byte: usize,
) -> anyhow::Result<()> {
    let mut owned_safe_tensors = tensor::OwnedSafeTensors::from_file(
        model_filename,
        bits_per_byte,
    )?;

    let message = tensor::Message::from_owned_safe_tensors(
        &mut owned_safe_tensors,
    )?;

    std::fs::write(output_filename, message.as_bytes())?;

    println!(
        "Extracted {} bytes into {:?}.",
        message.len(),
        output_filename
    );

    Ok(())
}

pub fn inspect(
    model_filename: &OsStr,
    bits_per_byte: usize,
) -> anyhow::Result<()> {
    let mut owned_safe_tensors = tensor::OwnedSafeTensors::from_file(
        model_filename,
        bits_per_byte,
    )?;

    let model_info = tensor::ModelInfo::from_owned_safe_tensors(
        &mut owned_safe_tensors,
    )?;

    println!("Model file {:?}:", model_filename);
    println!();
    println!("  Bytes:         {}", model_info.n_bytes);

    for bit_pos in (0..8).rev() {
        println!(
            "  Zero {}{} bits: {}",
            bit_pos + 1,
            match bit_pos + 1 {
                1 => "st",
                2 => "nd",
                3 => "rd",
                _ => "th",
            },
            model_info.n_zero_bits[bit_pos]
        );
    }

    println!();
    println!("  Assuming {} bits/byte:", model_info.bits_per_byte);
    println!();
    println!(
        "    Capacity:        {} bits",
        model_info.capacity * model_info.bits_per_byte
    );
    println!("    Message length:  {} bytes", model_info.length);
    println!("    Message content: {}", model_info.repr_truncated());
    println!();

    Ok(())
}
