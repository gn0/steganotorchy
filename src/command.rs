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

    println!(
        "Model file {:?} ({} bits/byte):",
        model_filename, bits_per_byte
    );
    println!();
    println!("{}", model_info);

    Ok(())
}
