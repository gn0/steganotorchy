use clap::{Parser, Subcommand};
use std::ffi::OsString;

use steganotorchy::command;

#[derive(Debug, Clone, Subcommand)]
enum Command {
    Embed {
        message_filename: OsString,
        output_filename: OsString,
    },
    Extract {
        output_filename: OsString,
    },
    Inspect,
}

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(long, short)]
    model_filename: OsString,

    #[arg(
        long,
        short,
        default_value_t = 1,
        value_parser = clap::value_parser!(u8).range(1..=8)
      )]
    bits_per_byte: u8,

    #[command(subcommand)]
    command: Command,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match &args.command {
        Command::Embed {
            message_filename,
            output_filename,
        } => command::embed(
            &args.model_filename,
            message_filename,
            output_filename,
            args.bits_per_byte as usize,
        ),
        Command::Extract { output_filename } => command::extract(
            &args.model_filename,
            output_filename,
            args.bits_per_byte as usize,
        ),
        Command::Inspect => command::inspect(
            &args.model_filename,
            args.bits_per_byte as usize,
        ),
    }
}
