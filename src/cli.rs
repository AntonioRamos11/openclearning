use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Greet someone
    Greet {
        /// Name of the person to greet
        name: String,
    },
    /// Count up to a number
    Count {
        /// The number to count to
        to: u32,
    },
}

pub fn parse_args() -> Cli {
    Cli::parse()
}
