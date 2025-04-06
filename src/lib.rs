mod cli;

use anyhow::{Context, Result};
use cli::Cli;

pub fn run_app() -> Result<()> {
    let args = cli::parse_args();
    
    match args.command {
        cli::Commands::Greet { name } => {
            greet(&name)?;
        }
        cli::Commands::Count { to } => {
            count(to)?;
        }
    }
    
    Ok(())
}

fn greet(name: &str) -> Result<()> {
    println!("Hello, {}!", name);
    Ok(())
}

fn count(to: u32) -> Result<()> {
    println!("Counting to {}:", to);
    for i in 1..=to {
        println!("{}", i);
    }
    Ok(())
}
