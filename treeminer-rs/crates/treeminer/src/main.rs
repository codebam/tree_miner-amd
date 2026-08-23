//! Entry point. Parses arguments, resolves the configuration, and hands off to the miner.

use std::process::ExitCode;

use treeminer::{init_local_offset, resolve, Cli, ResolveOptions, StdioPrompter};

fn main() -> ExitCode {
    // FIRST, before anything can spawn a thread: the local UTC offset can only be read
    // while the process is single-threaded (see `clock`).
    init_local_offset();

    // The hash CLI is a separate tool sharing this binary, exactly as in the C++; it must
    // be recognised before the mining options are parsed.
    let args: Vec<String> = std::env::args().collect();
    if treeminer::is_hash_api_command(&args) {
        return treeminer::run_hash_cli(&args);
    }

    let cli = Cli::from_env();
    let options = ResolveOptions::from_process();
    let mut prompter = StdioPrompter;

    let config = match resolve(&cli, &options, &mut prompter) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(warning) = &config.display_warning {
        eprintln!("{warning}");
    }
    for message in &config.startup_messages {
        println!("{message}");
    }

    ExitCode::from(treeminer::run(&config))
}
