extern crate clap;
extern crate external_sort;
extern crate fine_grained;
extern crate rand;

use std::fs::File;
use std::io::{Read, BufReader, BufRead, Write};

use clap::{Arg, App};

use external_sort::conversions::{to_usize_saturating};

const MEGABYTE_SIZE : u64 = 1024 * 1024;

fn main() {
    let matches = App::new("validate")
        .version("0.1.0")
        .author("Drake Tetreault <drakeat@amazon.com>")
        .about("Validates output files for CodeDeploy reading group project 1")
        .arg(Arg::with_name("source")
            .value_name("SOURCE_PATH")
            .index(1)
            .help("Path to the file which will be checked for correctness; if not present then the generated file will be written to standard output"))
        .get_matches();

    if let Some(source_path) = matches.value_of("source") {
        let mut file = File::open(source_path).expect("Unable to create destination file");
        validate(&mut file);
    } else {
        let mut stdin = ::std::io::stdin();
        let mut stdin_lock = stdin.lock();
        validate(&mut stdin_lock);
    }
}

fn validate<TRead: Read>(reader: &mut TRead) {
    let reader = BufReader::with_capacity(to_usize_saturating(MEGABYTE_SIZE), reader);

    let stopwatch = fine_grained::Stopwatch::start_new();

    let mut prev_line = String::new();
    for line in reader.lines() {
        match line {
            Ok(curr_line) => {
                if !is_monotonically_increasing(&prev_line, &curr_line) {
                    writeln!(std::io::stderr(), "Found pair of lines that are not monotonically increasing: \"{}\" and \"{}\"", prev_line, curr_line).unwrap();
                    return;
                }
                prev_line.clear();
                prev_line.push_str(&curr_line);
            }
            Err(error) => {
                writeln!(std::io::stderr(), "Unable to read line: {}", error).unwrap();
                return;
            }
        }
    }

    writeln!(std::io::stderr(), "Elapsed time: {duration}", duration = stopwatch).unwrap();
}

fn is_monotonically_increasing(prev: &str, curr: &str) -> bool {
    prev <= curr
}
