extern crate clap;
extern crate fine_grained;
extern crate rand;

use std::io::{Write, BufWriter};
use std::fs::File;

use clap::{Arg, App};
use rand::{Rng, SeedableRng, XorShiftRng};
use rand::os::OsRng;
use rand::distributions::{IndependentSample, Range};

const LINE_LENGTH_WITH_NEWLINE : usize = 128;
const MEGABYTE_SIZE : u64 = 1024 * 1024;

fn main() {
    let matches = App::new("generate")
        .version("0.1.0")
        .author("Drake Tetreault <drakeat@amazon.com>")
        .about("Generates test files for CodeDeploy reading group project 1")
        .arg(Arg::with_name("size")
            .short("s")
            .long("size")
            .value_name("SIZE")
            .help("Size of the output to generate in megabytes; defaults to 1024 if not provided")
            .takes_value(true))
        .arg(Arg::with_name("destination")
            .short("d")
            .long("destination")
            .value_name("OUTPUT_PATH")
            .help("Path to the output file; if not present then the generated file will be written to standard output")
            .takes_value(true))
        .get_matches();

    let size_in_megabytes : u64 = matches.value_of("size").map_or(1024u64, |size| {
        size.parse::<u64>().expect("Invalid size value")
    });

    if let Some(destination) = matches.value_of("destination") {
        let mut file = File::create(destination).expect("Unable to create destination file");
        let mut writer = BufWriter::with_capacity(to_usize_saturating(MEGABYTE_SIZE), file);
        generate(size_in_megabytes, &mut writer);
    } else {
        let mut stdout = std::io::stdout();
        let mut stdout_lock = stdout.lock();
        let mut writer = BufWriter::with_capacity(to_usize_saturating(MEGABYTE_SIZE), stdout_lock);
        generate(size_in_megabytes, &mut writer);
    }
}

fn generate<TWrite : Write>(size_in_megabytes: u64, writer: &mut TWrite) {
    let mut rng = create_prng();
    let mut line = create_blank_line_with_newline();
    let letter_range = Range::new('a' as u8, 'z' as u8);

    let stopwatch = fine_grained::Stopwatch::start_new();

    let lines_needed = size_in_megabytes * MEGABYTE_SIZE / to_u64_panicking(LINE_LENGTH_WITH_NEWLINE);

    for _ in 0..lines_needed {
        fill_line(&mut line[0..LINE_LENGTH_WITH_NEWLINE - 1], &mut rng, &letter_range);
        writer.write(&line).expect("Failed to write line");
    }
    
    writeln!(std::io::stderr(), "Elapsed time: {duration}", duration = stopwatch).unwrap();
}

fn create_prng() -> XorShiftRng {
    let mut seeding_rng = OsRng::new().expect("Failed to get initial random entropy from OS source");
    let seed = [
        seeding_rng.next_u32(),
        seeding_rng.next_u32(),
        seeding_rng.next_u32(),
        seeding_rng.next_u32()
    ];
    XorShiftRng::from_seed(seed)
}

fn create_blank_line_with_newline() -> [u8; LINE_LENGTH_WITH_NEWLINE] {
    let mut line = ['!' as u8; LINE_LENGTH_WITH_NEWLINE];
    line[LINE_LENGTH_WITH_NEWLINE - 1] = '\n' as u8;
    line
}

fn fill_line<TRng: Rng>(line: &mut [u8], rng: &mut TRng, range: &Range<u8>) {
    for entry in line.iter_mut() {
        *entry = range.ind_sample(rng);
    }
}

fn to_usize_saturating(num: u64) -> usize {
    if (num as usize as u64) < num { 
        std::usize::MAX
    } else {
        num as usize
    }
}

fn to_u64_panicking(num: usize) -> u64 {
    if (num as u64 as usize) != num {
        panic!("Cannot losslessly convert usize value {} to u64", num);
    } else {
        num as u64
    }
}