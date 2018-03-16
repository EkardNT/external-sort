extern crate clap;
extern crate external_sort;

use std::fs::File;
use std::io::{Read, BufReader, BufRead, Write, BufWriter};
use std::path::{Path, PathBuf};

use clap::{App, Arg};

const MEGABYTE_SIZE : usize = 1024 * 1024;
const READ_BUFFER_SIZE : usize = 32 * MEGABYTE_SIZE;
const WRITE_BUFFER_SIZE : usize = 32 * MEGABYTE_SIZE;
const CHUNK_MAX_SIZE : usize = 256 * MEGABYTE_SIZE;

fn main() {
    let current_dir = ::std::env::current_dir();
    if let Err(err) = current_dir {
        writeln!(::std::io::stderr(), "Unable to determine current directory: {}", err).unwrap();
        ::std::process::exit(1);
    }
    let current_dir = current_dir.unwrap();

    let matches = App::new("sort")
        .version("0.1.0")
        .author("Drake Tetreault <drakeat@amazon.com>")
        .about("Sorts large text files for CodeDeploy reading group project 1")
        .arg(Arg::with_name("source")
            .value_name("SOURCE_PATH")
            .index(1)
            .help("Path to the input file which will be sorted")
            .required(true))
        .arg(Arg::with_name("destination")
            .value_name("DESTINATION_PATH")
            .index(2)
            .help("Path where the final sorted output file will be written")
            .required(true))
        .get_matches();

    let mut source_path = current_dir.clone();
    source_path.push(Path::new(matches.value_of("source").expect("No source path provided")));

    let mut destination_path = current_dir.clone();
    destination_path.push(Path::new(matches.value_of("destination").expect("No destination path provided")));

    chunk_input(&source_path);
}

fn chunk_input(source_path: &Path) {
    let file = File::open(source_path).expect("Unable to open source file");
    let mut reader = BufReader::with_capacity(READ_BUFFER_SIZE, file);

    let mut chunk_data : Vec<u8> = Vec::with_capacity(CHUNK_MAX_SIZE + MEGABYTE_SIZE);
    let mut word_data : Vec<Word> = Vec::with_capacity(100000);

    let mut curr_index = 0;
    let mut curr_entries_read = 0;
    let mut curr_bytes_read = 0;

    let mut completed_chunks = Vec::with_capacity(50);

    loop {
        let bytes_read = reader.read_until('\n' as u8, &mut chunk_data).expect("Failure while reading from source file");

        // Treat 0 bytes read as EOF.
        if bytes_read == 0 {
            break;
        }

        assert!(bytes_read <= MEGABYTE_SIZE, "A word cannot exceed one megabyte in size, including the trailing newline byte");

        // If adding the next line would exceed the max chunk size, rotate to the next chunk. Note that the bytes from
        // the word that exceeded the max size have already been added to the chunk_data and need to be preserved
        // during the rotation.
        if curr_entries_read > 0 && bytes_read + curr_bytes_read > CHUNK_MAX_SIZE {
            // Sorting in place here allows us to reuse the same chunk_data buffer over and over (fewer allocations),
            // but this won't work if we want to offload the sorting to a different thread. Once we do add dual-threading,
            // we'll need to allocate a new buffer for each chunk, so that we can pass the ownership to the sorting thread.
            sort_chunk(&mut chunk_data[0..curr_bytes_read], &word_data[0..curr_entries_read]);

            // Note that not all of the chunk_data is written.
            let curr_path = path_for_chunk(source_path, curr_index);
            write_chunk(&chunk_data[0..curr_bytes_read], &curr_path);

            // Once we switch to dual-threading, we'll need to push this to a queue and signal the other thread.
            completed_chunks.push(Chunk {
                path: curr_path,
                entries: curr_entries_read,
                bytes: curr_bytes_read
            });

            println!("Finished chunk {}, wrote {} entries ({} bytes total)", curr_index, curr_entries_read, curr_bytes_read);

            // Now drain off the previous chunk's bytes and word data.
            chunk_data.drain(..curr_bytes_read);
            word_data.clear();

            // Switch to new chunk.
            curr_index += 1;
            curr_bytes_read = 0;
            curr_entries_read = 0;
        }

        // Record the read word. Its important to do this after any necessary chunk swap has happened because otherwise the
        // offset record in the word may be incorrect.
        word_data.push(Word::new(curr_bytes_read, bytes_read));

        curr_bytes_read += bytes_read;
        curr_entries_read += 1;
    }

    // Complete the last chunk.
    if curr_entries_read > 0 {
        // Sorting in place here allows us to reuse the same chunk_data buffer over and over (fewer allocations),
        // but this won't work if we want to offload the sorting to a different thread. Once we do add dual-threading,
        // we'll need to allocate a new buffer for each chunk, so that we can pass the ownership to the sorting thread.
        sort_chunk(&mut chunk_data[..curr_bytes_read], &word_data[..curr_entries_read]);

        // Note that not all of the chunk_data is written.
        let curr_path = path_for_chunk(source_path, curr_index);
        write_chunk(&chunk_data[..curr_bytes_read], &curr_path);

        // Once we switch to dual-threading, we'll need to push this to a queue and signal the other thread.
        completed_chunks.push(Chunk {
            path: curr_path,
            entries: curr_entries_read,
            bytes: curr_bytes_read
        });
    }
}

struct Chunk {
    path: PathBuf,
    entries: usize,
    bytes: usize
}

struct Word {
    offset: usize,
    raw_length: usize
}

impl Word {
    fn new(offset: usize, raw_length: usize) -> Word {
        assert!(raw_length >= 1, "Raw length cannot be because each word's raw length must cover \
            the mandatory '\\n' byte, even if the word is otherwise of length 0");
        Word {
            offset, raw_length
        }
    }

    fn data_length(&self) -> usize {
        self.raw_length - 1
    }
}

fn path_for_chunk(source_path: &Path, chunk_index: usize) -> PathBuf {
    let mut chunk_path = source_path.to_path_buf();
    chunk_path.set_extension(format!("chunk_{:04}", chunk_index));
    chunk_path
}

// Sorts data in place.
fn sort_chunk(chunk_data: &mut [u8], word_data: &[Word]) {

}

fn write_chunk(chunk_data: &[u8], curr_path: &Path) {
    let curr_file = File::create(&curr_path).expect("Unable to create chunk file");
    let mut curr_writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, curr_file);
    curr_writer.write_all(chunk_data).expect("Failed to write chunk data");
}