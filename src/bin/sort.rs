extern crate clap;
extern crate external_sort;
extern crate fine_grained;

use std::cell::RefCell;
use std::cmp::{Ordering, Ord, Eq, PartialEq, PartialOrd};
use std::collections::binary_heap::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufRead, Write, BufWriter};
use std::path::{Path, PathBuf};
 
use clap::{App, Arg};

const MEGABYTE_SIZE : usize = 1024 * 1024;
const IO_BUFFER_SIZE : usize = 16 * MEGABYTE_SIZE;
const CHUNK_MAX_SIZE : usize = 256 * MEGABYTE_SIZE;
const MERGING_CHUNK_READ_BUFFER_COMBINED_MAX_SIZE : usize = 950 * MEGABYTE_SIZE;

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

    let timer = fine_grained::Stopwatch::start_new();
    let chunks = chunk_input(&source_path);
    println!("Time to chunk input: {}", to_seconds(&timer));
    
    let timer = fine_grained::Stopwatch::start_new();
    k_way_merge(&destination_path, &chunks);
    println!("Time to k-way merge: {}", to_seconds(&timer));
}

fn chunk_input(source_path: &Path) -> Vec<Chunk> {
    let file = File::open(source_path).expect("Unable to open source file");
    let mut reader = BufReader::with_capacity(IO_BUFFER_SIZE, file);

    let mut chunk_data : Vec<u8> = Vec::with_capacity(CHUNK_MAX_SIZE + MEGABYTE_SIZE);
    let mut word_data : Vec<Word> = Vec::with_capacity(100000);

    let mut curr_index = 0;
    let mut curr_entries_read = 0;
    let mut curr_bytes_read = 0;

    let mut completed_chunks = Vec::with_capacity(100);

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
            let timer = fine_grained::Stopwatch::start_new();
            // Sorting in place here allows us to reuse the same chunk_data buffer over and over (fewer allocations),
            // but this won't work if we want to offload the sorting to a different thread. Once we do add dual-threading,
            // we'll need to allocate a new buffer for each chunk, so that we can pass the ownership to the sorting thread.
            sort_chunk(&chunk_data[0..curr_bytes_read], &mut word_data[0..curr_entries_read]);
            println!("Chunk sort time: {}", to_seconds(&timer));
            let timer = fine_grained::Stopwatch::start_new();
            // Note that not all of the chunk_data is written.
            let curr_path = path_for_chunk(source_path, curr_index);
            write_chunk(&chunk_data[0..curr_bytes_read], &word_data[..curr_entries_read], &curr_path);
            println!("Chunk write time: {}", to_seconds(&timer));

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
        sort_chunk(&chunk_data[..curr_bytes_read], &mut word_data[..curr_entries_read]);

        // Note that not all of the chunk_data is written.
        let curr_path = path_for_chunk(source_path, curr_index);
        write_chunk(&chunk_data[..curr_bytes_read], &word_data[..curr_entries_read], &curr_path);

        // Once we switch to dual-threading, we'll need to push this to a queue and signal the other thread.
        completed_chunks.push(Chunk {
            path: curr_path,
            entries: curr_entries_read,
            bytes: curr_bytes_read
        });

        println!("Finished chunk {}, wrote {} entries ({} bytes total)", curr_index, curr_entries_read, curr_bytes_read);
    }

    completed_chunks
}

// Evenly split the total space for merge-phase read buffers among all chunk readers. The very
// last chunk may get more space than the rest.
fn compute_merge_read_buffer_sizes(chunk_count: usize) -> Vec<usize> {
    let even_split = {
        let split = MERGING_CHUNK_READ_BUFFER_COMBINED_MAX_SIZE / chunk_count;
        if split >= MEGABYTE_SIZE { split } else { MEGABYTE_SIZE }
    };
    let remainder = {
        let rem = MERGING_CHUNK_READ_BUFFER_COMBINED_MAX_SIZE - even_split * chunk_count;
        if rem < 0 { 0 } else { rem }
    };

    let mut sizes = Vec::with_capacity(chunk_count);
    // All except the last buffer get an even split.
    for _ in 0..chunk_count-1 {
        sizes.push(even_split);
    }
    // The very last buffer gets the remainder, if any.
    sizes.push(even_split + remainder);

    sizes
}

fn k_way_merge(destination_path: &Path, chunks: &Vec<Chunk>) {
    let file = File::create(destination_path).expect("Unable to create destination file");
    let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, file);
    
    let merging_chunk_buffer_sizes = compute_merge_read_buffer_sizes(chunks.len());
    let merging_chunks: Vec<RefCell<MergingChunk>> = chunks.iter()
        .zip(merging_chunk_buffer_sizes.iter())
        .map(|(chunk, read_buffer_size)| {
            let mut merging_chunk = MergingChunk::new(chunk, *read_buffer_size);
            // In addition to creating the MergingChunk, we also need to prime it by reading
            // the first line. But we ignore the result of reading the first line because
            // we simply don't add add an initial MergeCandidate for this MergingChunk to the
            // heap below if no line was present.
            let _ = merging_chunk.try_read_next_head();
            RefCell::new(merging_chunk)
        })
        .collect();

    let mut merge_candidates = {
        let mut heap = BinaryHeap::with_capacity(chunks.len());
        for chunk_index in 0..merging_chunks.len() {
            // Only add a MergeCandidate if this MergingChunk is non-empty.
            if !merging_chunks[chunk_index].borrow().head_data.is_empty() {
                heap.push(MergeCandidate {
                    chunk_index,
                    merging_chunk: &merging_chunks[chunk_index]
                });
            }
        }
        heap
    };

    // Continue until all merge candidates have been exhausted. Note that every time we take a candidate from
    // a MergingChunk, we also replace a candidate from that chunk with the next line, so this loop will not
    // exit until all chunks have been fully processed.
    while let Some(top_candidate) = merge_candidates.pop() {
        let merging_chunk = top_candidate.merging_chunk.borrow();
        writer.write_all(&merging_chunk.head_data)
            .expect("Unable to write top candidate from chunk into final output file");
        ::std::mem::drop(merging_chunk);

        let mut merging_chunk = top_candidate.merging_chunk.borrow_mut();
        let has_next_line = merging_chunk.try_read_next_head();
        ::std::mem::drop(merging_chunk);

        // If another line was read, then this merging chunk is not finished and we need to add
        // another MergeCandidate entry for it.
        if has_next_line {
            // Note: because of the interior mutability of the RefCell<MergingChunk>, we can reuse the
            // same MergeCandidate object here.
            merge_candidates.push(top_candidate);
        }
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
}

struct MergingChunk {
    reader: BufReader<File>,
    head_data: Vec<u8>
}

impl MergingChunk {
    fn new(chunk: &Chunk, read_buffer_size: usize) -> MergingChunk {
        let file = File::open(&chunk.path).expect("Unable to open chunk file for merging");
        let reader = BufReader::with_capacity(read_buffer_size, file);
        let head_data = Vec::with_capacity(1024);
        MergingChunk {
            reader,
            head_data
        }
    }

    fn try_read_next_head(&mut self) -> bool {
        self.head_data.clear();
        let bytes_read = self.reader.read_until('\n' as u8, &mut self.head_data).expect("Unable to read chunk file for merging");
        bytes_read > 0
    }
}

// Represents a candidate 'head' line that the merge step is considering. The actual line data is stored in
// the MergingChunk at merging_chunks[chunk_index]. Exactly one merge candidate exists for every MergingChunk.
struct MergeCandidate<'a> {
    chunk_index: usize,
    merging_chunk: &'a RefCell<MergingChunk>
}

// Necessary to store MergeCandidate in the heap.
impl<'a> Eq for MergeCandidate<'a> {}

// Necessary to store MergeCandidate in the heap.
impl<'a> PartialEq for MergeCandidate<'a> {
    fn eq(&self, other: &Self) -> bool {
        let self_merging_chunk = self.merging_chunk.borrow();
        let other_merging_chunk = other.merging_chunk.borrow();
        match compare_words(self_merging_chunk.head_data.as_slice(), other_merging_chunk.head_data.as_slice()) {
            Ordering::Equal => true,
            _ => false
        }
    }
}

// Necessary to store MergeCandidate in the heap. The final reverse() is due to the binary_heap being a max-heap by default.
// We need it to be a min heap so we reverse the ordering.
impl<'a> Ord for MergeCandidate<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_merging_chunk = self.merging_chunk.borrow();
        let other_merging_chunk = other.merging_chunk.borrow();
        compare_words(self_merging_chunk.head_data.as_slice(), other_merging_chunk.head_data.as_slice()).reverse()
    }
}

// Necessary to store MergeCandidate in the heap
impl<'a> PartialOrd for MergeCandidate<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn path_for_chunk(source_path: &Path, chunk_index: usize) -> PathBuf {
    let mut chunk_path = source_path.to_path_buf();
    chunk_path.set_extension(format!("chunk_{:04}", chunk_index));
    chunk_path
}

// Sorts data in place.
fn sort_chunk(chunk_data: &[u8], word_data: &mut [Word]) {
    word_data.sort_unstable_by(|lhs, rhs| {
        let lhs_range = lhs.offset..(lhs.offset + lhs.raw_length);
        let rhs_range = rhs.offset..(rhs.offset + rhs.raw_length);
        compare_words(&chunk_data[lhs_range], &chunk_data[rhs_range])
    });
}

// Tests whether the words 'lhs' and 'rhs' are lexicographically ordered.
fn compare_words(lhs: &[u8], rhs: &[u8]) -> Ordering {
    if lhs.len() < rhs.len() {
        return Ordering::Less;
    } 
    if lhs.len() > rhs.len() {
        return Ordering::Greater;
    }

    // Most significant byte is located at lowest offset.
    for i in 0..lhs.len() {
        let lhs_val = lhs[i];
        let rhs_val = rhs[i];
        if lhs_val < rhs_val {
            return Ordering::Less;
        }
        if lhs_val > rhs_val {
            return Ordering::Greater;
        }
    }

    Ordering::Equal
}

fn write_chunk(chunk_data: &[u8], word_data: &[Word], curr_path: &Path) {
    let curr_file = File::create(&curr_path).expect("Unable to create chunk file");
    let mut curr_writer = BufWriter::with_capacity(IO_BUFFER_SIZE, curr_file);
    for word in word_data {
        curr_writer.write_all(&chunk_data[word.offset..(word.offset + word.raw_length)])
            .expect("Failed to write chunk data");
    }
}

fn to_seconds(timer: &fine_grained::Stopwatch) -> f64 {
    const NANOS_PER_SECOND :f64 = 1000000000.0;
    timer.total_time() as f64 / NANOS_PER_SECOND
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use super::compare_words;
    use super::Word;

    #[test]
    fn compare_words_less() {
        let cases = vec![
            (vec![0], vec![1]),
            (vec![0, 1], vec![0, 2]),
            (vec![13, 22, 74], vec![10, 12, 44, 32]),
            (vec![50, 44, 10], vec![50, 44, 11])
        ];
        for (ref chunk_data, ref lhs, ref rhs) in cases {
            assert_eq!(Ordering::Less, compare_words(lhs, rhs));
        }
    }

    #[test]
    fn compare_words_greater() {
        let cases = vec![
            (vec![1], vec![0]),
            (vec![0, 2], vec![0, 1]),
            (vec![10, 12, 44, 32], vec![13, 22, 74]),
            (vec![50, 44, 10], vec![10, 44, 11])
        ];
        for (ref chunk_data, ref lhs, ref rhs) in cases {
            assert_eq!(Ordering::Greater, compare_words(chunk_data, lhs, rhs));
        }
    }

    #[test]
    fn compare_words_equal() {
        let cases = vec![
            (vec![1], vec![1]),
            (vec![0, 2], vec![0, 2]),
            (vec![10, 12, 44, 32], vec![10, 12, 44, 32]),
            (vec![50, 44, 10], vec![50, 44, 10])
        ];
        for (ref chunk_data, ref lhs, ref rhs) in cases {
            assert_eq!(Ordering::Equal, compare_words(chunk_data, lhs, rhs));
        }
    }
}