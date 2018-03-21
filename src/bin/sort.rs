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
// Note this is number of lines, not number of bytes.
const CHUNK_MAX_LINES : usize = 10 * 1024 * 1024;
// 26 lowercase letters possible, takes up 5 bits because 2^5 = 32. 2^4 = 16 is too small.
const BITS_PER_LETTER : usize = 5;
const BITS_PER_BYTE : usize = 8;
// How many letters can fit in a usize on this platform? This will be 12 for 64-bit or 6 for 32-bit.
const PREFIX_MAX_LETTERS : usize = std::mem::size_of::<usize>() * BITS_PER_BYTE / BITS_PER_LETTER;

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
        .arg(Arg::with_name("bucket")
            .short("b")
            .long("bucket")
            .help("If present, allows bucket sorting to be used on inputs where every word is the same size"))
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

    let allow_bucket_sort = matches.is_present("bucket");

    let timer = fine_grained::Stopwatch::start_new();
    let chunks = chunk_input(&source_path, allow_bucket_sort);
    println!("Time to chunk input: {}", to_seconds(&timer));
    
    let timer = fine_grained::Stopwatch::start_new();
    k_way_merge(&destination_path, &chunks);
    println!("Time to k-way merge: {}", to_seconds(&timer));
}

fn chunk_input(source_path: &Path, allow_bucket_sort: bool) -> Vec<Chunk> {
    let file = File::open(source_path).expect("Unable to open source file");
    let mut reader = BufReader::with_capacity(IO_BUFFER_SIZE, file);

    let mut chunk_data : Vec<u8> = Vec::with_capacity(CHUNK_MAX_SIZE + MEGABYTE_SIZE);
    let mut word_data : Vec<Word> = Vec::with_capacity(100000);

    let mut curr_index = 0;
    let mut curr_entries_read = 0;
    let mut curr_bytes_read = 0;
    let mut line_min_length = std::usize::MAX;
    let mut line_max_length = std::usize::MIN;

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
        if curr_entries_read > 0 && (bytes_read + curr_bytes_read > CHUNK_MAX_SIZE || curr_entries_read + 1 > CHUNK_MAX_LINES) {
            let timer = fine_grained::Stopwatch::start_new();
            // Sorting in place here allows us to reuse the same chunk_data buffer over and over (fewer allocations),
            // but this won't work if we want to offload the sorting to a different thread. Once we do add dual-threading,
            // we'll need to allocate a new buffer for each chunk, so that we can pass the ownership to the sorting thread.
            sort_chunk(
                &chunk_data[0..curr_bytes_read], 
                &mut word_data[0..curr_entries_read],
                line_min_length,
                line_max_length,
                allow_bucket_sort);
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
                bytes: curr_bytes_read,
                line_min_length,
                line_max_length
            });

            println!("Finished chunk {}, wrote {} entries ({} bytes total)", curr_index, curr_entries_read, curr_bytes_read);

            // Now drain off the previous chunk's bytes and word data.
            chunk_data.drain(..curr_bytes_read);
            word_data.clear();

            // println!("Exiting after first chunk");
            // std::process::exit(1);

            // Switch to new chunk.
            curr_index += 1;
            curr_bytes_read = 0;
            curr_entries_read = 0;
            line_max_length = std::usize::MIN;
            line_min_length = std::usize::MAX;
        }

        // Record the read word. Its important to do this after any necessary chunk swap has happened because otherwise the
        // offset record in the word may be incorrect.
        let prefix_value = get_prefix_value(&chunk_data, curr_bytes_read, bytes_read);
        let proxmap_bucket_index = proxmap_bucket_index_for_8192_buckets(prefix_value);
        word_data.push(Word::new(curr_bytes_read, bytes_read, proxmap_bucket_index, prefix_value));

        curr_bytes_read += bytes_read;
        curr_entries_read += 1;

        if bytes_read > line_max_length {
            line_max_length = bytes_read;
        }
        if bytes_read < line_min_length {
            line_min_length = bytes_read;
        }
    }

    // Complete the last chunk.
    if curr_entries_read > 0 {
        // Sorting in place here allows us to reuse the same chunk_data buffer over and over (fewer allocations),
        // but this won't work if we want to offload the sorting to a different thread. Once we do add dual-threading,
        // we'll need to allocate a new buffer for each chunk, so that we can pass the ownership to the sorting thread.
        sort_chunk(
            &chunk_data[..curr_bytes_read],
            &mut word_data[..curr_entries_read], 
            line_min_length, 
            line_max_length,
            allow_bucket_sort);

        // Note that not all of the chunk_data is written.
        let curr_path = path_for_chunk(source_path, curr_index);
        write_chunk(&chunk_data[..curr_bytes_read], &word_data[..curr_entries_read], &curr_path);

        // Once we switch to dual-threading, we'll need to push this to a queue and signal the other thread.
        completed_chunks.push(Chunk {
            path: curr_path,
            entries: curr_entries_read,
            bytes: curr_bytes_read,
            line_min_length,
            line_max_length
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
        let total_allocated = even_split * chunk_count;
        if MERGING_CHUNK_READ_BUFFER_COMBINED_MAX_SIZE <= total_allocated {
            0
        } else {
            MERGING_CHUNK_READ_BUFFER_COMBINED_MAX_SIZE - even_split * chunk_count
        }
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
    // Path to the temporary file containing the chunk data.
    path: PathBuf,
    // Number of entries in the file.
    entries: usize,
    // Number of bytes in the file.
    bytes: usize,
    // Total length; includes the trailing '\n'.
    line_max_length: usize,
    // Total length; includes the trailing '\n'.
    line_min_length: usize
}

#[derive(Clone, Copy, Debug)]
struct Word {
    offset: usize,
    raw_length: usize,
    proxmap_bucket_index: usize,
    prefix_value: usize
}

impl Word {
    fn new(offset: usize, raw_length: usize, proxmap_bucket_index: usize, prefix_value: usize) -> Word {
        assert!(raw_length >= 1, "Raw length cannot be because each word's raw length must cover \
            the mandatory '\\n' byte, even if the word is otherwise of length 0");
        Word {
            offset,
            raw_length,
            proxmap_bucket_index,
            prefix_value
        }
    }

    fn unset() -> Word {
        Word {
            offset: std::usize::MAX,
            raw_length: std::usize::MAX,
            proxmap_bucket_index: std::usize::MAX,
            prefix_value: std::usize::MAX
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
fn sort_chunk(
        chunk_data: &[u8],
        word_data: &mut [Word],
        line_min_length: usize,
        line_max_length: usize,
        allow_bucket_sort: bool) {
    // For chunks where every line is the same length, we can use a special O(n) sort based on looking at the
    // prefix of each line as an integer. I think this case should be able to also work for chunks where the
    // difference between line lengths is <= the maximum number of bytes that can be stored in a u64. As each
    // char holds 26 distinct values, a char corresponds to 5 bits (2^5 = 32), and floor(64 / 5) = 12, so the
    // maximum spread possible should be 12 characters. For simplicity I'm not going to implement that now
    // because the problem files are known to generate lines of the same length.
    // NOTE: get_prefix_value will need to be made smarter if we want to allow for non-identically-sized word
    // sorting.
    if allow_bucket_sort && line_min_length == line_max_length {
        bucket_sort(chunk_data, word_data, line_min_length, line_max_length)
    } else {
        // Otherwise, use the built-in O(n log(n)) comparison sort.
        word_data.sort_unstable_by(|lhs, rhs| {
            if lhs.prefix_value < rhs.prefix_value {
                Ordering::Less
            } else if lhs.prefix_value > rhs.prefix_value {
                Ordering::Greater
            } else {
                let lhs_range = lhs.offset..(lhs.offset + lhs.raw_length);
                let rhs_range = rhs.offset..(rhs.offset + rhs.raw_length);
                compare_words(&chunk_data[lhs_range], &chunk_data[rhs_range])
            }
        });
    }
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

fn bucket_sort(chunk_data: &[u8], word_data: &mut [Word], line_min_length: usize, line_max_length: usize) {
    const BUCKET_COUNT : usize = 8192;

    // 10 * 1024 * 1024 words max / 8192 buckets = 1280 expected words/bucket on average.
    let mut hit_counts = [0usize; BUCKET_COUNT];
    for word in word_data.iter() {
        hit_counts[word.proxmap_bucket_index] += 1;
    }

    // Compute prox maps aka proximity maps aka the index in the final array where each bucket begins.
    let mut prox_maps = [std::usize::MAX; BUCKET_COUNT];
    let mut running_sum = 0;
    for i in 0..BUCKET_COUNT {
        if hit_counts[i] > 0 {
            prox_maps[i] = running_sum;
            running_sum += hit_counts[i];
        }
    }

    let mut sorted_words = vec![Word::unset(); word_data.len()];

    // Write each word into its bucket. The difference from proxmap sort is we don't sort while inserting;
    // instead we wait to do the per-bucket sort as the next step.
    for word in word_data.iter() {
        let insertion_index = prox_maps.get_mut(word.proxmap_bucket_index).unwrap();
        sorted_words[*insertion_index] = *word;
        *insertion_index = *insertion_index + 1;
    }

    // Sort each bucket
    for i in 0..BUCKET_COUNT {
        if hit_counts[i] > 0 {
            let start = prox_maps[i] - hit_counts[i];
            let end = prox_maps[i];
            sorted_words[start..end].sort_unstable_by(|lhs, rhs| {
                if lhs.prefix_value < rhs.prefix_value {
                    Ordering::Less
                } else if lhs.prefix_value > rhs.prefix_value {
                    Ordering::Greater
                } else {
                    let lhs_range = lhs.offset..(lhs.offset + lhs.raw_length);
                    let rhs_range = rhs.offset..(rhs.offset + rhs.raw_length);
                    compare_words(&chunk_data[lhs_range], &chunk_data[rhs_range])
                }
            });
        }
    }

    // Copy sorted words to output.
    for i in 0..sorted_words.len() {
        word_data[i] = sorted_words[i];
    }
}

fn proxmap_bucket_index_for_8192_buckets(word_prefix_value: usize) -> usize {
    // To map a word to a bucket, we'll take the first 13 (2^13=8192) most significant bits from 
    // that words prefix.
    const BITS_FOR_8192 : usize = 13;
    const BITS_FOR_PREFIX : usize = PREFIX_MAX_LETTERS * BITS_PER_LETTER;
    assert!(BITS_FOR_PREFIX > BITS_FOR_8192);
    // Note that the prefix doesn't take up the full 64/32 bits necessarily, so its not right to
    // calculate SHIFT_RIGHT_AMOUNT as 64 - 13. Instead, its (12 * 5) - 13 == 60 - 13 == 47.
    const SHIFT_RIGHT_AMOUNT : usize = BITS_FOR_PREFIX - BITS_FOR_8192;

    // NOTE: at this time, the maximum value for bucket_index is 6606 rather than 8191. This is
    // because the value of 'z' in when converted to binary is '11001' rather than all-ones (11111).
    // Thus, the buckets from 6607 to 8191 will never have any keys assigned.
    let bucket_index = word_prefix_value >> SHIFT_RIGHT_AMOUNT;
    
    // TODO: test the performance with and without fully using all 8192 buckets. With this division,
    // all buckets are used so the insertion sort phase works on smaller buckets, but on the other hand
    // floating point math may be more expensive. Maybe its better to just eat the cost of the unused
    // buckets?
    let bucket_index = (bucket_index as f32) / 6606f32;
    let bucket_index = (8191f32 * bucket_index).round() as usize;
    
    bucket_index
}

// Gets the prefix value for a word. The prefix value is the value of the most significant letters
// for that word, not including the trailing '\n'. This function can handle words of less than the
// number of letters that can fit inside a machine word (aka 12 for 64 bit, 6 for 32 bit).
fn get_prefix_value(chunk_data: &[u8], word_offset: usize, word_raw_length: usize) -> usize {
    // Remember to subtract 1 from word.raw_length due to the trailing '\n', which is not included
    // in the prefix.
    let prefix_length = std::cmp::min(PREFIX_MAX_LETTERS, word_raw_length - 1);
    assert!(prefix_length > 0);

    // Prefix value holder.
    let mut prefix_value = 0;

    // We're going to successively build the prefix_value by shifting in the value of each letter
    // in the prefix, starting with the most significant byte in the prefix and ending with the least.
    for msb_offset in 0..prefix_length {
        let left_shift_amount = 5 * (prefix_length - msb_offset - 1);
        // Note that the actual bytes in chunk_data are the UTF-8 letter codepoints, so adjust them so
        // that a == 0, b == 1, ... z = 25.

        let letter_val = (chunk_data[word_offset + msb_offset] as u8) - ('a' as u8);

        prefix_value |= (letter_val as usize) << left_shift_amount;
    }

    prefix_value
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use super::{compare_words, get_prefix_value};
    use super::Word;

    #[test]
    fn compare_words_less() {
        let cases = vec![
            (vec![0], vec![1]),
            (vec![0, 1], vec![0, 2]),
            (vec![13, 22, 74], vec![10, 12, 44, 32]),
            (vec![50, 44, 10], vec![50, 44, 11])
        ];
        for (ref lhs, ref rhs) in cases {
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
        for (ref lhs, ref rhs) in cases {
            assert_eq!(Ordering::Greater, compare_words(lhs, rhs));
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
        for (ref lhs, ref rhs) in cases {
            assert_eq!(Ordering::Equal, compare_words(lhs, rhs));
        }
    }

    #[cfg(all(unix, target_pointer_width = "32"))]
    #[test]
    fn get_prefix_value_32bit() {
        unimplemented!();
    }

    #[cfg(all(unix, target_pointer_width = "64"))]
    #[test]
    fn get_prefix_value_64bit() {
        let cases : Vec<(Vec<u8>, usize, usize, usize)> = vec![
            (vec!['a' as u8, 'b' as u8, 'c' as u8, '\n' as u8], 0, 4,
                0b00000_00001_00010),
            (vec!['a' as u8, 'a' as u8, 'a' as u8, '\n' as u8], 0, 4,
                0b00000_00000_00000),
            (vec!['z' as u8, 'z' as u8, 'z' as u8, '\n' as u8], 0, 4,
                0b11001_11001_11001),
            (vec!['c' as u8, 'b' as u8, 'a' as u8, '\n' as u8], 0, 4,
                0b00010_00001_00000),
            (vec![
                // "zemblazonmentz\n"
                'z' as u8, 'e' as u8, 'm' as u8, 'b' as u8, 'l' as u8, 'a' as u8, 'z' as u8, 'o' as u8,
                'n' as u8, 'm' as u8, 'e' as u8, 'n' as u8, 't' as u8, 'z' as u8, '\n' as u8
            ], 1, 14,
                //e     m     b     l     a     z     o     n     m     e     n     t
                0b00100_01100_00001_01011_00000_11001_01110_01101_01100_00100_01101_10011),
            (vec![
                'a' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'z' as u8,
                'a' as u8,
                '\n' as u8
            ], 1, 14,
                //z     z     z     z     z     z     z     z     z     z     z     z
                0b11001_11001_11001_11001_11001_11001_11001_11001_11001_11001_11001_11001)
        ];
        for (data, offset, raw_length, expected_value) in cases {
            assert_eq!(expected_value, get_prefix_value(&data, offset, raw_length));
        }
    }
}