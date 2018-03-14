pub fn to_usize_saturating(num: u64) -> usize {
    if (num as usize as u64) < num { 
        ::std::usize::MAX
    } else {
        num as usize
    }
}

pub fn to_u64_panicking(num: usize) -> u64 {
    if (num as u64 as usize) != num {
        panic!("Cannot losslessly convert usize value {} to u64", num);
    } else {
        num as u64
    }
}