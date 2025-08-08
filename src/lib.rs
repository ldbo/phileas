use pyo3::prelude::*;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shuffle, m)?)?;
    Ok(())
}

/// Shuffle a `position` in a collection with size `bound`. The specific
/// permutation that is used is represented by the given key.
///
/// Internally, shuffling is done by cycle-walking a 3-round Feistel network
/// with SipHash 1-3 round function.
#[pyfunction]
fn shuffle(position: u128, bound: u128, keys: Vec<(u64, u64)>) -> PyResult<u128> {
    let half_width = bound.ilog2() / 2 + 1;

    let mut position = position;
    loop {
        position = generalized_feistel_network(position, half_width, &keys);
        if position < bound {
            return Ok(position);
        }
    }
}

fn generalized_feistel_network(m: u128, half_width: u32, keys: &Vec<(u64, u64)>) -> u128 {
    use siphasher::sip::SipHasher13;

    let mask = (1 << half_width) - 1 as u64;

    let mut m1 = (m as u64) & mask;
    let mut m2 = (m >> half_width) as u64;

    for (k0, k1) in keys.iter() {
        m1 = m1 ^ (SipHasher13::new_with_keys(*k0, *k1).hash(&m2.to_le_bytes()) & mask);
        m2 = m2 ^ (SipHasher13::new_with_keys(*k0, *k1).hash(&m1.to_le_bytes()) & mask);
    }

    ((m2 as u128) << half_width) + (m1 as u128)
}
