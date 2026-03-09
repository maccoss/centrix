// Benchmark: full two-pass centroid_spectrum().
// Target: < 3 ms per spectrum single-threaded.
// Implemented in Phase 4.

use criterion::{criterion_group, criterion_main, Criterion};

fn centroid_benchmark(_c: &mut Criterion) {
    // Phase 4 stub
}

criterion_group!(benches, centroid_benchmark);
criterion_main!(benches);
