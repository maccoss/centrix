// Benchmark: non-negative LASSO solver.
// Target: < 0.1 ms per solve for n_basis=30, active_set=3.
// Implemented in Phase 3.

use criterion::{criterion_group, criterion_main, Criterion};

fn lasso_benchmark(_c: &mut Criterion) {
    // Phase 3 stub
}

criterion_group!(benches, lasso_benchmark);
criterion_main!(benches);
