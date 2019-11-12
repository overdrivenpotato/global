use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use global::Immutable;
use std::time::Duration;

fn run_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple");
    group.bench_function("million_gets", |b| b.iter(|| {
        let n = black_box(1_000_000);
        (0..n).for_each(|_| {
            static N: Immutable<i32> = Immutable::new();
            &*N;
        });
    }));
    group.finish();

    let mut group = c.benchmark_group("throughput");
    // one access per iteration
    group.throughput(Throughput::Elements(1));
    group.bench_function("get_global", |b| {
        b.iter(|| {
            static N: Immutable<i32> = Immutable::new();
            &*N;
        })
    });
    group.finish();
}

fn main() {
    let mut c = Criterion::default()
        // bump warm-up time from 3s to prevent false regression reports
        .warm_up_time(Duration::from_secs_f32(7.5));

    run_benchmarks(&mut c);
}
