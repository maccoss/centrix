.PHONY: check build install test bench clean

check:
	cargo fmt
	cargo clippy --all-targets -- -D warnings

build: check
	cargo build --release

test: check
	cargo test

install: build
	@echo "Installing centrix to ~/.cargo/bin/"
	@cp target/release/centrix ~/.cargo/bin/centrix
	@echo "Installed: $$(centrix --version 2>/dev/null || echo 'centrix')"

bench:
	cargo bench

clean:
	cargo clean
