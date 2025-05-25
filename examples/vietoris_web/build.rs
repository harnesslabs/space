use std::process::Command;

fn main() {
  // Only build WASM when the web feature is enabled
  if cfg!(feature = "web") {
    println!("cargo:rerun-if-changed=examples/vietoris_web.rs");

    // Check if wasm-pack is available
    if let Ok(output) = Command::new("wasm-pack").arg("--version").output() {
      if output.status.success() {
        println!("cargo:warning=Building WASM module with wasm-pack...");

        // Build the WASM package for the web target
        let build_result = Command::new("wasm-pack")
          .args(&[
            "build",
            "--target",
            "web",
            "--out-dir",
            "target/wasm-pkg",
            "--release",
            "--",
            "--bin",
            "vietoris_web",
            "--features",
            "web",
          ])
          .status();

        match build_result {
          Ok(status) if status.success() => {
            println!("cargo:warning=WASM module built successfully!");
          },
          Ok(_) => {
            println!("cargo:warning=WASM build failed - some features may not work");
          },
          Err(e) => {
            println!("cargo:warning=Failed to run wasm-pack: {}", e);
          },
        }
      } else {
        println!("cargo:warning=wasm-pack found but not working correctly");
      }
    } else {
      println!("cargo:warning=wasm-pack not found - install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh");
      println!("cargo:warning=The web demo will not be fully functional without WASM compilation");
    }
  }
}
