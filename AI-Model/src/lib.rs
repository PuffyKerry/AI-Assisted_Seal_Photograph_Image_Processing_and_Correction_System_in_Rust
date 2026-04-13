// =============================================================================
// AI-Model library crate — exposes CNN inference for use by the web server
//
// The main binary (main.rs) handles training, demos, comparisons, etc.
// This lib.rs exposes only the inference-relevant modules so that the
// web_server crate can load a pre-trained model and run predictions without
// duplicating any code.
// =============================================================================

// Required for burn's derive macros (Module, Config) which generate deeply nested types
#![recursion_limit = "512"]
#![allow(non_snake_case)]  // iteration_2_CNN matches original module naming

pub mod iteration_2_CNN;


