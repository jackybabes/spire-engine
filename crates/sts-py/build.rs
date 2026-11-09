fn main() {
    // Lets the cdylib load into Python without maturin (`-undefined dynamic_lookup` on macOS).
    pyo3_build_config::add_extension_module_link_args();
}
