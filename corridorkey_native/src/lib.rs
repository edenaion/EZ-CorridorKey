use pyo3::prelude::*;

mod plane_shuffle;
mod color_convert;

#[pymodule]
fn corridorkey_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plane_shuffle::gbr_planar_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::bgr_u8_to_rgb_f32, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::rgb_f32_to_bgr_u8, m)?)?;
    Ok(())
}
