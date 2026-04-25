use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// Convert FFmpeg gbrpf32le planar output to interleaved RGB float32.
///
/// Input: flat f32 array of length 3*H*W (plane order: G, B, R).
/// Output: [H, W, 3] f32 array in RGB order.
/// The scatter loop auto-vectorizes with LTO + codegen-units=1.
#[pyfunction]
pub fn gbr_planar_to_rgb<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let input = data.as_slice()?;
    let plane_size = height * width;

    if input.len() < 3 * plane_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input buffer too small for given dimensions",
        ));
    }

    let g_plane = &input[..plane_size];
    let b_plane = &input[plane_size..2 * plane_size];
    let r_plane = &input[2 * plane_size..3 * plane_size];

    let out = PyArray3::<f32>::zeros(py, [height, width, 3], false);
    // SAFETY: we just created this array and hold the GIL.
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(
            out.data() as *mut f32,
            3 * plane_size,
        );
        for i in 0..plane_size {
            let dst = i * 3;
            out_slice[dst] = r_plane[i];
            out_slice[dst + 1] = g_plane[i];
            out_slice[dst + 2] = b_plane[i];
        }
    }

    Ok(out)
}
