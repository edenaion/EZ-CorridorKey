use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Fused BGR uint8 → RGB float32 conversion in a single pass.
///
/// Replaces cv2.cvtColor(frame, COLOR_BGR2RGB) + .astype(np.float32) / 255.0
#[pyfunction]
pub fn bgr_u8_to_rgb_f32<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let shape = input.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected 3-channel BGR image",
        ));
    }

    let in_slice = input.as_slice()?;
    let out = PyArray3::<f32>::zeros(py, [h, w, 3], false);
    let inv = 1.0_f32 / 255.0;

    // SAFETY: we just created this array and hold the GIL.
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(out.data() as *mut f32, h * w * 3);
        for i in 0..h * w {
            let src = i * 3;
            // BGR → RGB swap + uint8 → float32
            out_slice[src] = in_slice[src + 2] as f32 * inv;
            out_slice[src + 1] = in_slice[src + 1] as f32 * inv;
            out_slice[src + 2] = in_slice[src] as f32 * inv;
        }
    }

    Ok(out)
}

/// Fused RGB float32 → BGR uint8 conversion in a single pass.
///
/// Replaces cv2.cvtColor(img, COLOR_RGB2BGR) after (np.clip(...) * 255).astype(np.uint8)
#[pyfunction]
pub fn rgb_f32_to_bgr_u8<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let shape = input.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected 3-channel RGB image",
        ));
    }

    let in_slice = input.as_slice()?;
    let out = PyArray3::<u8>::zeros(py, [h, w, 3], false);

    // SAFETY: we just created this array and hold the GIL.
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(out.data() as *mut u8, h * w * 3);
        for i in 0..h * w {
            let src = i * 3;
            let r = (in_slice[src].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let g = (in_slice[src + 1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let b = (in_slice[src + 2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            // RGB → BGR swap
            out_slice[src] = b;
            out_slice[src + 1] = g;
            out_slice[src + 2] = r;
        }
    }

    Ok(out)
}
