use anyhow::Result;
use minifb::{Window, WindowOptions, Key, ScaleMode};
use nokhwa::{utils::{CameraIndex, RequestedFormat, RequestedFormatType}, pixel_format::RgbFormat, Camera};
use opencv::{prelude::*, imgcodecs::imwrite, core::{Vector, CV_8UC3}};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

fn main() -> Result<()>{
    let index = CameraIndex::Index(0); 
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    
    let mut camera = Camera::new(index, requested)?;

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "webcam",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            // borderless: true,
            // transparency: false,
            // title: false,
            // none: true,
            // topmost:false,
            ..WindowOptions::default()
        },
    )?;

    window.set_position(100, 50);

    window.limit_update_rate(Some(std::time::Duration::from_millis(1000/30)));

    let mut decoded = None;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // let t = Instant::now();
        let frame = camera.frame()?;
        // decode into an ImageBuffer
        let decoded_frame = frame.decode_image::<RgbFormat>()?;
        
        for (pixel, target) in decoded_frame.pixels().zip(buffer.iter_mut()){
            *target = u32::from_be_bytes([0, pixel[0], pixel[1], pixel[2]]);
        }

        decoded = Some(decoded_frame);
        
        window.update_with_buffer(&buffer, WIDTH, HEIGHT)?;
    }

    if let Some(decoded) = decoded{
        let img = rgb_bytes_to_mat(&decoded, decoded.width(), decoded.height())?;
        imwrite("frame.png", &img, &Vector::<i32>::default())?;
    }

    Ok(())
}

fn rgb_bytes_to_mat(rgb_data: &[u8], width: u32, height: u32) -> Result<Mat> {
    let mut rgb_data: Vec<u8> = rgb_data.to_vec();
    // rgb 转 bgr
    for pixel in rgb_data.chunks_mut(3) {
        pixel.reverse();
    }

    // Create a cv::Mat with the same width, height and type as the RGB image
    let mut mat = Mat::new_rows_cols_with_default(height as i32, width as i32, CV_8UC3, opencv::core::Scalar::all(0.0))?;

    // Copy the RGB bytes to the cv::Mat
    let rgb_slice = &rgb_data[..];
    let mat_slice = mat.data_mut();
    unsafe{ mat_slice.copy_from(rgb_slice.as_ptr(), rgb_slice.len()); }

    // Return the cv::Mat
    Ok(mat)
}