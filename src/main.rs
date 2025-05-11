use clap::error::RichFormatter;
use ocl::{core::Mem, Buffer, MemFlags, ProQue, Result};
use std::time::{Duration, Instant};  // Added Duration import

use plotters::prelude::*; // Importing plotters for plotting


// This is a simple OpenCL program that creates a buffer, writes data to it, and runs a kernel.
// It uses the ocl crate to handle OpenCL operations in Rust.
// The kernel is a simple function that takes a buffer and writes to it.

//                Open cl                      hip 
//G Cores           (get group id,blockIdx )    __ocl_get_group_id(0),threadgrouposition_in_grid)
//L Threads        (get local id,threadIdx)  __ocl_get_local_id(0),threadposition_in_block)

//128 cores  , 1 threads 
//64 croes  , 2 threads


//nob leasson
//3080 has 84 sms , ga102 , 10752 shader units
// 4090 has 144 sms ,3
//g104m has 40sms , 5120 shader units  ,128 threads per block
//on Nvidia cores are SM cores ,streaming multiprocessors ,AMD are CU cores compute units

//Gpus has warps , warps are groups threads moderm gpus has warps of 32 threads

//SIMD are single instruction multiple data , 32 threads in a warp execute the same instruction at the same time
//vector register 
// float <32> 1024 bits
// c=a+b on vector register, this is a single instruction on 32 pieces of data

//SIMT single instruction multiple thread
//  similar to SIMD but threads are not in lockstep, but load/stores are difreerent
//  load stores are implicict scater gather , delcare- where simds its explicit
//  you only declare float but  behind the scenes its a float32

//gpus are multicores processors with 32 threads 


//on apple  m3 max hjas 40 core gpu ,640 executions units ,5120 "ALus"


    //proque.set_dims(1024);
    //let a_data = vec![1.0f32; data_size];
    //let b_data = vec![2.0f32; data_size];
    
    // Create buffers with the same size as our data
    //let a_buf = proque.create_buffer::<f32>()?;
    //let b_buf = proque.create_buffer::<f32>()?;


  //let c_buf = proque.create_buffer::<f32>()?;

    //a_buf.cmd().write(&a_data).enq()?;
    //b_buf.cmd().write(&b_data).enq()?;

//100000000000
fn main() -> Result<()> {
    // Define the size of our data
    let data_size: i32 = 262144;
    
    
    //this kernel takes Kernel execution time: execcute   ,8.083µs round to 8ms

    //1e6 takes Kernel execution time: 5.224µs
 
    let kernel_src: &str = r#"kernel void add (global float  *c) {
        int a  =get_local_id(0);
        for (int i = 0; i < 100000000; i++) {
            a*=2;
         }
        c[get_global_id(0)] = a+get_local_id(0);
    }"#;
    
    // Add dimensions to the ProQue builderf
    let mut proque: ProQue = ProQue::builder()
        .src(kernel_src)
        .dims(128)
        //.dims(data_size)  // Specify dimensions here
        .queue_properties(ocl::core::QUEUE_PROFILING_ENABLE)  // Add this line
        .build()?;

    proque.set_dims(32);        

    let c_buf = Buffer::builder()
        .queue(proque.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(32768)
        .build()?;
  

    //build the kernel and set arguments
    let kernel = proque.kernel_builder("add")
        .arg(&c_buf)
        .build()?;
        //.arg(&a_buf)
        //.arg(&b_buf)

    //warmmip 
    for _warmup in 0..2 {
        unsafe {
            kernel.cmd()
                .global_work_size(64)
                .local_work_size(32)
                .enq()?;
        }
        let __ = proque.finish()?;
    }
    
    // Test with different local work sizes
    let local_sizes = [1, 16, 32, 64];
    let colors = [&GREEN, &BLUE, &RED, &YELLOW];
    let mut all_points: Vec<(usize, Vec<(i32, i64)>)> = Vec::new();
    
    for &local_size in &local_sizes {
        let mut points: Vec<(i32, i64)> = Vec::new();
        println!("\nTesting with local_work_size = {}", local_size);
        
        // Calculate the range bounds as multiples of local_size
        let start = ((600 + local_size - 1) / local_size) * local_size; // Round up to next multiple
        let end = (800 / local_size) * local_size; // Round down to previous multiple
        
        // Make sure global size is multiple of local size
        for test_cores in (start..=end).step_by(local_size as usize) {
            let now = Instant::now(); 
            
            unsafe {
                kernel.cmd()
                    .global_work_size(test_cores)
                    .local_work_size(local_size) 
                    .enq()?;
            }
            let __ = proque.finish()?;
            let elapsed = now.elapsed();
            println!("Threads: {}, Time: {:?}", test_cores, elapsed);
            points.push((test_cores, elapsed.as_nanos() as i64));
        }
        
        all_points.push((local_size as usize, points));
    }

    // Find max Y value across all series for proper scaling
    let max_y = all_points.iter()
        .flat_map(|(_, points)| points.iter().map(|(_, y)| *y))
        .max()
        .unwrap_or(0);
    let y_range: std::ops::Range<i64> = 0..(max_y + (max_y / 10));

    // Create chart with all series
    let root = BitMapBackend::new("output.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).expect("Failed to fill background");
    let mut chart: ChartContext<'_, BitMapBackend<'_>, Cartesian2d<plotters::coord::types::RangedCoordi32, plotters::coord::types::RangedCoordi64>> = ChartBuilder::on(&root)
        .caption("GPU Kernel Execution Time by Local Size", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..1024, y_range)
        .expect("Failed to build chart");

    chart.configure_mesh()
        .x_desc("Number of Threads")
        .y_desc("Execution Time (ns)")
        .draw()
        .expect("Failed to draw mesh");
    
    // Draw each series with different color
    for (i, (local_size, points)) in all_points.iter().enumerate() {
        let color: &RGBColor = colors[i];
        chart.draw_series(LineSeries::new(
            points.iter().map(|(x, y)| (*x, *y)),
            color,
        ))
        .expect("Failed to draw series")
        .label(format!("Local Size: {}", local_size))
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }
    
    // Add a legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .expect("Failed to draw legend");

    // Print the path to the output file
    println!("Plot saved to: {}", std::env::current_dir()?.join("output.png").display());
    
    let mut c_data = vec![0.0f32; 128];
    c_buf.cmd().read(&mut c_data).enq()?;

    let mut i = 0;
    for &c in &c_data {
        if i % 16 == 0 && i != 0 { 
            println!("");
        }
        i += 1;
        print!("{:>5.1} ", c);  // Changed to print! with spacing for better readability
    }
    println!("\n");
    Ok(())
}