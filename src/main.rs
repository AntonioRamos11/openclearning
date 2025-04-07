use ocl::{ProQue, Result};

// This is a simple OpenCL program that creates a buffer, writes data to it, and runs a kernel.
// It uses the ocl crate to handle OpenCL operations in Rust.
// The kernel is a simple function that takes a buffer and writes to it.

///                 Open cl                      hip 
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
//100000000000
fn main() -> Result<()> {
    // Define the size of our data
    let data_size = 128;
    
    let kernel_src = r#"kernel void add (global float *c) {
        int a = get_local_id(0);
        for (int i = 0; i < 100000; i++) {
                a+=i;
         }
        c[get_global_id(0)] = a;
    }"#;
    
    // Add dimensions to the ProQue builder
    let proque = ProQue::builder()
        .src(kernel_src)
        .dims(data_size)  // Specify dimensions here
        .build()?;
        
    let a_data = vec![1.0f32; data_size];
    let b_data = vec![2.0f32; data_size];
    
    // Create buffers with the same size as our data
    let a_buf = proque.create_buffer::<f32>()?;
    let b_buf = proque.create_buffer::<f32>()?;
    let c_buf = proque.create_buffer::<f32>()?;

    a_buf.cmd().write(&a_data).enq()?;
    b_buf.cmd().write(&b_data).enq()?;

    //build the kernel and set arguments
    let kernel = proque.kernel_builder("add")
        .arg(&c_buf)
        .build()?;
        //.arg(&a_buf)
        //.arg(&b_buf)


    use std::time::Instant;
    let start = Instant::now();
        
    unsafe {
        kernel.cmd()
            .global_work_size(a_buf.len())
            .local_work_size(32)
            .enq()?;
    }
    let elapsed = start.elapsed();
    println!("Kernel execution time: {:?}", elapsed);
  
    let mut c_data = vec![0.0f32; data_size];
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