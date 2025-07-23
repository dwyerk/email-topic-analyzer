use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use mbox_reader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 4 {
        eprintln!("Usage: {} <input.mbox> <output.mbox> <sample_percentage>", args[0]);
        eprintln!("Example: sample_mbox input.mbox sample.mbox 10");
        std::process::exit(1);
    }
    
    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);
    let sample_percentage: f32 = args[3].parse()
        .map_err(|_| "Sample percentage must be a number")?;
    
    if sample_percentage <= 0.0 || sample_percentage > 100.0 {
        eprintln!("Sample percentage must be between 0 and 100");
        std::process::exit(1);
    }
    
    println!("Sampling {}% of emails from {} to {}", 
             sample_percentage, input_path.display(), output_path.display());
    
    let mbox = mbox_reader::MboxFile::from_file(&input_path)?;
    let output_file = File::create(&output_path)?;
    let mut writer = BufWriter::new(output_file);
    
    let total_messages = mbox.iter().count();
    println!("Total messages in source: {}", total_messages);
    
    let sample_size = ((total_messages as f32) * (sample_percentage / 100.0)) as usize;
    println!("Will sample approximately {} messages", sample_size);
    
    let step = if sample_size > 0 { total_messages / sample_size } else { 1 };
    let step = step.max(1); // Ensure step is at least 1
    
    println!("Taking every {}th message", step);
    
    let mut sampled_count = 0;
    
    for (i, entry) in mbox.iter().enumerate() {
        if i % step == 0 {
            // Write the entire message entry to the output file
            if let Some(message_bytes) = entry.message() {
                // Write the From line (mbox separator)
                writeln!(writer, "From {}", entry.start().as_str())?;
                
                // Write the message content
                writer.write_all(message_bytes)?;
                
                // Ensure there's a newline at the end
                writer.write_all(b"\n")?;
                
                sampled_count += 1;
                
                if sampled_count % 100 == 0 {
                    println!("Sampled {} messages...", sampled_count);
                }
            }
        }
    }
    
    writer.flush()?;
    
    println!("✅ Successfully created sample file with {} messages", sampled_count);
    println!("Sample file: {}", output_path.display());
    
    Ok(())
}
