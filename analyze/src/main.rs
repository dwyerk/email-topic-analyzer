use std::{collections::HashMap, path::PathBuf};
use clap::Parser;
use mbox_reader;
use chrono::{NaiveDate, DateTime};
mod models;
mod topic_modeling;
use models::{Message, ThreadCollection};
use topic_modeling::{SimpleLDA, LDAConfig};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    path: PathBuf,
    #[clap(short, long)]
    from: Option<String>,
    #[clap(short, long, help = "Show only participation analysis, not full thread details")]
    analysis_only: bool,
    #[clap(short, long, help = "Mailing list address to filter out (e.g., 'pen@penhood.net'). If not specified, no mailing list filtering is applied.")]
    mailing_list_address: Option<String>,
    #[clap(short, long, help = "Show topic analysis based on subject lines")]
    topic_analysis: bool,
    #[clap(long, help = "Run unsupervised topic modeling using LDA")]
    topic_modeling: bool,
    #[clap(long, help = "Number of topics for LDA modeling", default_value = "10")]
    num_topics: usize,
    #[clap(long, help = "Disable statistical significance filtering for topic modeling")]
    no_significance_filter: bool,
    #[clap(long, help = "Start date for filtering emails (format: YYYY-MM-DD)")]
    start_date: Option<String>,
    #[clap(long, help = "End date for filtering emails (format: YYYY-MM-DD)")]
    end_date: Option<String>,
}

fn main() {
    let args = Args::parse();
    println!("Path: {}", args.path.display());

    // Validate and parse date arguments
    let date_range = match parse_date_range(&args.start_date, &args.end_date) {
        Ok(range) => range,
        Err(e) => {
            eprintln!("Date parsing error: {}", e);
            return;
        }
    };

    if let Some((start, end)) = &date_range {
        println!("Filtering emails from {} to {}", start.format("%Y-%m-%d"), end.format("%Y-%m-%d"));
    }

    match parse_mbox(&args.path, &args.from, &args.mailing_list_address, &date_range) {
        Ok(thread_collection) => {
            if args.topic_modeling {
                run_topic_modeling(&thread_collection, args.num_topics, !args.no_significance_filter);
            } else if args.topic_analysis {
                thread_collection.print_topic_analysis();
            } else if args.analysis_only {
                thread_collection.print_participation_analysis();
            } else {
                thread_collection.print_summary();
                thread_collection.print_participation_analysis();
            }
        },
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn parse_email_date(date_str: &str) -> Option<NaiveDate> {
    // Clean up the date string - remove common timezone abbreviations and extra text
    let cleaned = date_str
        .replace(" (GMT)", "")
        .replace(" (UTC)", "")
        .replace(" GMT", "")
        .replace(" UTC", "")
        .trim()
        .to_string();
    
    // Try parsing as DateTime first (handles timezone-aware formats)
    // ISO 8601 formats commonly found in email headers
    if let Ok(dt) = DateTime::parse_from_rfc3339(&cleaned) {
        return Some(dt.naive_utc().date());
    }
    
    // Try alternative ISO format with space instead of 'T'
    let space_format = cleaned.replace('T', " ");
    if let Ok(dt) = DateTime::parse_from_rfc3339(&space_format.replace(" ", "T")) {
        return Some(dt.naive_utc().date());
    }
    
    // Common email date formats to try
    let formats = [
        "%Y-%m-%dT%H:%M:%S%z",          // 2021-03-30T15:33:16-04:00
        "%Y-%m-%dT%H:%M:%SZ",           // 2021-03-30T15:33:16Z
        "%Y-%m-%dT%H:%M:%S%.3fZ",       // With milliseconds
        "%a, %d %b %Y %H:%M:%S %z",     // RFC 2822 format: "Wed, 18 Oct 2023 15:30:45 +0000"
        "%d %b %Y %H:%M:%S %z",         // Without day name: "18 Oct 2023 15:30:45 +0000"
        "%Y-%m-%d %H:%M:%S",            // ISO format: "2023-10-18 15:30:45"
        "%Y-%m-%d",                     // Simple date: "2023-10-18"
        "%a, %d %b %Y %H:%M:%S",        // Without timezone: "Wed, 18 Oct 2023 15:30:45"
        "%d %b %Y %H:%M:%S",            // Without day name and timezone: "18 Oct 2023 15:30:45"
        "%d %b %Y",                     // Just date: "18 Oct 2023"
    ];
    
    // Try parsing with different formats
    for format in &formats {
        // For timezone-aware formats, try parsing as DateTime first
        if format.contains("%z") {
            if let Ok(dt) = DateTime::parse_from_str(&cleaned, format) {
                return Some(dt.naive_utc().date());
            }
        } else {
            // For timezone-naive formats, parse directly as NaiveDateTime or NaiveDate
            if format.contains("%H") {
                if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(&cleaned, format) {
                    return Some(ndt.date());
                }
            } else if let Ok(nd) = NaiveDate::parse_from_str(&cleaned, format) {
                return Some(nd);
            }
        }
    }
    
    None
}

fn parse_date_range(start_date: &Option<String>, end_date: &Option<String>) -> Result<Option<(NaiveDate, NaiveDate)>, Box<dyn std::error::Error>> {
    match (start_date, end_date) {
        (None, None) => Ok(None),
        (Some(start), None) => {
            let start_parsed = NaiveDate::parse_from_str(start, "%Y-%m-%d")?;
            // If only start date is provided, use current date as end
            let end_parsed = chrono::Local::now().naive_local().date();
            Ok(Some((start_parsed, end_parsed)))
        },
        (None, Some(end)) => {
            let end_parsed = NaiveDate::parse_from_str(end, "%Y-%m-%d")?;
            // If only end date is provided, use a very early date as start
            let start_parsed = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            Ok(Some((start_parsed, end_parsed)))
        },
        (Some(start), Some(end)) => {
            let start_parsed = NaiveDate::parse_from_str(start, "%Y-%m-%d")?;
            let end_parsed = NaiveDate::parse_from_str(end, "%Y-%m-%d")?;
            if start_parsed > end_parsed {
                return Err("Start date must be before or equal to end date".into());
            }
            Ok(Some((start_parsed, end_parsed)))
        }
    }
}

fn parse_mbox(path: &PathBuf, from: &Option<String>, mailing_list_address: &Option<String>, date_range: &Option<(NaiveDate, NaiveDate)>) -> Result<ThreadCollection, Box<dyn std::error::Error>> {
    let mbox = mbox_reader::MboxFile::from_file(path);
    match mbox {
        Ok(mbox) => process_messages(mbox, from, mailing_list_address, date_range),
        Err(e) => Err(e.into()),
    }
}

fn process_messages(mbox: mbox_reader::MboxFile, from: &Option<String>, mailing_list_address: &Option<String>, date_range: &Option<(NaiveDate, NaiveDate)>) -> Result<ThreadCollection, Box<dyn std::error::Error>> {
    let parser = mail_parser::MessageParser::new();
    let mut messages = Vec::new();

    println!("Starting to parse messages...");
    if let Some(address) = mailing_list_address {
        println!("Filtering mailing list address: {}", address);
    } else {
        println!("No mailing list filtering applied");
    }

    // Phase 1: Parse all messages and collect them
    for (i, entry) in mbox.iter().enumerate() {
        if i % 100 == 0 {
            println!("Processing message {}", i);
        }

        let Some(message_bytes) = entry.message() else {
            eprintln!("No message: {}", entry.start().as_str());
            continue;
        };

        let Some(message) = parser.parse(message_bytes) else {
            eprintln!("Failed to parse message: {}", entry.start().as_str());
            continue;
        };

        let Some(sender) = message.from().or(message.sender()) else {
            eprintln!("No sender: {}", entry.start().as_str());
            continue;
        };
        let sender_address = match sender {
            mail_parser::Address::List(addresses) => addresses.first(),
            mail_parser::Address::Group(groups) => groups.first().unwrap().addresses.first(),
        };
        let sender_email = match sender_address {
            None => "",
            Some(address) => address.address().unwrap(),
        };

        let subject = message.subject().unwrap_or("");

        let date = match message.date() {
            None => "".to_string(),
            Some(date) => date.to_string(),
        };

        // Apply date filter if specified
        if let Some((start_date, end_date)) = date_range {
            if let Some(message_date) = message.date() {
                // Convert mail-parser DateTime to chrono NaiveDate for comparison
                // Use the date string representation and parse it
                let date_str = message_date.to_string();
                
                // Try to parse various date formats commonly found in email headers
                let parsed_date = parse_email_date(&date_str);
                
                if let Some(message_naive_date) = parsed_date {
                    if message_naive_date < *start_date || message_naive_date > *end_date {
                        continue;
                    }
                } else {
                    // Skip messages with unparseable dates when date filtering is enabled
                    eprintln!("Warning: Could not parse date '{}' for message filtering", date_str);
                    continue;
                }
            } else {
                // Skip messages without dates when date filtering is enabled
                continue;
            }
        }

        // Apply email filter if specified
        if let Some(from_email) = from {
            if !sender_email.contains(from_email) {
                continue;
            }
        }

        // Extract threading information
        let in_reply_to = message.in_reply_to().as_text().map(|s| s.to_string());
        let references = message.references().as_text_list()
            .map(|refs| refs.iter().map(|s| s.to_string()).collect())
            .unwrap_or_else(Vec::new);

        // Extract body content
        let body = match message.body_text(0) {
            Some(text) => text.to_string(),
            None => {
                // Try to get HTML body if text is not available
                match message.body_html(0) {
                    Some(html) => html.to_string(),
                    None => String::new(),
                }
            }
        };

        let msg = Message::new(
            message.message_id().unwrap_or("").to_string(),
            in_reply_to,
            references,
            sender_email.to_string(),
            subject.to_string(),
            body,
            date,
        );

        // Skip messages without message ID
        if msg.id.is_empty() {
            eprintln!("Skipping message without ID from {}", sender_email);
            continue;
        }

        messages.push(msg);
    }

    println!("Parsed {} messages. Building threads...", messages.len());

    // Phase 2: Build threads
    build_threads(messages, mailing_list_address)
}

fn build_threads(messages: Vec<Message>, mailing_list_address: &Option<String>) -> Result<ThreadCollection, Box<dyn std::error::Error>> {
    let mut message_map: HashMap<String, Message> = HashMap::new();
    let mut child_to_parent: HashMap<String, String> = HashMap::new();
    
    println!("Indexing messages...");
    
    // Index all messages by ID
    for message in messages {
        message_map.insert(message.id.clone(), message);
    }

    println!("Found {} unique messages. Building relationships...", message_map.len());

    // Build parent-child relationships
    for (msg_id, message) in &message_map {
        // Check in_reply_to first (most direct relationship)
        if let Some(parent_id) = &message.in_reply_to {
            if message_map.contains_key(parent_id) {
                child_to_parent.insert(msg_id.clone(), parent_id.clone());
                continue;
            }
        }

        // If no in_reply_to or parent not found, check references
        // Use the last reference as the most immediate parent
        if let Some(parent_id) = message.references.last() {
            if message_map.contains_key(parent_id) {
                child_to_parent.insert(msg_id.clone(), parent_id.clone());
            }
        }
    }

    // Find root messages (those that are not children of any other message)
    let mut root_ids: Vec<String> = message_map.keys()
        .filter(|id| !child_to_parent.contains_key(*id))
        .cloned()
        .collect();

    // Sort root messages by date for consistent ordering
    root_ids.sort_by(|a, b| {
        let date_a = &message_map[a].date;
        let date_b = &message_map[b].date;
        date_a.cmp(date_b)
    });

    let mut thread_collection = match mailing_list_address {
        Some(address) => ThreadCollection::new_with_address(address),
        None => ThreadCollection::new(),
    };

    // Build thread trees
    for root_id in root_ids {
        if let Some(root_message) = message_map.remove(&root_id) {
            let thread = build_thread_recursive(root_message, &mut message_map, &child_to_parent);
            thread_collection.threads.push(thread);
        }
    }

    // Add any remaining messages as orphaned
    for (_, message) in message_map {
        thread_collection.orphaned_messages.push(message);
    }

    Ok(thread_collection)
}

fn build_thread_recursive(
    message: Message,
    message_map: &mut HashMap<String, Message>,
    child_to_parent: &HashMap<String, String>,
) -> models::MessageThread {
    let mut thread = models::MessageThread::new(message.clone());
    
    // Find all children of this message
    let children: Vec<String> = child_to_parent.iter()
        .filter(|(_, parent_id)| *parent_id == &message.id)
        .map(|(child_id, _)| child_id.clone())
        .collect();

    // Recursively build child threads
    for child_id in children {
        if let Some(child_message) = message_map.remove(&child_id) {
            let child_thread = build_thread_recursive(child_message, message_map, child_to_parent);
            thread.add_child(child_thread);
        }
    }

    thread
}

fn run_topic_modeling(thread_collection: &ThreadCollection, num_topics: usize, use_significance_filter: bool) {
    println!("\n=== Unsupervised Topic Modeling (LDA) ===");
    println!("Analyzing {} threads and {} orphaned messages...", 
             thread_collection.threads.len(), 
             thread_collection.orphaned_messages.len());
    
    // Collect all message content for topic modeling
    let mut texts = Vec::new();
    
    // Collect from threads
    for thread in &thread_collection.threads {
        collect_thread_texts(thread, &mut texts);
    }
    
    // Collect from orphaned messages
    for message in &thread_collection.orphaned_messages {
        let combined_text = format!("{} {}", message.subject, message.processed_body);
        texts.push(combined_text);
    }
    
    if texts.is_empty() {
        println!("No text content found for topic modeling.");
        return;
    }
    
    println!("Processing {} documents for topic modeling...", texts.len());
    
    // Configure and run LDA
    let config = LDAConfig {
        num_topics,
        max_iterations: 50,
        alpha: 0.1,
        beta: 0.01,
        min_word_freq: 2,
        max_vocab_size: 500,
        min_tf_idf: 0.01,  // Minimum TF-IDF threshold for significance
        use_significance_filtering: use_significance_filter,  // Use parameter from command line
    };
    
    let lda = SimpleLDA::new(config);
    let model = lda.fit(texts);
    
    // Display results
    println!("\n🔍 Discovered {} topics:", model.topics.len());
    println!("📖 Vocabulary size: {}", model.vocabulary.len());
    
    println!("\n=== Topic Details ===");
    for topic in &model.topics {
        println!("\n📌 Topic {}: \"{}\"", topic.id + 1, topic.name);
        println!("   Documents: {} ({:.1}%)", 
                topic.documents.len(), 
                (topic.documents.len() as f64 / model.documents.len() as f64) * 100.0);
        println!("   Coherence: {:.3}", topic.coherence_score);
        
        println!("   Top words:");
        for (i, (word, prob)) in topic.words.iter().take(8).enumerate() {
            if i % 4 == 0 && i > 0 {
                println!();
            }
            print!("     {:<12}({:.3}) ", word, prob);
        }
        println!();
        
        // Show sample documents
        if !topic.documents.is_empty() {
            println!("   Sample documents:");
            for &doc_id in topic.documents.iter().take(3) {
                if let Some(doc) = model.documents.get(doc_id) {
                    let preview = if doc.text.len() > 80 {
                        format!("{}...", &doc.text[..77])
                    } else {
                        doc.text.clone()
                    };
                    println!("     • {}", preview);
                }
            }
        }
    }
    
    // Topic distribution summary
    println!("\n=== Topic Distribution Summary ===");
    for topic in &model.topics {
        println!("Topic {:<2} ({:<20}): {:<3} docs ({:>5.1}%)", 
                topic.id + 1, 
                &topic.name[..std::cmp::min(20, topic.name.len())],
                topic.documents.len(),
                (topic.documents.len() as f64 / model.documents.len() as f64) * 100.0);
    }
}

fn collect_thread_texts(thread: &models::MessageThread, texts: &mut Vec<String>) {
    // Add current message text
    let combined_text = format!("{} {}", thread.message.subject, thread.message.processed_body);
    texts.push(combined_text);
    
    // Recursively collect from children
    for child in &thread.children {
        collect_thread_texts(child, texts);
    }
}