use std::{collections::HashMap, path::PathBuf};
use clap::Parser;
use mbox_reader;
mod models;
use models::{Message, ThreadCollection};

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
}

fn main() {
    let args = Args::parse();
    println!("Path: {}", args.path.display());

    match parse_mbox(&args.path, &args.from, &args.mailing_list_address) {
        Ok(thread_collection) => {
            if args.analysis_only {
                thread_collection.print_participation_analysis();
            } else {
                thread_collection.print_summary();
                thread_collection.print_participation_analysis();
            }
        },
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn parse_mbox(path: &PathBuf, from: &Option<String>, mailing_list_address: &Option<String>) -> Result<ThreadCollection, Box<dyn std::error::Error>> {
    let mbox = mbox_reader::MboxFile::from_file(path);
    match mbox {
        Ok(mbox) => process_messages(mbox, from, mailing_list_address),
        Err(e) => Err(e.into()),
    }
}

fn process_messages(mbox: mbox_reader::MboxFile, from: &Option<String>, mailing_list_address: &Option<String>) -> Result<ThreadCollection, Box<dyn std::error::Error>> {
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

        let msg = Message {
            id: message.message_id().unwrap_or("").to_string(),
            in_reply_to,
            references,
            sender: sender_email.to_string(),
            subject: subject.to_string(),
            date,
        };

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