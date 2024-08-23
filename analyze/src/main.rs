use std::cell::RefCell;
use std::rc::Rc;
use std::{collections::HashMap, path::PathBuf};
use clap::Parser;
use mbox_reader;
mod models;
use models::MessageThread;
use models::Message;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    path: PathBuf,
    #[clap(short, long)]
    from: Option<String>,
}

fn main() {
    let args = Args::parse();
    println!("Path: {}", args.path.display());

    match parse_mbox(&args.path, &args.from) {
        Ok(_) => println!("Done"),
        Err(e) => eprintln!("Error: {}", e),
    }

    // Connect every message from the mailbox by thread

    // Cluster users by their participation in the same threads

}

fn parse_mbox(path: &PathBuf, from: &Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mbox = mbox_reader::MboxFile::from_file(path);
    match mbox {
        Ok(mbox) => process_messages(mbox, from),
        Err(e) => Err(e.into()),
    }
}

fn process_messages(mbox: mbox_reader::MboxFile, from: &Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let parser = mail_parser::MessageParser::new();
    let mut message_to_thread = HashMap::new();

    for entry in mbox.iter() {
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
            None => "",
            Some(date) => &date.to_string(),
        };

        if let Some(from_email) = from {
            if !sender_email.contains(from_email) {
                continue;
            }
        }
        println!("From: {}", sender_email);
        println!("Subject: {}", subject);
        println!("Date: {}", date);

        let msg = Message {
            id: message.message_id().unwrap().to_string(),
            in_reply_to: None,
            references: vec![],
            sender: sender_email.to_string(),
            subject: subject.to_string(),
        };

        // store every message as its own thread
        let message_id = msg.id.clone();
        let thread = MessageThread::new(msg);
        message_to_thread.insert(message_id.clone(), Rc::new(RefCell::new(thread)));


        // for each id in references and in_reply_to, add msg to the thread with that id
        // if no thread exists, create a new thread with msg as the parent

        let somerefs = message.references().as_text_list();
        match somerefs {
            Some(reflist) => {
                for reference in reflist {
                    if message_to_thread.contains_key(reference) {
                        println!("Found reference: {}", reference);
                        let this_thread_id = message_id.as_str();
                        let this_thread = message_to_thread.get(this_thread_id).unwrap();
                        let thread = message_to_thread.get(reference).unwrap();
                        thread.borrow_mut().add_child(this_thread);
                    } else {
                        println!("Missing reference: {}", reference);
                        // this might happen if the reference was missed, or came out of order, or is just
                        // junk. 
                    }
                }
            },
            None => (),
        }

        let somereply = message.in_reply_to().as_text();
        match somereply {
            Some(in_reply_to_id) => {
                if message_to_thread.contains_key(in_reply_to_id) {
                    println!("Found reply parent: {}", in_reply_to_id);
                    let this_thread_id = message_id.as_str();
                    let this_thread = message_to_thread.get(this_thread_id).unwrap();
                    let thread = message_to_thread.get(in_reply_to_id).unwrap();
                    thread.borrow_mut().add_child(this_thread);
                } else {
                    println!("Missing reply parent: {}", in_reply_to_id);
                    // this might happen if the parent was missed, or came out of order, or is just
                    // junk. 
                }
            },
            None => (),
        }

        // for pos in 0..message.text_body_count() - 1 {
        //     println!("pos: {}", pos);
        //     println!("Text Body: {}", message.body_text(pos).unwrap_or(std::borrow::Cow::Borrowed("")));
        // }


        println!();
    }
    Ok(())
}