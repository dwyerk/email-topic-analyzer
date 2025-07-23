use std::collections::{HashMap, HashSet};
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Message {
    pub id: String,
    pub in_reply_to: Option<String>,
    pub references: Vec<String>,
    pub sender: String,
    pub subject: String,
    pub normalized_subject: String,  // Subject with Re:, Fwd:, etc. removed
    pub body: String,                // Email body content
    pub processed_body: String,      // Body with quotes removed and cleaned
    pub topic_keywords: Vec<String>, // Extracted keywords from subject and body
    pub date: String,
}

impl Message {
    pub fn is_from_mailing_list(&self, mailing_list_patterns: &[String]) -> bool {
        mailing_list_patterns.iter().any(|pattern| {
            self.sender.contains(pattern)
        })
    }

    pub fn new(id: String, in_reply_to: Option<String>, references: Vec<String>, 
               sender: String, subject: String, body: String, date: String) -> Self {
        let normalized_subject = Self::normalize_subject(&subject);
        let processed_body = Self::process_body(&body);
        let topic_keywords = Self::extract_keywords_from_content(&normalized_subject, &processed_body);
        
        Message {
            id,
            in_reply_to,
            references,
            sender,
            subject,
            normalized_subject,
            body,
            processed_body,
            topic_keywords,
            date,
        }
    }

    fn normalize_subject(subject: &str) -> String {
        use regex::Regex;
        
        // Remove common prefixes like Re:, Fwd:, etc.
        let re = Regex::new(r"(?i)^(re:|fwd:|fw:|forward:|reply:|\[.*?\])\s*").unwrap();
        let mut normalized = subject.to_string();
        
        // Keep applying the regex until no more matches (handles multiple Re: Re: etc.)
        loop {
            let new_normalized = re.replace(&normalized, "").to_string();
            if new_normalized == normalized {
                break;
            }
            normalized = new_normalized;
        }
        
        normalized.trim().to_string()
    }

    fn process_body(body: &str) -> String {
        use regex::Regex;
        
        // Remove quoted lines (lines starting with >)
        let quote_re = Regex::new(r"(?m)^>.*$").unwrap();
        let without_quotes = quote_re.replace_all(body, "");
        
        // Remove email headers that sometimes appear in body
        let header_re = Regex::new(r"(?m)^[A-Za-z-]+:\s*.+$").unwrap();
        let without_headers = header_re.replace_all(&without_quotes, "");
        
        // Remove excessive whitespace and normalize
        let whitespace_re = Regex::new(r"\s+").unwrap();
        let normalized = whitespace_re.replace_all(&without_headers, " ");
        
        // Remove common email signatures and footers
        let sig_re = Regex::new(r"(?s)--\s*$.*").unwrap();
        let without_sig = sig_re.replace_all(&normalized, "");
        
        without_sig.trim().to_string()
    }

    fn extract_keywords_from_content(subject: &str, body: &str) -> Vec<String> {
        use regex::Regex;
        
        // Combine subject and body for keyword extraction
        let combined_text = format!("{} {}", subject, body);
        
        // Extract meaningful words (3+ characters, not common words)
        let word_re = Regex::new(r"\b[a-zA-Z]{3,}\b").unwrap();
        let stop_words: HashSet<&str> = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", 
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", 
            "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy", 
            "did", "man", "way", "what", "when", "where", "will", "with", "this",
            "that", "have", "from", "they", "know", "want", "been", "good", "much",
            "some", "time", "very", "when", "come", "here", "just", "like", "long",
            "make", "many", "over", "such", "take", "than", "them", "well", "were", "there",
            // Add email-specific stop words
            "wrote", "said", "email", "message", "sent", "reply", "forward", "original",
            "mailto", "http", "https", "www", "com", "org", "net", "edu", "gmail"
        ].iter().cloned().collect();
        
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Count word frequencies
        for word_match in word_re.find_iter(&combined_text) {
            let word = word_match.as_str().to_lowercase();
            if !stop_words.contains(word.as_str()) && word.len() >= 3 {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }
        
        // Sort by frequency and return top keywords
        let mut keywords: Vec<(String, usize)> = word_counts.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));
        
        keywords.into_iter()
            .take(10)  // Take top 10 keywords
            .map(|(word, _)| word)
            .collect()
    }

    fn extract_keywords(subject: &str) -> Vec<String> {
        // Keep this for backward compatibility, but it now just calls the new function
        Self::extract_keywords_from_content(subject, "")
    }
}

#[derive(Debug)]
pub struct MessageThread {
    pub message: Message,
    pub children: Vec<MessageThread>,
}

impl MessageThread {
    pub fn new(message: Message) -> MessageThread {
        MessageThread {
            message,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, child: MessageThread) {
        self.children.push(child);
    }

    pub fn print_thread(&self, indent: usize) {
        let indent_str = "  ".repeat(indent);
        println!("{}Subject: {}", indent_str, self.message.subject);
        println!("{}From: {}", indent_str, self.message.sender);
        println!("{}Date: {}", indent_str, self.message.date);
        println!("{}Message-ID: {}", indent_str, self.message.id);
        
        for child in &self.children {
            child.print_thread(indent + 1);
        }
    }

    pub fn count_messages(&self) -> usize {
        1 + self.children.iter().map(|child| child.count_messages()).sum::<usize>()
    }

    pub fn get_participants(&self) -> HashSet<String> {
        let mut participants = HashSet::new();
        participants.insert(self.message.sender.clone());
        
        for child in &self.children {
            participants.extend(child.get_participants());
        }
        
        participants
    }

    pub fn get_real_participants(&self, mailing_list_filters: &[String]) -> HashSet<String> {
        let mut participants = HashSet::new();
        
        if !self.is_mailing_list_sender(&self.message.sender, mailing_list_filters) {
            participants.insert(self.message.sender.clone());
        }
        
        for child in &self.children {
            participants.extend(child.get_real_participants(mailing_list_filters));
        }
        
        participants
    }

    fn is_mailing_list_sender(&self, email: &str, mailing_list_filters: &[String]) -> bool {
        mailing_list_filters.iter().any(|pattern| {
            email.contains(pattern)
        })
    }
}

pub struct ThreadCollection {
    pub threads: Vec<MessageThread>,
    pub orphaned_messages: Vec<Message>,
    pub mailing_list_filters: Vec<String>,
}

#[derive(Debug)]
pub struct UserParticipation {
    pub user: String,
    pub total_messages: usize,
    pub threads_participated: usize,
    pub threads_started: usize,
    pub collaborators: HashMap<String, usize>, // user -> number of shared threads
}

#[derive(Debug)]
pub struct ParticipationAnalysis {
    pub user_stats: Vec<UserParticipation>,
    pub thread_collaboration_matrix: HashMap<String, HashMap<String, usize>>,
    pub most_collaborative_pairs: Vec<(String, String, usize)>,
}

#[derive(Debug, Clone)]
pub struct Topic {
    pub name: String,
    pub keywords: Vec<String>,
    pub message_count: usize,
    pub thread_count: usize,
    pub participants: HashSet<String>,
}

#[derive(Debug)]
pub struct TopicAnalysis {
    pub topics: Vec<Topic>,
    pub keyword_frequency: HashMap<String, usize>,
    pub topic_timeline: HashMap<String, Vec<String>>, // topic -> dates
}

impl ThreadCollection {
    pub fn new() -> Self {
        ThreadCollection {
            threads: Vec::new(),
            orphaned_messages: Vec::new(),
            mailing_list_filters: vec![
                "noreply".to_string(),
                "mailer-daemon".to_string(),
                "postmaster".to_string(),
                "listserv".to_string(),
                "mailman".to_string(),
            ],
        }
    }

    pub fn new_with_address(mailing_list_address: &str) -> Self {
        // Extract the domain from the email address for additional filtering
        let domain = if let Some(at_pos) = mailing_list_address.find('@') {
            &mailing_list_address[at_pos+1..]
        } else {
            ""
        };
        
        ThreadCollection {
            threads: Vec::new(),
            orphaned_messages: Vec::new(),
            mailing_list_filters: vec![
                mailing_list_address.to_string(),
                format!("@{}", domain),
                "noreply".to_string(),
                "mailer-daemon".to_string(),
                "postmaster".to_string(),
                "listserv".to_string(),
                "mailman".to_string(),
            ],
        }
    }

    pub fn add_mailing_list_filter(&mut self, pattern: String) {
        self.mailing_list_filters.push(pattern);
    }

    fn is_mailing_list_address(&self, email: &str) -> bool {
        self.mailing_list_filters.iter().any(|pattern| {
            email.contains(pattern)
        })
    }

    pub fn print_summary(&self) {
        println!("\n=== Threading Summary ===");
        println!("Total threads: {}", self.threads.len());
        println!("Orphaned messages: {}", self.orphaned_messages.len());
        
        let total_messages: usize = self.threads.iter().map(|t| t.count_messages()).sum::<usize>() + self.orphaned_messages.len();
        println!("Total messages: {}", total_messages);
        
        println!("\n=== Thread Details ===");
        for (i, thread) in self.threads.iter().enumerate() {
            println!("\nThread {} ({} messages):", i + 1, thread.count_messages());
            thread.print_thread(0);
        }

        if !self.orphaned_messages.is_empty() {
            println!("\n=== Orphaned Messages ===");
            for msg in &self.orphaned_messages {
                println!("Subject: {} (From: {})", msg.subject, msg.sender);
            }
        }
    }

    pub fn analyze_user_participation(&self) -> ParticipationAnalysis {
        let mut user_message_counts: HashMap<String, usize> = HashMap::new();
        let mut user_threads_participated: HashMap<String, HashSet<usize>> = HashMap::new();
        let mut user_threads_started: HashMap<String, usize> = HashMap::new();
        let mut thread_participants: Vec<HashSet<String>> = Vec::new();

        // Analyze each thread
        for (thread_idx, thread) in self.threads.iter().enumerate() {
            // Get real participants (excluding mailing list addresses)
            let participants = thread.get_real_participants(&self.mailing_list_filters);
            thread_participants.push(participants.clone());

            // Count messages per user in this thread (excluding mailing list messages)
            self.count_real_messages_in_thread(thread, &mut user_message_counts);

            // Track thread participation (only real users)
            for participant in &participants {
                user_threads_participated
                    .entry(participant.clone())
                    .or_insert_with(HashSet::new)
                    .insert(thread_idx);
            }

            // Track who started the thread (only if not a mailing list address)
            let thread_starter = &thread.message.sender;
            if !self.is_mailing_list_address(thread_starter) {
                *user_threads_started.entry(thread_starter.clone()).or_insert(0) += 1;
            }
        }

        // Count orphaned messages (excluding mailing list messages)
        for msg in &self.orphaned_messages {
            if !self.is_mailing_list_address(&msg.sender) {
                *user_message_counts.entry(msg.sender.clone()).or_insert(0) += 1;
            }
        }

        // Build collaboration matrix (only between real users)
        let mut thread_collaboration_matrix: HashMap<String, HashMap<String, usize>> = HashMap::new();
        
        for participants in &thread_participants {
            // Only count collaboration if there are at least 2 real participants
            if participants.len() >= 2 {
                for user1 in participants {
                    for user2 in participants {
                        if user1 != user2 {
                            *thread_collaboration_matrix
                                .entry(user1.clone())
                                .or_insert_with(HashMap::new)
                                .entry(user2.clone())
                                .or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Create user participation stats
        let mut user_stats: Vec<UserParticipation> = Vec::new();
        
        for (user, message_count) in &user_message_counts {
            let threads_participated = user_threads_participated
                .get(user)
                .map(|set| set.len())
                .unwrap_or(0);
            
            let threads_started = user_threads_started.get(user).copied().unwrap_or(0);
            
            let collaborators = thread_collaboration_matrix
                .get(user)
                .cloned()
                .unwrap_or_default();

            user_stats.push(UserParticipation {
                user: user.clone(),
                total_messages: *message_count,
                threads_participated,
                threads_started,
                collaborators,
            });
        }

        // Sort by total messages (most active first)
        user_stats.sort_by(|a, b| b.total_messages.cmp(&a.total_messages));

        // Find most collaborative pairs
        let mut collaboration_pairs: Vec<(String, String, usize)> = Vec::new();
        
        for (user1, collaborators) in &thread_collaboration_matrix {
            for (user2, shared_threads) in collaborators {
                if user1 < user2 { // Avoid duplicates
                    collaboration_pairs.push((user1.clone(), user2.clone(), *shared_threads));
                }
            }
        }
        
        collaboration_pairs.sort_by(|a, b| b.2.cmp(&a.2));
        let most_collaborative_pairs = collaboration_pairs.into_iter().take(10).collect();

        ParticipationAnalysis {
            user_stats,
            thread_collaboration_matrix,
            most_collaborative_pairs,
        }
    }

    fn count_messages_in_thread(&self, thread: &MessageThread, counts: &mut HashMap<String, usize>) {
        *counts.entry(thread.message.sender.clone()).or_insert(0) += 1;
        
        for child in &thread.children {
            self.count_messages_in_thread(child, counts);
        }
    }

    fn count_real_messages_in_thread(&self, thread: &MessageThread, counts: &mut HashMap<String, usize>) {
        // Only count if sender is not a mailing list address
        if !self.is_mailing_list_address(&thread.message.sender) {
            *counts.entry(thread.message.sender.clone()).or_insert(0) += 1;
        }
        
        for child in &thread.children {
            self.count_real_messages_in_thread(child, counts);
        }
    }

    pub fn print_participation_analysis(&self) {
        let analysis = self.analyze_user_participation();
        
        println!("\n=== User Participation Analysis ===");
        println!("Note: Mailing list addresses filtered out: {:?}", self.mailing_list_filters);
        println!("Top 20 Most Active Users (Real Participants Only):");
        println!("{:<30} {:<8} {:<12} {:<12} {:<15}", "User", "Messages", "Threads", "Started", "Collaborators");
        println!("{}", "=".repeat(80));
        
        for user_stat in analysis.user_stats.iter().take(20) {
            println!("{:<30} {:<8} {:<12} {:<12} {:<15}", 
                user_stat.user,
                user_stat.total_messages,
                user_stat.threads_participated,
                user_stat.threads_started,
                user_stat.collaborators.len()
            );
        }

        if !analysis.most_collaborative_pairs.is_empty() {
            println!("\n=== Most Collaborative User Pairs (Real Users Only) ===");
            println!("{:<25} {:<25} {:<15}", "User 1", "User 2", "Shared Threads");
            println!("{}", "=".repeat(65));
            
            for (user1, user2, shared_count) in &analysis.most_collaborative_pairs {
                println!("{:<25} {:<25} {:<15}", user1, user2, shared_count);
            }
        } else {
            println!("\n=== No Real Collaborative Pairs Found ===");
            println!("Most messages appear to be individual posts rather than conversations between users.");
        }

        // Show detailed collaboration for most active users
        println!("\n=== Detailed Collaboration (Top 5 Users) ===");
        let mut found_collaborations = false;
        for user_stat in analysis.user_stats.iter().take(5) {
            if !user_stat.collaborators.is_empty() {
                found_collaborations = true;
                println!("\n{} collaborates with:", user_stat.user);
                let mut collab_vec: Vec<(&String, &usize)> = user_stat.collaborators.iter().collect();
                collab_vec.sort_by(|a, b| b.1.cmp(a.1));
                
                for (collaborator, shared_threads) in collab_vec.iter().take(10) {
                    println!("  {} - {} shared threads", collaborator, shared_threads);
                }
            }
        }
        
        if !found_collaborations {
            println!("No significant collaborations found among top users.");
            println!("This suggests most messages are individual posts to the mailing list");
            println!("rather than conversations between specific users.");
        }
    }

    pub fn analyze_topics(&self) -> TopicAnalysis {
        let mut keyword_frequency: HashMap<String, usize> = HashMap::new();
        let mut topic_threads: HashMap<String, Vec<usize>> = HashMap::new();
        let mut topic_participants: HashMap<String, HashSet<String>> = HashMap::new();
        let mut topic_timeline: HashMap<String, Vec<String>> = HashMap::new();

        // Analyze all messages for keywords
        for (thread_idx, thread) in self.threads.iter().enumerate() {
            self.collect_thread_keywords(thread, &mut keyword_frequency, &mut topic_threads, 
                                       &mut topic_participants, &mut topic_timeline, thread_idx);
        }

        // Analyze orphaned messages
        for msg in &self.orphaned_messages {
            for keyword in &msg.topic_keywords {
                *keyword_frequency.entry(keyword.clone()).or_insert(0) += 1;
                
                if !self.is_mailing_list_address(&msg.sender) {
                    topic_participants.entry(keyword.clone())
                        .or_insert_with(HashSet::new)
                        .insert(msg.sender.clone());
                }
                
                topic_timeline.entry(keyword.clone())
                    .or_insert_with(Vec::new)
                    .push(msg.date.clone());
            }
        }

        // Create topic clusters based on keyword frequency
        let mut topics = Vec::new();
        let min_frequency = 3; // Keywords must appear at least 3 times to be a topic

        for (keyword, frequency) in &keyword_frequency {
            if *frequency >= min_frequency {
                let thread_count = topic_threads.get(keyword)
                    .map(|threads| threads.len())
                    .unwrap_or(0);
                
                let participants = topic_participants.get(keyword)
                    .cloned()
                    .unwrap_or_default();

                topics.push(Topic {
                    name: keyword.clone(),
                    keywords: vec![keyword.clone()], // Could be expanded to include related words
                    message_count: *frequency,
                    thread_count,
                    participants,
                });
            }
        }

        // Sort topics by frequency
        topics.sort_by(|a, b| b.message_count.cmp(&a.message_count));

        TopicAnalysis {
            topics,
            keyword_frequency,
            topic_timeline,
        }
    }

    fn collect_thread_keywords(&self, thread: &MessageThread, 
                             keyword_frequency: &mut HashMap<String, usize>,
                             topic_threads: &mut HashMap<String, Vec<usize>>,
                             topic_participants: &mut HashMap<String, HashSet<String>>,
                             topic_timeline: &mut HashMap<String, Vec<String>>,
                             thread_idx: usize) {
        // Process current message
        for keyword in &thread.message.topic_keywords {
            *keyword_frequency.entry(keyword.clone()).or_insert(0) += 1;
            
            topic_threads.entry(keyword.clone())
                .or_insert_with(Vec::new)
                .push(thread_idx);
            
            if !self.is_mailing_list_address(&thread.message.sender) {
                topic_participants.entry(keyword.clone())
                    .or_insert_with(HashSet::new)
                    .insert(thread.message.sender.clone());
            }
            
            topic_timeline.entry(keyword.clone())
                .or_insert_with(Vec::new)
                .push(thread.message.date.clone());
        }

        // Process children recursively
        for child in &thread.children {
            self.collect_thread_keywords(child, keyword_frequency, topic_threads, 
                                       topic_participants, topic_timeline, thread_idx);
        }
    }

    pub fn print_topic_analysis(&self) {
        let analysis = self.analyze_topics();
        
        println!("\n=== Topic Analysis ===");
        println!("Discovered {} topics from subject lines and email bodies", analysis.topics.len());
        println!("\nTop 20 Topics by Frequency:");
        println!("{:<20} {:<10} {:<10} {:<15}", "Topic", "Messages", "Threads", "Participants");
        println!("{}", "=".repeat(60));
        
        for topic in analysis.topics.iter().take(20) {
            println!("{:<20} {:<10} {:<10} {:<15}", 
                topic.name,
                topic.message_count,
                topic.thread_count,
                topic.participants.len()
            );
        }

        // Show keyword cloud (most frequent keywords)
        println!("\n=== Keyword Frequency (Top 30) ===");
        let mut sorted_keywords: Vec<(&String, &usize)> = analysis.keyword_frequency.iter().collect();
        sorted_keywords.sort_by(|a, b| b.1.cmp(a.1));
        
        for (i, (keyword, count)) in sorted_keywords.iter().take(30).enumerate() {
            if i % 5 == 0 && i > 0 {
                println!();
            }
            print!("{:<15}({:>3}) ", keyword, count);
        }
        println!("\n");

        // Show some topic details
        if !analysis.topics.is_empty() {
            println!("=== Top 5 Topic Details ===");
            for topic in analysis.topics.iter().take(5) {
                println!("\n🔍 Topic: \"{}\"", topic.name);
                println!("   {} messages across {} threads", topic.message_count, topic.thread_count);
                println!("   {} unique participants", topic.participants.len());
                
                if topic.participants.len() <= 10 {
                    println!("   Participants: {}", 
                        topic.participants.iter().cloned().collect::<Vec<_>>().join(", "));
                } else {
                    let sample: Vec<String> = topic.participants.iter().take(10).cloned().collect();
                    println!("   Participants (sample): {}...", sample.join(", "));
                }
                
                // Show sample content from this topic
                self.show_sample_content_for_topic(&topic.name);
            }
        }
    }

    fn show_sample_content_for_topic(&self, topic_keyword: &str) {
        let mut found_samples = 0;
        let max_samples = 2;
        
        println!("   📧 Sample content:");
        
        'thread_loop: for thread in &self.threads {
            if self.find_sample_in_thread(thread, topic_keyword, &mut found_samples, max_samples) {
                if found_samples >= max_samples {
                    break 'thread_loop;
                }
            }
        }
        
        if found_samples == 0 {
            println!("      (No sample content found)");
        }
    }

    fn find_sample_in_thread(&self, thread: &MessageThread, topic_keyword: &str, 
                           found_samples: &mut usize, max_samples: usize) -> bool {
        if *found_samples >= max_samples {
            return true;
        }
        
        // Check if this message contains the topic keyword
        if thread.message.topic_keywords.contains(&topic_keyword.to_string()) {
            let subject = &thread.message.subject;
            let body_preview = if thread.message.processed_body.len() > 100 {
                format!("{}...", &thread.message.processed_body[..100])
            } else {
                thread.message.processed_body.clone()
            };
            
            println!("      • Subject: \"{}\"", subject);
            if !body_preview.trim().is_empty() {
                println!("        Body: \"{}\"", body_preview.trim());
            }
            println!("        From: {}", thread.message.sender);
            println!();
            
            *found_samples += 1;
            
            if *found_samples >= max_samples {
                return true;
            }
        }
        
        // Search in children
        for child in &thread.children {
            if self.find_sample_in_thread(child, topic_keyword, found_samples, max_samples) {
                return true;
            }
        }
        
        false
    }
}