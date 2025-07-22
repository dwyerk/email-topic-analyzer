use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Message {
    pub id: String,
    pub in_reply_to: Option<String>,
    pub references: Vec<String>,
    pub sender: String,
    pub subject: String,
    pub date: String,
}

impl Message {
    pub fn is_from_mailing_list(&self, mailing_list_patterns: &[String]) -> bool {
        mailing_list_patterns.iter().any(|pattern| {
            self.sender.contains(pattern)
        })
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
}