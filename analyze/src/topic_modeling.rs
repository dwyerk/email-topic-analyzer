use std::collections::{HashMap, HashSet};
use regex::Regex;
use counter::Counter;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct WordStatistics {
    pub word: String,
    pub term_frequency: usize,      // Total occurrences across all documents
    pub document_frequency: usize,  // Number of documents containing this word
    pub tf_idf_score: f64,         // TF-IDF score
    pub variance: f64,             // Variance in document frequencies
    pub significance_score: f64,   // Combined significance measure
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeledTopic {
    pub id: usize,
    pub name: String,
    pub words: Vec<(String, f64)>,  // Word and its probability in this topic
    pub documents: Vec<usize>,       // Document indices belonging to this topic
    pub coherence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: usize,
    pub text: String,
    pub words: Vec<String>,
    pub topic_distribution: Vec<f64>,  // Probability distribution over topics
    pub primary_topic: usize,
}

#[derive(Debug)]
pub struct TopicModel {
    pub topics: Vec<ModeledTopic>,
    pub documents: Vec<Document>,
    pub vocabulary: Vec<String>,
    pub word_topic_matrix: Vec<Vec<f64>>,  // Words x Topics
    pub doc_topic_matrix: Vec<Vec<f64>>,   // Documents x Topics
}

pub struct LDAConfig {
    pub num_topics: usize,
    pub max_iterations: usize,
    pub alpha: f64,  // Document-topic concentration
    pub beta: f64,   // Topic-word concentration
    pub min_word_freq: usize,
    pub max_vocab_size: usize,
    pub min_tf_idf: f64,  // Minimum TF-IDF score for word inclusion
    pub use_significance_filtering: bool,  // Enable statistical significance filtering
}

impl Default for LDAConfig {
    fn default() -> Self {
        LDAConfig {
            num_topics: 10,
            max_iterations: 100,
            alpha: 0.1,
            beta: 0.01,
            min_word_freq: 3,
            max_vocab_size: 1000,
            min_tf_idf: 0.01,  // Minimum TF-IDF threshold
            use_significance_filtering: true,  // Enable by default
        }
    }
}

pub struct SimpleLDA {
    config: LDAConfig,
}

impl SimpleLDA {
    pub fn new(config: LDAConfig) -> Self {
        SimpleLDA {
            config,
        }
    }

    pub fn fit(&self, texts: Vec<String>) -> TopicModel {
        // Step 1: Preprocess documents
        let documents = self.preprocess_documents(texts);
        
        // Step 2: Build vocabulary and get word statistics
        let (vocabulary, word_stats) = self.build_vocabulary_with_stats(&documents);
        
        // Step 3: Convert documents to word indices
        let word_docs = self.vectorize_documents(&documents, &vocabulary);
        
        // Step 4: Run simplified LDA algorithm
        let (word_topic_matrix, doc_topic_matrix) = self.run_lda(&word_docs, vocabulary.len());
        
        // Step 5: Extract topics and assign documents (with significance-aware naming)
        let topics = self.extract_topics_with_significance(&word_topic_matrix, &vocabulary, &doc_topic_matrix, &word_stats);
        
        // Step 6: Create final documents with topic assignments
        let final_documents = self.create_final_documents(documents, &doc_topic_matrix);
        
        TopicModel {
            topics,
            documents: final_documents,
            vocabulary,
            word_topic_matrix,
            doc_topic_matrix,
        }
    }

    fn preprocess_documents(&self, texts: Vec<String>) -> Vec<Vec<String>> {
        let stop_words: HashSet<&str> = [
            // Common English stop words
            "the", "and", "for", "are", "but", "not", "you", "your", "all", "can", "had", 
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", 
            "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy", 
            "did", "man", "way", "what", "when", "where", "will", "with", "this",
            "that", "have", "from", "they", "know", "want", "been", "good", "much",
            "some", "time", "very", "when", "come", "here", "just", "like", "long",
            "make", "many", "over", "such", "take", "than", "them", "well", "were", "there",
            // Pronouns and possessives
            "his", "her", "hers", "their", "theirs", "mine", "yours", "ours",
            // Common verbs with low semantic value
            "would", "could", "should", "might", "will", "shall", "can", "may",
            "does", "did", "done", "being", "having", "doing",
            // Common adjectives/adverbs with low topic value
            "more", "most", "less", "least", "few", "many", "much", "little",
            "big", "small", "large", "great", "same", "different", "other",
            "first", "last", "next", "each", "every", "any", "some", "all",
            // Communication/email specific stop words
            "wrote", "said", "email", "message", "sent", "reply", "forward", "original",
            "mailto", "http", "https", "www", "com", "org", "net", "edu", "gmail",
            // Common discourse markers
            "also", "too", "even", "still", "yet", "already", "again", "once",
            "really", "quite", "pretty", "very", "so", "too", "quite",
            // Mailing list specific words
            "list", "pen", "penhood", "thanks", "please", "hello", "dear"
        ].iter().cloned().collect();

        let word_re = Regex::new(r"\b[a-zA-Z]{3,}\b").unwrap();

        texts.into_iter().map(|text| {
            word_re.find_iter(&text.to_lowercase())
                .map(|m| m.as_str())
                .filter(|word| !stop_words.contains(word) && word.len() >= 3)
                .map(|word| word.to_string())
                .collect()
        }).collect()
    }

    fn build_vocabulary(&self, documents: &[Vec<String>]) -> Vec<String> {
        let (vocab, _) = self.build_vocabulary_with_stats(documents);
        vocab
    }

    fn build_vocabulary_with_stats(&self, documents: &[Vec<String>]) -> (Vec<String>, HashMap<String, WordStatistics>) {
        if self.config.use_significance_filtering {
            self.build_vocabulary_with_significance_and_stats(documents)
        } else {
            let vocab = self.build_vocabulary_simple(documents);
            (vocab, HashMap::new()) // Empty stats for simple mode
        }
    }

    fn build_vocabulary_simple(&self, documents: &[Vec<String>]) -> Vec<String> {
        let mut word_counts: Counter<String> = Counter::new();
        
        for doc in documents {
            for word in doc {
                word_counts[word] += 1;
            }
        }

        // Filter by minimum frequency and take top words
        let mut vocab: Vec<(String, usize)> = word_counts.into_iter()
            .filter(|(_, count)| *count >= self.config.min_word_freq)
            .collect();
        
        vocab.sort_by(|a, b| b.1.cmp(&a.1));
        vocab.truncate(self.config.max_vocab_size);
        
        vocab.into_iter().map(|(word, _)| word).collect()
    }

    fn build_vocabulary_with_significance_and_stats(&self, documents: &[Vec<String>]) -> (Vec<String>, HashMap<String, WordStatistics>) {
        // Step 1: Calculate basic statistics
        let word_stats = self.calculate_word_statistics(documents);
        
        // Step 2: Filter by significance measures
        let mut significant_words: Vec<WordStatistics> = word_stats.into_iter()
            .filter(|stats| {
                stats.term_frequency >= self.config.min_word_freq &&
                stats.tf_idf_score >= self.config.min_tf_idf &&
                stats.significance_score > 0.0
            })
            .collect();
        
        // Step 3: Sort by significance score (combination of TF-IDF and variance)
        significant_words.sort_by(|a, b| b.significance_score.partial_cmp(&a.significance_score).unwrap());
        
        // Step 4: Take top words up to vocabulary size
        significant_words.truncate(self.config.max_vocab_size);
        
        println!("📊 Vocabulary filtering results:");
        println!("   Total unique words: {}", significant_words.len());
        println!("   Top 10 significant words:");
        for (i, stats) in significant_words.iter().take(10).enumerate() {
            println!("   {:<2}. {:<15} TF-IDF: {:.3}, Sig: {:.3}", 
                    i + 1, stats.word, stats.tf_idf_score, stats.significance_score);
        }
        
        // Create stats map for quick lookup
        let stats_map: HashMap<String, WordStatistics> = significant_words.iter()
            .map(|stats| (stats.word.clone(), stats.clone()))
            .collect();
        
        let vocabulary: Vec<String> = significant_words.into_iter().map(|stats| stats.word).collect();
        (vocabulary, stats_map)
    }

    fn build_vocabulary_with_significance(&self, documents: &[Vec<String>]) -> Vec<String> {
        let (vocab, _) = self.build_vocabulary_with_significance_and_stats(documents);
        vocab
    }

    fn calculate_word_statistics(&self, documents: &[Vec<String>]) -> Vec<WordStatistics> {
        let total_docs = documents.len() as f64;
        let mut word_doc_freq: HashMap<String, usize> = HashMap::new();
        let mut word_total_freq: Counter<String> = Counter::new();
        let mut word_doc_counts: HashMap<String, Vec<usize>> = HashMap::new();

        // Calculate frequencies and document distributions
        for doc in documents {
            let mut doc_word_counts: Counter<String> = Counter::new();
            
            // Count words in this document
            for word in doc {
                doc_word_counts[word] += 1;
                word_total_freq[word] += 1;
            }
            
            // Track which documents contain each word and their frequencies
            for (word, count) in doc_word_counts.into_iter() {
                *word_doc_freq.entry(word.clone()).or_insert(0) += 1;
                word_doc_counts.entry(word).or_insert_with(Vec::new).push(count);
            }
        }

        // Calculate TF-IDF and significance scores
        word_total_freq.into_iter()
            .filter(|(_, freq)| *freq >= self.config.min_word_freq)
            .map(|(word, term_freq)| {
                let doc_freq = word_doc_freq.get(&word).copied().unwrap_or(0);
                let tf_idf = self.calculate_tf_idf(term_freq, doc_freq, total_docs);
                let variance = self.calculate_word_variance(&word_doc_counts.get(&word).unwrap_or(&vec![]));
                let significance = self.calculate_significance_score(tf_idf, variance, doc_freq, total_docs);
                
                WordStatistics {
                    word,
                    term_frequency: term_freq,
                    document_frequency: doc_freq,
                    tf_idf_score: tf_idf,
                    variance,
                    significance_score: significance,
                }
            })
            .collect()
    }

    fn calculate_tf_idf(&self, term_freq: usize, doc_freq: usize, total_docs: f64) -> f64 {
        if doc_freq == 0 {
            return 0.0;
        }
        
        let tf = term_freq as f64;
        let idf = (total_docs / doc_freq as f64).ln();
        tf * idf
    }

    fn calculate_word_variance(&self, doc_counts: &[usize]) -> f64 {
        if doc_counts.len() <= 1 {
            return 0.0;
        }
        
        let mean = doc_counts.iter().sum::<usize>() as f64 / doc_counts.len() as f64;
        let variance = doc_counts.iter()
            .map(|&count| {
                let diff = count as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / doc_counts.len() as f64;
        
        variance
    }

    fn calculate_significance_score(&self, tf_idf: f64, variance: f64, doc_freq: usize, total_docs: f64) -> f64 {
        // Combine multiple factors for significance:
        // 1. TF-IDF score (higher = more discriminative)
        // 2. Document frequency ratio (not too rare, not too common)
        // 3. Variance (higher = more varied usage patterns)
        
        let doc_freq_ratio = doc_freq as f64 / total_docs;
        
        // Prefer words that appear in 5-50% of documents (not too rare or too common)
        let doc_freq_score = if doc_freq_ratio < 0.05 {
            doc_freq_ratio * 20.0  // Boost rare words slightly
        } else if doc_freq_ratio > 0.5 {
            1.0 - (doc_freq_ratio - 0.5) * 2.0  // Penalize very common words
        } else {
            1.0  // Optimal range
        };
        
        // Normalize variance (higher variance = more interesting distribution)
        let variance_score = (variance / (variance + 1.0)).min(1.0);
        
        // Combine scores with weights
        tf_idf * 0.6 + doc_freq_score * 0.3 + variance_score * 0.1
    }

    fn vectorize_documents(&self, documents: &[Vec<String>], vocabulary: &[String]) -> Vec<Vec<usize>> {
        let vocab_map: HashMap<String, usize> = vocabulary.iter()
            .enumerate()
            .map(|(i, word)| (word.clone(), i))
            .collect();

        documents.iter().map(|doc| {
            doc.iter()
                .filter_map(|word| vocab_map.get(word).copied())
                .collect()
        }).collect()
    }

    fn run_lda(&self, word_docs: &[Vec<usize>], vocab_size: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let num_docs = word_docs.len();
        let num_topics = self.config.num_topics;
        
        // Initialize matrices
        let mut word_topic_counts = vec![vec![0; num_topics]; vocab_size];
        let mut doc_topic_counts = vec![vec![0; num_topics]; num_docs];
        let mut topic_counts = vec![0; num_topics];
        
        // Random initialization
        let mut doc_word_topics: Vec<Vec<usize>> = Vec::new();
        
        for (doc_id, doc) in word_docs.iter().enumerate() {
            let mut word_topics = Vec::new();
            for &word_id in doc {
                let topic = doc_id % num_topics; // Simple initialization
                word_topic_counts[word_id][topic] += 1;
                doc_topic_counts[doc_id][topic] += 1;
                topic_counts[topic] += 1;
                word_topics.push(topic);
            }
            doc_word_topics.push(word_topics);
        }

        // Simplified Gibbs sampling (just a few iterations for demonstration)
        for _iteration in 0..std::cmp::min(self.config.max_iterations, 10) {
            for (doc_id, doc) in word_docs.iter().enumerate() {
                for (word_pos, &word_id) in doc.iter().enumerate() {
                    let old_topic = doc_word_topics[doc_id][word_pos];
                    
                    // Remove current assignment
                    word_topic_counts[word_id][old_topic] -= 1;
                    doc_topic_counts[doc_id][old_topic] -= 1;
                    topic_counts[old_topic] -= 1;
                    
                    // Sample new topic (simplified)
                    let new_topic = self.sample_topic(word_id, doc_id, &word_topic_counts, &doc_topic_counts, &topic_counts);
                    
                    // Add new assignment
                    word_topic_counts[word_id][new_topic] += 1;
                    doc_topic_counts[doc_id][new_topic] += 1;
                    topic_counts[new_topic] += 1;
                    doc_word_topics[doc_id][word_pos] = new_topic;
                }
            }
        }

        // Convert counts to probabilities
        let word_topic_matrix = self.normalize_word_topic_matrix(&word_topic_counts, &topic_counts);
        let doc_topic_matrix = self.normalize_doc_topic_matrix(&doc_topic_counts);

        (word_topic_matrix, doc_topic_matrix)
    }

    fn sample_topic(&self, word_id: usize, doc_id: usize, 
                   word_topic_counts: &[Vec<usize>], 
                   doc_topic_counts: &[Vec<usize>],
                   topic_counts: &[usize]) -> usize {
        // Simplified topic sampling - in practice would use proper probability sampling
        let mut best_topic = 0;
        let mut best_score = 0.0;
        
        for topic in 0..self.config.num_topics {
            let word_prob = (word_topic_counts[word_id][topic] as f64 + self.config.beta) / 
                           (topic_counts[topic] as f64 + word_topic_counts.len() as f64 * self.config.beta);
            let doc_prob = (doc_topic_counts[doc_id][topic] as f64 + self.config.alpha) / 
                          (doc_topic_counts[doc_id].iter().sum::<usize>() as f64 + self.config.num_topics as f64 * self.config.alpha);
            
            let score = word_prob * doc_prob;
            if score > best_score {
                best_score = score;
                best_topic = topic;
            }
        }
        
        best_topic
    }

    fn normalize_word_topic_matrix(&self, counts: &[Vec<usize>], topic_counts: &[usize]) -> Vec<Vec<f64>> {
        counts.iter().map(|word_counts| {
            word_counts.iter().enumerate().map(|(topic, &count)| {
                (count as f64 + self.config.beta) / (topic_counts[topic] as f64 + counts.len() as f64 * self.config.beta)
            }).collect()
        }).collect()
    }

    fn normalize_doc_topic_matrix(&self, counts: &[Vec<usize>]) -> Vec<Vec<f64>> {
        counts.iter().map(|doc_counts| {
            let total: usize = doc_counts.iter().sum();
            if total == 0 {
                vec![1.0 / self.config.num_topics as f64; self.config.num_topics]
            } else {
                doc_counts.iter().map(|&count| {
                    (count as f64 + self.config.alpha) / (total as f64 + self.config.num_topics as f64 * self.config.alpha)
                }).collect()
            }
        }).collect()
    }

    fn extract_topics(&self, word_topic_matrix: &[Vec<f64>], vocabulary: &[String], doc_topic_matrix: &[Vec<f64>]) -> Vec<ModeledTopic> {
        // Use the new method with empty stats for backward compatibility
        let empty_stats = HashMap::new();
        self.extract_topics_with_significance(word_topic_matrix, vocabulary, doc_topic_matrix, &empty_stats)
    }

    fn extract_topics_with_significance(&self, word_topic_matrix: &[Vec<f64>], vocabulary: &[String], 
                                      doc_topic_matrix: &[Vec<f64>], word_stats: &HashMap<String, WordStatistics>) -> Vec<ModeledTopic> {
        let mut topics = Vec::new();
        
        for topic_id in 0..self.config.num_topics {
            // Get top words for this topic
            let mut word_probs: Vec<(String, f64)> = vocabulary.iter()
                .enumerate()
                .map(|(word_id, word)| (word.clone(), word_topic_matrix[word_id][topic_id]))
                .collect();
            
            word_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            word_probs.truncate(10); // Top 10 words per topic
            
            // Generate topic name considering both LDA probabilities and significance
            let topic_name = self.generate_topic_name_with_significance(&word_probs, word_stats);
            
            // Find documents primarily assigned to this topic
            let documents: Vec<usize> = doc_topic_matrix.iter()
                .enumerate()
                .filter_map(|(doc_id, probs)| {
                    let max_topic = probs.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)?;
                    if max_topic == topic_id { Some(doc_id) } else { None }
                })
                .collect();
            
            // Calculate coherence score (simplified)
            let coherence_score = self.calculate_coherence(&word_probs);
            
            topics.push(ModeledTopic {
                id: topic_id,
                name: topic_name,
                words: word_probs,
                documents,
                coherence_score,
            });
        }
        
        topics
    }

    fn generate_topic_name(&self, word_probs: &[(String, f64)]) -> String {
        // Use the enhanced method with empty stats for backward compatibility  
        let empty_stats = HashMap::new();
        self.generate_topic_name_with_significance(word_probs, &empty_stats)
    }

    fn generate_topic_name_with_significance(&self, word_probs: &[(String, f64)], word_stats: &HashMap<String, WordStatistics>) -> String {
        if word_probs.is_empty() {
            return format!("Topic");
        }
        
        // If we have significance stats, use them to enhance topic naming
        if !word_stats.is_empty() && self.config.use_significance_filtering {
            // Create combined scores: LDA probability * significance score
            let mut enhanced_words: Vec<(String, f64, f64, f64)> = word_probs.iter()
                .filter_map(|(word, lda_prob)| {
                    if let Some(stats) = word_stats.get(word) {
                        let combined_score = lda_prob * stats.significance_score.sqrt(); // Square root to moderate the effect
                        Some((word.clone(), *lda_prob, stats.significance_score, combined_score))
                    } else {
                        // Fallback for words without stats
                        Some((word.clone(), *lda_prob, 0.0, *lda_prob))
                    }
                })
                .collect();
            
            // Sort by combined score (LDA probability weighted by significance)
            enhanced_words.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
            
            // Take top 3 words based on significance-weighted scores
            let top_words: Vec<String> = enhanced_words.iter()
                .take(3)
                .map(|(word, _, _, _)| word.clone())
                .collect();
            
            // Debug output to show the selection process
            if enhanced_words.len() >= 3 {
                println!("🏷️  Topic naming (significance-aware):");
                for (i, (word, lda_prob, sig_score, combined)) in enhanced_words.iter().take(5).enumerate() {
                    let marker = if i < 3 { "✓" } else { " " };
                    println!("   {} {:<12} LDA: {:.3}, Sig: {:.3}, Combined: {:.3}", 
                            marker, word, lda_prob, sig_score, combined);
                }
            }
            
            top_words.join("-")
        } else {
            // Fallback to original simple method
            let top_words: Vec<String> = word_probs.iter().take(3).map(|(word, _)| word.clone()).collect();
            top_words.join("-")
        }
    }

    fn calculate_coherence(&self, word_probs: &[(String, f64)]) -> f64 {
        // Simplified coherence calculation based on word probabilities
        if word_probs.is_empty() {
            return 0.0;
        }
        
        let total_prob: f64 = word_probs.iter().map(|(_, prob)| prob).sum();
        let avg_prob = total_prob / word_probs.len() as f64;
        
        // Higher average probability = more coherent topic
        avg_prob
    }

    fn create_final_documents(&self, preprocessed_docs: Vec<Vec<String>>, doc_topic_matrix: &[Vec<f64>]) -> Vec<Document> {
        preprocessed_docs.into_iter()
            .enumerate()
            .map(|(id, words)| {
                let topic_distribution = doc_topic_matrix[id].clone();
                let primary_topic = topic_distribution.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                
                Document {
                    id,
                    text: words.join(" "),
                    words,
                    topic_distribution,
                    primary_topic,
                }
            })
            .collect()
    }
}
