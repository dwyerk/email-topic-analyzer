# TODO List

## Features

1. ✅ Actual Topic Analysis (The Missing Core Feature)
The tool is called "email-topic-analyzer" but currently only does threading and collaboration. The biggest gap is actual topic analysis:

    1. ✅ Subject Line Analysis: Extract and cluster common themes from email subjects
    2. ✅ Content Analysis: Parse email bodies to identify discussion topics
        - Email body parsing and content extraction
        - Quote stripping and content cleaning
        - Enhanced keyword extraction from both subjects and bodies
        - Sample content display for topic verification
        - ~2.5x increase in topic discovery vs. subject-only analysis
    3. Topic Modeling: Use LDA or some other unsupervised learning to group emails by topic and derive a name for the topic.
    4. Topic Clustering: Group emails by semantic similarity
    5. Topic Evolution: Track how topics change over time
    6. ✅ Keyword Extraction: Identify the most discussed terms and concepts

2. Enhanced Analytics & Insights
    1. Time-based Analysis:
        1. Topic trends over time
        2. Peak discussion periods
        3. Seasonal patterns in topics
    2. Thread Quality Metrics:
        1. Thread depth vs. breadth analysis
        2. Response rates and engagement levels
        3. Thread resolution tracking

3. Content Processing Capabilities
    1. ✅ Email Body Parsing: Extract and process email body content
        - Text body extraction with HTML fallback
        - Quote removal (lines starting with >)
        - Email header filtering from body content
        - Signature and footer removal
        - Whitespace normalization
    2. Attachment Handling: Analyze attachments and their topics
    3. ✅ Quote Stripping: Remove quoted text to focus on new content
    4. Language Detection: Handle multi-language mailing lists
    5. HTML Content Processing: Better HTML parsing and text extraction
    6. Email Thread Context: Analyze conversation flow and context

4. Advanced Visualization & Reporting
    1. Topic Word Clouds: Visual representation of key terms
    2. Timeline Visualizations: Topic evolution over time
    3. Network Graphs: User collaboration networks
    4. Export Capabilities: JSON, CSV, HTML reports

## Testing

1. ✅ Create a smaller test dataset that can be run more quickly than the full dataset.
   - Created `sample_mbox` utility to extract samples
   - Generated `medium.mbox` (10% sample, ~12.5k messages)
   - Generated `tiny.mbox` (1% sample, ~1.2k messages)
   - Added VS Code tasks for fast testing: `sample-topics-fast`, `sample-analysis-fast`
2. 

## Performance

1. Incremental Processing: Handle large archives efficiently
2. Parallel Processing: Multi-threaded analysis for speed
3. Memory Optimization: Handle very large mbox files
4. Progress Indicators: Better user feedback during long operations

## Implementation Notes

### Email Body Content Analysis (Completed)
- **Message Structure Enhanced**: Added `body`, `processed_body` fields to Message struct
- **Content Extraction**: Uses mail-parser to extract text/html body content with fallback
- **Content Processing**: Implements regex-based quote removal, header filtering, signature stripping
- **Keyword Enhancement**: Combined subject+body analysis with frequency-based ranking
- **Stop Words**: Expanded stop word list including email-specific terms (wrote, reply, forward, etc.)
- **Sample Display**: Shows actual email content samples for topic verification
- **Performance Impact**: ~2.5x increase in topic discovery with minimal performance cost

### Current Email Body Processing Pipeline:
1. Extract text body (fallback to HTML if needed)
2. Remove quoted lines (starting with >)
3. Filter email headers that appear in body
4. Remove email signatures and footers
5. Normalize whitespace
6. Combine with subject for keyword extraction
7. Rank keywords by frequency and filter stop words

### Future Enhancements:
- Semantic similarity clustering of topics
- Time-based topic evolution tracking
- Better HTML content extraction and cleaning
- Multi-language support and internationalization
- Advanced NLP features (sentiment analysis, named entity recognition)
