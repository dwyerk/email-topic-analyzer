# TODO List

## Features

1. ✅ Actual Topic Analysis (The Missing Core Feature)
The tool is called "email-topic-analyzer" but currently only does threading and collaboration. The biggest gap is actual topic analysis:

    1. ✅ Subject Line Analysis: Extract and cluster common themes from email subjects
    2. Content Analysis: Parse email bodies to identify discussion topics
    3. Topic Clustering: Group emails by semantic similarity
    4. Topic Evolution: Track how topics change over time
    5. ✅ Keyword Extraction: Identify the most discussed terms and concepts

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
    1. Email Body Parsing: Currently only uses headers - add body content analysis
    2. Attachment Handling: Analyze attachments and their topics
    3. Quote Stripping: Remove quoted text to focus on new content
    4. Language Detection: Handle multi-language mailing lists

4. Advanced Visualization & Reporting
    1. Topic Word Clouds: Visual representation of key terms
    2. Timeline Visualizations: Topic evolution over time
    3. Network Graphs: User collaboration networks
    4. Export Capabilities: JSON, CSV, HTML reports

## Testing

1. ✅ Create a smaller test dataset that can be run more quickly than the full dataset.
   - Created `sample_mbox` utility to extract samples
   - Generated `sample_takoma.mbox` (10% sample, ~12.5k messages)
   - Generated `tiny_takoma.mbox` (1% sample, ~1.2k messages)
   - Added VS Code tasks for fast testing: `sample-topics-fast`, `sample-analysis-fast`
2. 

## Performance

1. Incremental Processing: Handle large archives efficiently
2. Parallel Processing: Multi-threaded analysis for speed
3. Memory Optimization: Handle very large mbox files
4. Progress Indicators: Better user feedback during long operations
