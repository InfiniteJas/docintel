# DocIntel

AI-powered document processing and analytics platform

## Features

- Multi-format document processing (PDF, Images, Text)
- Advanced entity extraction and classification
- Sentiment analysis and keyword extraction
- Trend analysis and pattern recognition
- Real-time processing with async architecture
- REST API

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from docintel import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process a document
results = await processor.process_document("path/to/document.pdf")
print(results)
```

## Testing

```bash
pytest tests/test_docintel.py -v
```
