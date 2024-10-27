from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import spacy
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from sentence_transformers import SentenceTransformer
import logging
import asyncio
from datetime import datetime

class DocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = pipeline("zero-shot-classification")
        self.kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=2,
            dedupLim=0.3,
            windowsSize=2,
            top=20
        )
        
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document and extract structured information."""
        try:
            text = await self._extract_text(file_path)
            
            # Parallel processing tasks
            tasks = [
                self._extract_entities(text),
                self._analyze_sentiment(text),
                self._extract_keywords(text),
                self._classify_content(text),
                self._generate_summary(text)
            ]
            
            results = await asyncio.gather(*tasks)
            
            return {
                "entities": results[0],
                "sentiment": results[1],
                "keywords": results[2],
                "classification": results[3],
                "summary": results[4],
                "processed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise
            
    async def _extract_text(self, file_path: str) -> str:
        """Extract text from various document formats."""
        if file_path.endswith('.pdf'):
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            return pytesseract.image_to_string(Image.open(file_path))
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text."""
        sentiment = pipeline("sentiment-analysis")
        result = sentiment(text[:512])[0]  # Truncate for performance
        return {
            "sentiment": result["label"],
            "confidence": result["score"]
        }
    
    async def _extract_keywords(self, text: str) -> List[tuple]:
        """Extract key phrases from text."""
        keywords = self.kw_extractor.extract_keywords(text)
        return [(kw, score) for kw, score in keywords]
    
    async def _classify_content(self, text: str) -> Dict[str, float]:
        """Classify document content."""
        categories = [
            "financial_report", "legal_document", "technical_specification",
            "business_proposal", "internal_memo"
        ]
        result = self.classifier(text[:512], categories)
        return dict(zip(result['labels'], result['scores']))
    
    async def _generate_summary(self, text: str, max_length: int = 150) -> str:
        """Generate a concise summary of the document."""
        summarizer = pipeline("summarization")
        summary = summarizer(text[:1024], max_length=max_length, min_length=30)
        return summary[0]['summary_text']

class DocumentAnalytics:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.embeddings = {}
        
    def process_document_batch(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process batch of documents for trend analysis."""
        df = pd.DataFrame(documents)
        
        # Calculate document similarity matrix
        embeddings = self.sentence_model.encode([doc.get('text', '') for doc in documents])
        similarity_matrix = torch.nn.functional.cosine_similarity(
            torch.tensor(embeddings).unsqueeze(0),
            torch.tensor(embeddings).unsqueeze(1)
        )
        
        # Extract trends and patterns
        trends = self._analyze_trends(df)
        
        return {
            "similarity_matrix": similarity_matrix.numpy(),
            "trends": trends,
            "statistics": self._calculate_statistics(df)
        }
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in document batch."""
        trends = {
            "temporal": self._analyze_temporal_patterns(df),
            "topical": self._analyze_topical_patterns(df),
            "entity_frequency": self._analyze_entity_frequency(df)
        }
        return trends
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical measures for the document batch."""
        return {
            "document_count": len(df),
            "average_sentiment": df['sentiment'].mean() if 'sentiment' in df else None,
            "common_categories": df['classification'].value_counts().to_dict() if 'classification' in df else None
        }

# FastAPI application
app = FastAPI(title="DocIntel API")

class ProcessingResponse(BaseModel):
    document_id: str
    analysis_results: Dict[str, Any]

@app.post("/process-document/", response_model=ProcessingResponse)
async def process_document_endpoint(file: UploadFile = File(...)):
    processor = DocumentProcessor()
    results = await processor.process_document(file.filename)
    
    return ProcessingResponse(
        document_id=file.filename,
        analysis_results=results
    )

@app.post("/batch-analysis/")
async def batch_analysis_endpoint(documents: List[Dict[str, Any]]):
    analytics = DocumentAnalytics()
    results = analytics.process_document_batch(documents)
    return results
