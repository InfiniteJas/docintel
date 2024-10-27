import pytest
import asyncio
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
from fastapi.testclient import TestClient
from docintel.core import DocumentProcessor, DocumentAnalytics
from docintel.api import app

# Фикстуры для тестов
@pytest.fixture
def sample_text():
    return """
    Apple Inc. reported strong quarterly earnings, with revenue reaching $89.5 billion.
    The company's CEO Tim Cook expressed optimism about future growth prospects.
    Customer satisfaction remains high, though some concerns about supply chain were noted.
    """

@pytest.fixture
def sample_pdf_path(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    # Создаем тестовый PDF файл
    return str(pdf_path)

@pytest.fixture
def document_processor():
    return DocumentProcessor()

@pytest.fixture
def document_analytics():
    return DocumentAnalytics()

@pytest.fixture
def api_client():
    return TestClient(app)

# Тесты для DocumentProcessor
class TestDocumentProcessor:
    @pytest.mark.asyncio
    async def test_extract_entities(self, document_processor, sample_text):
        entities = await document_processor._extract_entities(sample_text)
        assert isinstance(entities, dict)
        assert "ORG" in entities
        assert "Apple Inc." in entities["ORG"]
        assert "PERSON" in entities
        assert "Tim Cook" in entities["PERSON"]
        assert "MONEY" in entities
        assert "$89.5 billion" in entities["MONEY"]

    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, document_processor, sample_text):
        sentiment = await document_processor._analyze_sentiment(sample_text)
        assert isinstance(sentiment, dict)
        assert "sentiment" in sentiment
        assert "confidence" in sentiment
        assert isinstance(sentiment["confidence"], float)
        assert 0 <= sentiment["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_extract_keywords(self, document_processor, sample_text):
        keywords = await document_processor._extract_keywords(sample_text)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(k, tuple) and len(k) == 2 for k in keywords)

    @pytest.mark.asyncio
    async def test_classify_content(self, document_processor, sample_text):
        classification = await document_processor._classify_content(sample_text)
        assert isinstance(classification, dict)
        assert len(classification) > 0
        assert all(isinstance(v, float) for v in classification.values())
        assert all(0 <= v <= 1 for v in classification.values())

    @pytest.mark.asyncio
    async def test_generate_summary(self, document_processor, sample_text):
        summary = await document_processor._generate_summary(sample_text)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) <= 150

# Тесты для DocumentAnalytics
class TestDocumentAnalytics:
    def test_process_document_batch(self, document_analytics):
        documents = [
            {
                "text": "Sample document 1",
                "sentiment": 0.8,
                "classification": "financial_report"
            },
            {
                "text": "Sample document 2",
                "sentiment": 0.6,
                "classification": "legal_document"
            }
        ]
        
        results = document_analytics.process_document_batch(documents)
        assert isinstance(results, dict)
        assert "similarity_matrix" in results
        assert "trends" in results
        assert "statistics" in results
        
        # Проверка similarity matrix
        assert isinstance(results["similarity_matrix"], np.ndarray)
        assert results["similarity_matrix"].shape == (2, 2)
        
        # Проверка статистики
        assert results["statistics"]["document_count"] == 2
        assert isinstance(results["statistics"]["average_sentiment"], float)

    def test_analyze_trends(self, document_analytics):
        df = pd.DataFrame({
            "text": ["doc1", "doc2"],
            "sentiment": [0.8, 0.6],
            "classification": ["type1", "type2"]
        })
        
        trends = document_analytics._analyze_trends(df)
        assert isinstance(trends, dict)
        assert "temporal" in trends
        assert "topical" in trends
        assert "entity_frequency" in trends

# Тесты API endpoints
class TestAPI:
    def test_process_document_endpoint(self, api_client):
        # Подготовка тестового файла
        with open("test.txt", "w") as f:
            f.write("Test document content")
        
        with open("test.txt", "rb") as f:
            response = api_client.post(
                "/process-document/",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert "analysis_results" in data

    def test_batch_analysis_endpoint(self, api_client):
        documents = [
            {
                "text": "Sample document 1",
                "sentiment": 0.8,
                "classification": "financial_report"
            },
            {
                "text": "Sample document 2",
                "sentiment": 0.6,
                "classification": "legal_document"
            }
        ]
        
        response = api_client.post(
            "/batch-analysis/",
            json=documents
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "similarity_matrix" in data
        assert "trends" in data
        assert "statistics" in data

# Интеграционные тесты
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_document_processing(self, document_processor, sample_text):
        results = await document_processor.process_document("test.txt")
        assert isinstance(results, dict)
        assert all(k in results for k in [
            "entities",
            "sentiment",
            "keywords",
            "classification",
            "summary",
            "processed_at"
        ])

    def test_end_to_end_processing(self, api_client, sample_text):
        # Создаем тестовый документ
        with open("test.txt", "w") as f:
            f.write(sample_text)
        
        # Тестируем обработку документа
        with open("test.txt", "rb") as f:
            response = api_client.post(
                "/process-document/",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Тестируем batch analysis с результатами
        batch_response = api_client.post(
            "/batch-analysis/",
            json=[data["analysis_results"]]
        )
        
        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert all(k in batch_data for k in ["similarity_matrix", "trends", "statistics"])

if __name__ == "__main__":
    pytest.main(["-v"])
