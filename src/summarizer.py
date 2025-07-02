"""
Abstractive Summarizer Module
Provides both local (HuggingFace) and cloud (OpenAI) summarization capabilities
"""

import os
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer

try:
    import openai
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for summarization parameters."""
    max_length: int = 512
    min_length: int = 50
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_beams: int = 4
    early_stopping: bool = True


class LocalSummarizer:
    """Local summarization using HuggingFace Transformers."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize local summarizer.
        
        Args:
            model_name: HuggingFace model name for summarization
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.summarizer_pipeline = None
        
        logger.info(f"Initializing local summarizer with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model and tokenizer."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Create pipeline for easier use
            self.summarizer_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def chunk_text(self, text: str, max_chunk_length: int = 1024) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Input text to chunk
            max_chunk_length: Maximum length of each chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max length, start new chunk
            if len(current_chunk + sentence) > max_chunk_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize(self, text: str, config: SummarizationConfig = None) -> Dict[str, Any]:
        """
        Generate abstractive summary using local model.
        
        Args:
            text: Input text to summarize
            config: Summarization configuration
            
        Returns:
            Dictionary containing summary and metadata
        """
        if config is None:
            config = SummarizationConfig()
        
        if not text.strip():
            return {"summary": "", "error": "Empty text provided"}
        
        try:
            # Chunk text if it's too long
            chunks = self.chunk_text(text)
            
            if len(chunks) == 1:
                # Single chunk - direct summarization
                summary = self._summarize_chunk(chunks[0], config)
            else:
                # Multiple chunks - summarize each and then combine
                chunk_summaries = []
                for chunk in chunks:
                    chunk_summary = self._summarize_chunk(chunk, config)
                    chunk_summaries.append(chunk_summary)
                
                # Combine chunk summaries
                combined_text = " ".join(chunk_summaries)
                summary = self._summarize_chunk(combined_text, config)
            
            return {
                "summary": summary,
                "model": self.model_name,
                "chunks_processed": len(chunks),
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {"summary": "", "error": str(e)}
    
    def _summarize_chunk(self, text: str, config: SummarizationConfig) -> str:
        """Summarize a single text chunk."""
        try:
            # Use pipeline for easier processing
            result = self.summarizer_pipeline(
                text,
                max_length=config.max_length,
                min_length=config.min_length,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                num_beams=config.num_beams,
                early_stopping=config.early_stopping,
                truncation=True
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Chunk summarization error: {e}")
            return text[:config.max_length]  # Fallback to truncation


class OpenAISummarizer:
    """Cloud-based summarization using OpenAI API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI summarizer.
        
        Args:
            api_key: OpenAI API key (will try to get from environment if not provided)
        """
        if openai is None:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        openai.api_key = self.api_key
        logger.info("OpenAI summarizer initialized")
    
    def summarize(self, text: str, config: SummarizationConfig = None) -> Dict[str, Any]:
        """
        Generate abstractive summary using OpenAI.
        
        Args:
            text: Input text to summarize
            config: Summarization configuration (some parameters may not apply to OpenAI)
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not text.strip():
            return {"summary": "", "error": "Empty text provided"}
        
        try:
            # Prepare prompt
            prompt = f"""Please provide a comprehensive and accurate summary of the following text. 
            Focus on the main points, key insights, and important details. 
            Make the summary clear, coherent, and well-structured.

            Text to summarize:
            {text}

            Summary:"""
            
            # Make API call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional document summarizer. Create clear, accurate, and comprehensive summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.max_length if config else 500,
                temperature=config.temperature if config else 0.7,
                top_p=config.top_p if config else 0.9
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                "summary": summary,
                "model": "gpt-3.5-turbo",
                "tokens_used": response.usage.total_tokens,
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"OpenAI summarization error: {e}")
            return {"summary": "", "error": str(e)}


class HybridSummarizer:
    """Hybrid summarizer that can use both local and cloud models."""
    
    def __init__(self, local_model: str = "facebook/bart-large-cnn", openai_key: str = None):
        """
        Initialize hybrid summarizer.
        
        Args:
            local_model: HuggingFace model name for local summarization
            openai_key: OpenAI API key for cloud summarization
        """
        self.local_summarizer = LocalSummarizer(local_model)
        self.openai_summarizer = None
        
        if openai_key or os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_summarizer = OpenAISummarizer(openai_key)
                logger.info("Hybrid summarizer initialized with both local and OpenAI capabilities")
            except Exception as e:
                logger.warning(f"OpenAI summarizer not available: {e}")
        else:
            logger.info("Hybrid summarizer initialized with local capabilities only")
    
    def summarize(self, text: str, use_openai: bool = False, config: SummarizationConfig = None) -> Dict[str, Any]:
        """
        Generate summary using either local or cloud model.
        
        Args:
            text: Input text to summarize
            use_openai: Whether to use OpenAI (if available)
            config: Summarization configuration
            
        Returns:
            Dictionary containing summary and metadata
        """
        if use_openai and self.openai_summarizer:
            return self.openai_summarizer.summarize(text, config)
        else:
            return self.local_summarizer.summarize(text, config)
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get information about available models."""
        return {
            "local": True,
            "openai": self.openai_summarizer is not None
        } 