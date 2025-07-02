"""
Main Application Module
Combines document processing and summarization into a unified interface
"""

import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .document_processor import DocumentProcessor
from .summarizer import HybridSummarizer, SummarizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocStudio:
    """
    Main application class for document summarization.
    Combines document processing and summarization capabilities.
    """
    
    def __init__(self, local_model: str = "facebook/bart-large-cnn", openai_key: str = None):
        """
        Initialize DocStudio.
        
        Args:
            local_model: HuggingFace model name for local summarization
            openai_key: OpenAI API key for cloud summarization
        """
        self.document_processor = DocumentProcessor()
        self.summarizer = HybridSummarizer(local_model, openai_key)
        
        logger.info("DocStudio initialized successfully")
    
    def process_document(self, file_path: str, use_openai: bool = False, 
                        config: SummarizationConfig = None) -> Dict[str, Any]:
        """
        Process a document and generate a summary.
        
        Args:
            file_path: Path to the document file
            use_openai: Whether to use OpenAI for summarization
            config: Summarization configuration
            
        Returns:
            Dictionary containing processing results and summary
        """
        start_time = time.time()
        
        try:
            # Extract text from document
            logger.info(f"Processing document: {file_path}")
            extraction_result = self.document_processor.extract_text(file_path)
            
            # Clean the extracted text
            cleaned_text = self.document_processor.clean_text(extraction_result['text'])
            
            # Get document statistics
            stats = self.document_processor.get_document_stats(cleaned_text)
            
            # Generate summary
            logger.info("Generating summary...")
            summary_result = self.summarizer.summarize(cleaned_text, use_openai, config)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare final result
            result = {
                'file_path': file_path,
                'file_info': {
                    'type': extraction_result['file_type'],
                    'size': extraction_result['file_size'],
                    'metadata': extraction_result['metadata']
                },
                'document_stats': stats,
                'summary': summary_result.get('summary', ''),
                'summary_metadata': {
                    'model': summary_result.get('model', ''),
                    'provider': summary_result.get('provider', 'local'),
                    'chunks_processed': summary_result.get('chunks_processed', 1),
                    'tokens_used': summary_result.get('tokens_used', 0)
                },
                'processing_time_seconds': processing_time,
                'success': 'summary' in summary_result and not summary_result.get('error'),
                'error': summary_result.get('error', None)
            }
            
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                'file_path': file_path,
                'success': False,
                'error': str(e),
                'processing_time_seconds': time.time() - start_time
            }
    
    def summarize_text(self, text: str, use_openai: bool = False, 
                      config: SummarizationConfig = None) -> Dict[str, Any]:
        """
        Summarize raw text input.
        
        Args:
            text: Input text to summarize
            use_openai: Whether to use OpenAI for summarization
            config: Summarization configuration
            
        Returns:
            Dictionary containing summary results
        """
        start_time = time.time()
        
        try:
            # Clean the text
            cleaned_text = self.document_processor.clean_text(text)
            
            # Get text statistics
            stats = self.document_processor.get_document_stats(cleaned_text)
            
            # Generate summary
            summary_result = self.summarizer.summarize(cleaned_text, use_openai, config)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                'text_stats': stats,
                'summary': summary_result.get('summary', ''),
                'summary_metadata': {
                    'model': summary_result.get('model', ''),
                    'provider': summary_result.get('provider', 'local'),
                    'chunks_processed': summary_result.get('chunks_processed', 1),
                    'tokens_used': summary_result.get('tokens_used', 0)
                },
                'processing_time_seconds': processing_time,
                'success': 'summary' in summary_result and not summary_result.get('error'),
                'error': summary_result.get('error', None)
            }
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time_seconds': time.time() - start_time
            }
    
    def get_supported_formats(self) -> list:
        """Get list of supported document formats."""
        return list(self.document_processor.supported_formats)
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get information about available summarization models."""
        return self.summarizer.get_available_models()
    
    def batch_process(self, file_paths: list, use_openai: bool = False, 
                     config: SummarizationConfig = None) -> list:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            use_openai: Whether to use OpenAI for summarization
            config: Summarization configuration
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing file {i}/{len(file_paths)}: {file_path}")
            result = self.process_document(file_path, use_openai, config)
            results.append(result)
        
        return results


def create_summarization_config(max_length: int = 512, min_length: int = 300,
                               temperature: float = 1.0, num_beams: int = 4) -> SummarizationConfig:
    """
    Create a summarization configuration with custom parameters.
    
    Args:
        max_length: Maximum length of the summary
        min_length: Minimum length of the summary
        temperature: Sampling temperature (higher = more creative)
        num_beams: Number of beams for beam search
        
    Returns:
        SummarizationConfig object
    """
    return SummarizationConfig(
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        num_beams=num_beams
    )


# Example usage and testing
if __name__ == "__main__":
    # Initialize DocStudio
    doc_studio = DocStudio()
    
    # Example: Process a document
    # result = doc_studio.process_document("path/to/document.pdf")
    # print(result)
    
    # Example: Summarize text
    sample_text = """
    Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that work and react like humans. Some of the activities computers with artificial intelligence are 
    designed for include speech recognition, learning, planning, and problem solving. AI has been used 
    in various applications including medical diagnosis, stock trading, robot control, law, scientific 
    discovery, and toys. However, many AI applications are not perceived as AI: "A lot of cutting edge 
    AI has filtered into general applications, often without being called AI because once something 
    becomes useful enough and common enough it's not labeled AI anymore."
    """
    
    result = doc_studio.summarize_text(sample_text)
    print("Summary:", result['summary'])
    print("Processing time:", result['processing_time_seconds'], "seconds") 