#!/usr/bin/env python3
"""
Test script for DocStudio summarizer
Tests the core functionality with sample text
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import DocStudio, create_summarization_config


def test_text_summarization():
    """Test summarization with sample text."""
    print("🧪 Testing Text Summarization...")
    
    # Initialize DocStudio
    doc_studio = DocStudio()
    
    # Sample text for testing
    sample_text = """
    Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that work and react like humans. Some of the activities computers with artificial intelligence are 
    designed for include speech recognition, learning, planning, and problem solving. AI has been used 
    in various applications including medical diagnosis, stock trading, robot control, law, scientific 
    discovery, and toys. However, many AI applications are not perceived as AI: "A lot of cutting edge 
    AI has filtered into general applications, often without being called AI anymore."
    
    Machine learning is a subset of artificial intelligence that provides systems the ability to 
    automatically learn and improve from experience without being explicitly programmed. Machine learning 
    focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    Deep learning is part of a broader family of machine learning methods based on artificial neural 
    networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. 
    Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural 
    networks and convolutional neural networks have been applied to fields including computer vision, 
    speech recognition, natural language processing, audio recognition, social network filtering, 
    machine translation, bioinformatics, drug design, medical image analysis, material inspection 
    and board game programs, where they have produced results comparable to and in some cases 
    surpassing human expert performance.
    """
    
    # Test with default configuration
    print("\n📝 Testing with default configuration...")
    result = doc_studio.summarize_text(sample_text)
    
    if result['success']:
        print("✅ Summarization successful!")
        print(f"📊 Text stats: {result['text_stats']}")
        print(f"🤖 Model used: {result['summary_metadata']['model']}")
        print(f"⏱️  Processing time: {result['processing_time_seconds']:.2f} seconds")
        print(f"📄 Summary: {result['summary']}")
    else:
        print(f"❌ Summarization failed: {result['error']}")
    
    # Test with custom configuration
    print("\n📝 Testing with custom configuration...")
    config = create_summarization_config(
        max_length=200,
        min_length=30,
        temperature=0.8,
        num_beams=3
    )
    
    result_custom = doc_studio.summarize_text(sample_text, config=config)
    
    if result_custom['success']:
        print("✅ Custom configuration summarization successful!")
        print(f"📄 Custom summary: {result_custom['summary']}")
    else:
        print(f"❌ Custom configuration failed: {result_custom['error']}")
    
    return result['success']


def test_system_info():
    """Test system information and available models."""
    print("\n🔍 Testing System Information...")
    
    doc_studio = DocStudio()
    
    # Check supported formats
    formats = doc_studio.get_supported_formats()
    print(f"📁 Supported formats: {formats}")
    
    # Check available models
    models = doc_studio.get_available_models()
    print(f"🤖 Available models: {models}")
    
    return True


def main():
    """Main test function."""
    print("🚀 DocStudio Summarizer Test Suite")
    print("=" * 50)
    
    try:
        # Test system info
        test_system_info()
        
        # Test text summarization
        success = test_text_summarization()
        
        if success:
            print("\n🎉 All tests passed! DocStudio is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 