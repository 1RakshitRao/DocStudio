#!/usr/bin/env python3
"""
Example usage of DocStudio summarizer
Demonstrates various features and use cases
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import DocStudio, create_summarization_config


def example_text_summarization():
    """Example: Summarize text input."""
    print("📝 Example: Text Summarization")
    print("-" * 40)
    
    doc_studio = DocStudio()
    
    # Sample text about AI
    text = """
    Artificial intelligence (AI) is transforming the way we live and work. From virtual assistants 
    like Siri and Alexa to recommendation systems on Netflix and Amazon, AI is becoming increasingly 
    integrated into our daily lives. Machine learning, a subset of AI, enables computers to learn 
    from data without being explicitly programmed. Deep learning, which uses neural networks with 
    multiple layers, has achieved remarkable success in image recognition, natural language processing, 
    and autonomous vehicles.
    
    The field of AI has seen tremendous growth in recent years, driven by advances in computing power, 
    the availability of large datasets, and improvements in algorithms. Companies across industries 
    are investing heavily in AI to gain competitive advantages, improve efficiency, and create new 
    products and services. However, this rapid advancement also raises important questions about 
    privacy, job displacement, and the ethical use of AI technology.
    
    Despite the challenges, AI continues to evolve and expand into new domains. Researchers are 
    working on developing more sophisticated AI systems that can understand context, reason about 
    complex problems, and interact more naturally with humans. The future of AI holds immense 
    potential for solving some of humanity's most pressing challenges, from climate change to 
    healthcare, while also presenting new opportunities for innovation and growth.
    """
    
    # Basic summarization
    print("🔧 Basic summarization...")
    result = doc_studio.summarize_text(text)
    
    if result['success']:
        print(f"✅ Summary: {result['summary']}")
        print(f"⏱️  Time: {result['processing_time_seconds']:.2f}s")
        print(f"📊 Stats: {result['text_stats']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    print()


def example_custom_configuration():
    """Example: Custom summarization configuration."""
    print("⚙️ Example: Custom Configuration")
    print("-" * 40)
    
    doc_studio = DocStudio()
    
    text = """
    Climate change is one of the most pressing challenges facing humanity today. The Earth's average 
    temperature has increased by about 1.1 degrees Celsius since pre-industrial times, primarily due 
    to human activities such as burning fossil fuels and deforestation. This warming is causing 
    widespread changes in weather patterns, rising sea levels, and more frequent extreme weather events.
    
    The scientific consensus is clear: human activities are the dominant cause of observed warming 
    since the mid-20th century. The Intergovernmental Panel on Climate Change (IPCC) has warned that 
    limiting global warming to 1.5 degrees Celsius above pre-industrial levels would require rapid, 
    far-reaching, and unprecedented changes in all aspects of society.
    
    Addressing climate change requires a comprehensive approach involving governments, businesses, 
    and individuals. This includes transitioning to renewable energy sources, improving energy 
    efficiency, protecting and restoring forests, and developing new technologies for carbon capture 
    and storage. The transition to a low-carbon economy also presents opportunities for innovation, 
    job creation, and improved public health.
    """
    
    # Custom configuration for shorter, more focused summary
    config = create_summarization_config(
        max_length=150,
        min_length=30,
        temperature=0.7,  # Slightly more creative
        num_beams=3
    )
    
    print("🔧 Custom configuration summarization...")
    result = doc_studio.summarize_text(text, config=config)
    
    if result['success']:
        print(f"✅ Summary: {result['summary']}")
        print(f"⚙️ Config: max_length={config.max_length}, temperature={config.temperature}")
    else:
        print(f"❌ Error: {result['error']}")
    
    print()


def example_batch_processing():
    """Example: Batch processing multiple texts."""
    print("📦 Example: Batch Processing")
    print("-" * 40)
    
    doc_studio = DocStudio()
    
    # Multiple texts to summarize
    texts = [
        "The Internet of Things (IoT) refers to the network of physical devices embedded with sensors, software, and connectivity that enables them to collect and exchange data. This technology is revolutionizing industries from manufacturing to healthcare.",
        
        "Blockchain technology is a decentralized digital ledger that records transactions across multiple computers securely and transparently. Originally developed for Bitcoin, it now has applications in finance, supply chain management, and digital identity verification.",
        
        "Quantum computing represents a paradigm shift in computational power, using quantum mechanical phenomena like superposition and entanglement to process information. While still in early stages, it promises to solve complex problems that are currently intractable for classical computers."
    ]
    
    print(f"🔧 Processing {len(texts)} texts...")
    
    for i, text in enumerate(texts, 1):
        print(f"\n📄 Text {i}:")
        result = doc_studio.summarize_text(text)
        
        if result['success']:
            print(f"✅ Summary: {result['summary']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    print()


def example_system_information():
    """Example: Get system information."""
    print("🔍 Example: System Information")
    print("-" * 40)
    
    doc_studio = DocStudio()
    
    # Get supported formats
    formats = doc_studio.get_supported_formats()
    print(f"📁 Supported formats: {formats}")
    
    # Get available models
    models = doc_studio.get_available_models()
    print(f"🤖 Available models: {models}")
    
    print()


def example_error_handling():
    """Example: Error handling."""
    print("⚠️ Example: Error Handling")
    print("-" * 40)
    
    doc_studio = DocStudio()
    
    # Empty text
    print("🔧 Testing empty text...")
    result = doc_studio.summarize_text("")
    print(f"Result: {result['success']} - {result.get('error', 'No error')}")
    
    # Very short text
    print("🔧 Testing very short text...")
    result = doc_studio.summarize_text("Hello world.")
    print(f"Result: {result['success']} - {result.get('summary', result.get('error', 'No result'))}")
    
    print()


def main():
    """Run all examples."""
    print("🚀 DocStudio Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_system_information()
        example_text_summarization()
        example_custom_configuration()
        example_batch_processing()
        example_error_handling()
        
        print("🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 