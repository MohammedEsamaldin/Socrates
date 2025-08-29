#!/usr/bin/env python3
"""
Debug script to test LLaVA model loading and generation issues.
"""

import os
import sys
import traceback
import torch

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cuda_availability():
    """Test CUDA availability and memory."""
    print("🔍 Testing CUDA Environment...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memory Allocated: {memory_allocated:.2f} GB")
        print(f"Memory Reserved: {memory_reserved:.2f} GB")
    print()

def test_llava_import():
    """Test importing LLaVA components."""
    print("🔍 Testing LLaVA Imports...")
    try:
        # Fix import path to match the actual module structure
        from socrates_system.mllm_evaluation.providers.llava_hf import LlavaHFGenerator
        print("✅ LlavaHFGenerator import successful")
        return True
    except Exception as e:
        print(f"❌ LlavaHFGenerator import failed: {e}")
        traceback.print_exc()
        return False

def test_llava_model_loading():
    """Test LLaVA model loading."""
    print("🔍 Testing LLaVA Model Loading...")
    
    try:
        from socrates_system.mllm_evaluation.providers.llava_hf import LlavaHFGenerator
        
        # Test different model configurations
        models_to_test = [
            ("llava-hf/llava-1.5-7b-hf", True, False),   # no_4bit=True, use_slow_tokenizer=False
            ("llava-hf/llava-1.5-7b-hf", False, True),   # no_4bit=False, use_slow_tokenizer=True
        ]
        
        for model_name, no_4bit, use_slow_tokenizer in models_to_test:
            print(f"\n📝 Testing {model_name} (no_4bit={no_4bit}, slow_tokenizer={use_slow_tokenizer})")
            
            try:
                generator = LlavaHFGenerator.get(
                    model_name, 
                    no_4bit=no_4bit, 
                    use_slow_tokenizer=use_slow_tokenizer
                )
                print(f"✅ Model {model_name} loaded successfully")
                
                # Test a simple generation
                test_prompt = "What do you see in this image?"
                
                # Try to find a test image
                possible_test_images = [
                    "socrates_system/mllm_evaluation/datasets/MME_Benchmark/OCR/images/0001.jpg",
                    "mllm_evaluation/datasets/MME_Benchmark/OCR/images/0001.jpg",
                    "datasets/MME_Benchmark/OCR/images/0001.jpg"
                ]
                
                test_image = None
                for img_path in possible_test_images:
                    if os.path.exists(img_path):
                        test_image = img_path
                        break
                
                if test_image:
                    print(f"🧪 Testing generation with {test_image}")
                    try:
                        result = generator.generate(
                            prompt=test_prompt,
                            image_path=test_image,
                            max_new_tokens=50,
                            temperature=0.1
                        )
                        print(f"✅ Generation successful: '{result[:100]}...'")
                        return True
                    except Exception as e:
                        print(f"❌ Generation failed: {e}")
                        if "enum ModelWrapper" in str(e):
                            print("💡 This is the same error from your evaluation!")
                        traceback.print_exc()
                else:
                    print(f"⚠️  No test image found, skipping generation test")
                    
            except Exception as e:
                print(f"❌ Model {model_name} loading failed: {e}")
                if "CUDA out of memory" in str(e):
                    print("💡 Try clearing CUDA cache: torch.cuda.empty_cache()")
                elif "enum ModelWrapper" in str(e):
                    print("💡 This is the same error from your evaluation!")
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ LLaVA testing failed: {e}")
        traceback.print_exc()
        return False
    
    return False

def test_transformers_version():
    """Test transformers library version compatibility."""
    print("🔍 Testing Library Versions...")
    
    try:
        import transformers
        import torch
        import PIL
        
        print(f"Transformers version: {transformers.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PIL version: {PIL.__version__}")
        
        # Check for known incompatible versions
        transformers_version = transformers.__version__
        if transformers_version.startswith("4.36") or transformers_version.startswith("4.37"):
            print("⚠️  Warning: Transformers 4.36-4.37 have known LLaVA compatibility issues")
            print("💡 Try: pip install transformers==4.35.2 or transformers>=4.38.0")
        elif transformers_version.startswith("4.40"):
            print("ℹ️  Transformers 4.40.2 should be compatible with LLaVA")
            
    except Exception as e:
        print(f"❌ Version check failed: {e}")

def clear_cuda_cache():
    """Clear CUDA cache."""
    print("🧹 Clearing CUDA cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ CUDA cache cleared")
    else:
        print("ℹ️  CUDA not available, nothing to clear")

def main():
    print("🚀 LLaVA Diagnostic Script")
    print("=" * 50)
    
    # Test environment
    test_cuda_availability()
    test_transformers_version()
    
    # Clear cache first
    clear_cuda_cache()
    
    # Test imports
    if not test_llava_import():
        print("❌ Cannot proceed - import failed")
        return
    
    # Test model loading
    success = test_llava_model_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 LLaVA is working correctly!")
    else:
        print("❌ LLaVA has issues. Recommendations:")
        print("1. Check transformers version compatibility")
        print("2. Ensure sufficient GPU memory")
        print("3. Try different model configurations")
        print("4. Consider using OpenAI GPT-4V as alternative")

if __name__ == "__main__":
    main()
