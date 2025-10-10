#!/usr/bin/env python3
"""
Test script to validate PEFT library and fine-tuned model loading
"""

import os
import sys
import traceback

def test_peft_imports():
    """Test if PEFT library can be imported"""
    print("🔍 Testing PEFT library imports...")
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print("✅ All required libraries imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_fine_tuned_model_path():
    """Test if fine-tuned model path exists and has required files"""
    print("\n🔍 Testing fine-tuned model path...")
    
    # Try different possible paths
    possible_paths = [
        "./backend/fine_tuned_tinyllama",
        "./fine_tuned_tinyllama",
        "/app/fine_tuned_tinyllama"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found fine-tuned model at: {path}")
            
            # Check required files
            required_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json"
            ]
            
            missing_files = []
            for file in required_files:
                file_path = os.path.join(path, file)
                if os.path.exists(file_path):
                    print(f"  ✅ {file} exists")
                else:
                    print(f"  ❌ {file} missing")
                    missing_files.append(file)
            
            if not missing_files:
                print(f"✅ All required files present in {path}")
                return path
            else:
                print(f"❌ Missing files: {missing_files}")
        else:
            print(f"❌ Path not found: {path}")
    
    return None

def test_peft_config_loading(model_path):
    """Test loading PeftConfig"""
    print(f"\n🔍 Testing PeftConfig loading from {model_path}...")
    try:
        from peft import PeftConfig
        config = PeftConfig.from_pretrained(model_path)
        print("✅ PeftConfig loaded successfully")
        print(f"  - PEFT Type: {config.peft_type}")
        print(f"  - Base Model: {config.base_model_name_or_path}")
        print(f"  - Target Modules: {config.target_modules}")
        print(f"  - LoRA Rank: {config.r}")
        print(f"  - LoRA Alpha: {config.lora_alpha}")
        return config
    except Exception as e:
        print(f"❌ PeftConfig loading failed: {e}")
        traceback.print_exc()
        return None

def test_base_model_loading():
    """Test loading base TinyLlama model"""
    print("\n🔍 Testing base model loading...")
    try:
        from transformers import AutoModelForCausalLM
        import torch
        
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Loading base model: {base_model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("✅ Base model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Base model loading failed: {e}")
        traceback.print_exc()
        return None

def test_peft_model_loading(base_model, model_path):
    """Test loading PEFT model with different methods"""
    print(f"\n🔍 Testing PEFT model loading from {model_path}...")
    
    methods = [
        ("Method 1: PeftConfig + PeftModel", lambda: test_method1(base_model, model_path)),
        ("Method 2: Direct PeftModel", lambda: test_method2(base_model, model_path)),
        ("Method 3: With trust_remote_code", lambda: test_method3(base_model, model_path)),
    ]
    
    for method_name, method_func in methods:
        print(f"\n  {method_name}...")
        try:
            result = method_func()
            if result:
                print(f"  ✅ {method_name} successful!")
                return result
            else:
                print(f"  ❌ {method_name} failed")
        except Exception as e:
            print(f"  ❌ {method_name} failed: {e}")
    
    return None

def test_method1(base_model, model_path):
    """Method 1: PeftConfig + PeftModel"""
    from peft import PeftConfig, PeftModel
    config = PeftConfig.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, model_path, config=config, local_files_only=True)
    return model

def test_method2(base_model, model_path):
    """Method 2: Direct PeftModel"""
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
    return model

def test_method3(base_model, model_path):
    """Method 3: With trust_remote_code"""
    from peft import PeftModel
    import torch
    model = PeftModel.from_pretrained(
        base_model, 
        model_path, 
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    return model

def test_tokenizer_loading(model_path):
    """Test loading tokenizer"""
    print(f"\n🔍 Testing tokenizer loading from {model_path}...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🚀 Starting PEFT library and fine-tuned model validation...")
    
    # Test 1: Import libraries
    if not test_peft_imports():
        print("❌ Cannot proceed without required libraries")
        return False
    
    # Test 2: Find fine-tuned model path
    model_path = test_fine_tuned_model_path()
    if not model_path:
        print("❌ Cannot find fine-tuned model path")
        return False
    
    # Test 3: Load PeftConfig
    config = test_peft_config_loading(model_path)
    if not config:
        print("❌ Cannot load PeftConfig")
        return False
    
    # Test 4: Load base model
    base_model = test_base_model_loading()
    if not base_model:
        print("❌ Cannot load base model")
        return False
    
    # Test 5: Load PEFT model
    peft_model = test_peft_model_loading(base_model, model_path)
    if not peft_model:
        print("❌ Cannot load PEFT model with any method")
        return False
    
    # Test 6: Load tokenizer
    tokenizer = test_tokenizer_loading(model_path)
    if not tokenizer:
        print("❌ Cannot load tokenizer")
        return False
    
    print("\n🎉 All tests passed! Fine-tuned model loading should work.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
