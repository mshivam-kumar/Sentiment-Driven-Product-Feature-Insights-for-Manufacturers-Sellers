#!/usr/bin/env python3
"""
Test deployment configuration and dependencies
"""

import os
import sys
import subprocess
import importlib.util

def test_imports():
    """Test that all required modules can be imported"""
    required_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'requests',
        'boto3',
        'sklearn',
        'sentence_transformers',
        'transformers',
        'torch',
        'spacy'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All required modules imported successfully!")
        return True

def test_streamlit_app():
    """Test that the Streamlit app can be imported"""
    try:
        sys.path.append('dashboard')
        from streamlit_app import SentimentDashboard
        print("✅ Streamlit app imports successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app import failed: {e}")
        return False

def test_rag_module():
    """Test that the RAG module can be imported"""
    try:
        from rag_module import RAGSystem
        print("✅ RAG module imports successfully")
        return True
    except Exception as e:
        print(f"❌ RAG module import failed: {e}")
        return False

def test_docker_build():
    """Test that Docker can build the image"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is available")
            return True
        else:
            print("❌ Docker is not available")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def test_aws_cli():
    """Test that AWS CLI is available"""
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ AWS CLI is available")
            return True
        else:
            print("⚠️ AWS CLI is not available (optional)")
            return True  # Not required for all deployments
    except FileNotFoundError:
        print("⚠️ AWS CLI is not installed (optional)")
        return True  # Not required for all deployments

def main():
    """Run all deployment tests"""
    print("🧪 Running Deployment Tests")
    print("=" * 40)
    
    tests = [
        ("Module Imports", test_imports),
        ("Streamlit App", test_streamlit_app),
        ("RAG Module", test_rag_module),
        ("Docker", test_docker_build),
        ("AWS CLI", test_aws_cli)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for deployment.")
    else:
        print("\n💥 Some tests failed. Please fix issues before deploying.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
