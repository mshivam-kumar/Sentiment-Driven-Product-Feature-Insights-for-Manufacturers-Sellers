#!/usr/bin/env python3
"""
Basic deployment test - tests core functionality without all dependencies
"""

import os
import sys
import subprocess

def test_core_imports():
    """Test core imports that are essential"""
    core_modules = [
        'streamlit',
        'pandas', 
        'numpy',
        'requests',
        'boto3'
    ]
    
    failed_imports = []
    
    for module in core_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_streamlit_app_structure():
    """Test that the Streamlit app file exists and is readable"""
    app_path = "dashboard/streamlit_app.py"
    
    if not os.path.exists(app_path):
        print(f"❌ Streamlit app not found at {app_path}")
        return False
    
    try:
        with open(app_path, 'r') as f:
            content = f.read()
            if 'streamlit' in content and 'st.' in content:
                print("✅ Streamlit app structure looks good")
                return True
            else:
                print("❌ Streamlit app doesn't seem to contain Streamlit code")
                return False
    except Exception as e:
        print(f"❌ Error reading Streamlit app: {e}")
        return False

def test_dockerfile():
    """Test that Dockerfile exists and is valid"""
    if not os.path.exists("Dockerfile"):
        print("❌ Dockerfile not found")
        return False
    
    try:
        with open("Dockerfile", 'r') as f:
            content = f.read()
            if 'FROM python' in content and 'streamlit' in content:
                print("✅ Dockerfile looks good")
                return True
            else:
                print("❌ Dockerfile doesn't seem to be configured for Streamlit")
                return False
    except Exception as e:
        print(f"❌ Error reading Dockerfile: {e}")
        return False

def test_requirements():
    """Test that requirements.txt exists and has essential packages"""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open("requirements.txt", 'r') as f:
            content = f.read()
            essential_packages = ['streamlit', 'pandas', 'numpy', 'boto3']
            missing = []
            
            for package in essential_packages:
                if package not in content:
                    missing.append(package)
            
            if missing:
                print(f"❌ Missing essential packages: {missing}")
                return False
            else:
                print("✅ requirements.txt has essential packages")
                return True
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def test_docker_available():
    """Test if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is available")
            return True
        else:
            print("❌ Docker is not working properly")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def main():
    """Run basic deployment tests"""
    print("🧪 Running Basic Deployment Tests")
    print("=" * 40)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Streamlit App Structure", test_streamlit_app_structure),
        ("Dockerfile", test_dockerfile),
        ("Requirements", test_requirements),
        ("Docker Available", test_docker_available)
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
        print("\n🎉 All basic tests passed! Ready for deployment.")
        print("\n🚀 You can now run:")
        print("   ./deploy.sh")
        print("   or")
        print("   docker build -t sentiment-analysis-app .")
        print("   docker run -p 8501:8501 sentiment-analysis-app")
    else:
        print("\n💥 Some tests failed. Please fix issues before deploying.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
