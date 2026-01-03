#!/usr/bin/env python3
"""
Diagnostic script to verify iDAQ setup.

Run this to check:
- Environment variables
- API keys validity
- Firebase configuration
- Dependencies
- File structure
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def color_text(text: str, color: str) -> str:
    """Add color to terminal text."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def check_mark(passed: bool) -> str:
    """Return check or cross mark."""
    return color_text("✓", "green") if passed else color_text("✗", "red")

def header(text: str):
    """Print section header."""
    print(f"\n{color_text('='*60, 'blue')}")
    print(color_text(f"  {text}", "blue"))
    print(color_text('='*60, 'blue'))

def test_result(name: str, passed: bool, details: str = ""):
    """Print test result."""
    mark = check_mark(passed)
    status = color_text("PASS", "green") if passed else color_text("FAIL", "red")
    print(f"{mark} {name}: {status}")
    if details:
        print(f"  → {details}")

def main():
    """Run all diagnostic checks."""
    print(color_text("\n🔍 iDAQ Diagnostics System Check\n", "blue"))
    
    all_passed = True
    
    # Check .env file
    header("Environment Configuration")
    
    env_file = Path(".env")
    env_exists = env_file.exists()
    test_result("Environment file (.env)", env_exists)
    if not env_exists:
        print(color_text("  ⚠ Create .env file from .env.example", "yellow"))
        all_passed = False
    else:
        load_dotenv()
    
    # Check OpenAI
    header("OpenAI Configuration")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    has_openai = bool(openai_key and openai_key.startswith("sk-"))
    test_result("OpenAI API key", has_openai)
    
    if has_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            # Test with a minimal request
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            test_result("OpenAI API connection", True, "Successfully connected")
        except ImportError:
            test_result("OpenAI library", False, "Run: pip install openai")
            all_passed = False
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg:
                test_result("OpenAI API", False, "Insufficient quota - add payment method")
            elif "invalid_api_key" in error_msg:
                test_result("OpenAI API", False, "Invalid API key")
            else:
                test_result("OpenAI API", False, error_msg[:50])
            all_passed = False
    else:
        test_result("OpenAI API key format", False, "Must start with 'sk-'")
        all_passed = False
    
    # Check Firebase
    header("Firebase Configuration")
    
    firebase_vars = [
        "FIREBASE_PROJECT_ID",
        "FIREBASE_API_KEY",
        "FIREBASE_AUTH_DOMAIN",
        "FIREBASE_STORAGE_BUCKET",
        "FIREBASE_MESSAGING_SENDER_ID",
        "FIREBASE_APP_ID"
    ]
    
    firebase_complete = all(os.getenv(var) for var in firebase_vars)
    test_result("Firebase client config", firebase_complete)
    
    if not firebase_complete:
        missing = [var for var in firebase_vars if not os.getenv(var)]
        print(color_text(f"  ⚠ Missing: {', '.join(missing)}", "yellow"))
        all_passed = False
    
    service_key_file = Path(os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE", "serviceAccountKey.json"))
    service_key_env = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    
    has_service_key = service_key_file.exists() or bool(service_key_env)
    test_result("Firebase service account", has_service_key)
    
    if not has_service_key:
        print(color_text("  ⚠ Download from Firebase Console → Project Settings → Service Accounts", "yellow"))
        all_passed = False
    
    if has_service_key:
        try:
            import firebase_admin
            from firebase_admin import credentials
            
            if not firebase_admin._apps:
                if service_key_env:
                    import json
                    cred = credentials.Certificate(json.loads(service_key_env))
                elif service_key_file.exists():
                    cred = credentials.Certificate(str(service_key_file))
                else:
                    cred = credentials.ApplicationDefault()
                
                firebase_admin.initialize_app(cred)
            
            test_result("Firebase Admin SDK", True, "Initialized successfully")
        except ImportError:
            test_result("Firebase Admin library", False, "Run: pip install firebase-admin")
            all_passed = False
        except Exception as e:
            test_result("Firebase Admin SDK", False, str(e)[:50])
            all_passed = False
    
    # Check Ollama (Optional)
    header("Ollama Configuration (Optional)")
    
    try:
        import requests
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        response = requests.get(ollama_host, timeout=2)
        ollama_running = response.status_code == 200
        test_result("Ollama service", ollama_running, f"Running at {ollama_host}")
        
        if ollama_running:
            model_name = os.getenv("MODEL_NAME", "llama3.2:1b")
            try:
                list_response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                models = list_response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                has_model = any(model_name in name for name in model_names)
                test_result(f"Model '{model_name}'", has_model)
                
                if not has_model:
                    print(color_text(f"  ⚠ Pull with: ollama pull {model_name}", "yellow"))
            except:
                test_result("Model check", False, "Could not list models")
    except:
        test_result("Ollama service", False, "Not running (optional - system works without it)")
    
    # Check Dependencies
    header("Python Dependencies")
    
    # Map package names to their import names (some differ from pip package names)
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "openai": "openai",
        "firebase_admin": "firebase_admin",
        "langchain": "langchain",
        "langchain_openai": "langchain_openai",
        "langchain_community": "langchain_community",
        "faiss": "faiss",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "python-dotenv": "dotenv"
    }
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            test_result(f"Package: {package_name}", True)
        except ImportError:
            test_result(f"Package: {package_name}", False)
            all_passed = False
    
    # Check File Structure
    header("File Structure")
    
    required_files = [
        ("templates/user.html", "User dashboard template"),
        ("templates/admin.html", "Admin dashboard template"),
        ("templates/login.html", "Login page template"),
        ("ai_agent.py", "AI agent module"),
        ("main.py", "FastAPI server"),
        ("requirements.txt", "Dependencies list")
    ]
    
    for file_path, description in required_files:
        exists = Path(file_path).exists()
        test_result(description, exists, file_path)
        if not exists:
            all_passed = False
    
    # Check Directories
    optional_dirs = [
        ("artifacts", "ML models storage"),
        ("vector_store", "FAISS index storage"),
        ("datasheets", "PDF uploads storage")
    ]
    
    print(f"\n{color_text('Optional directories (auto-created):', 'blue')}")
    for dir_path, description in optional_dirs:
        exists = Path(dir_path).exists()
        mark = "📁" if exists else "📂"
        status = "exists" if exists else "will be created"
        print(f"{mark} {description}: {status}")
    
    # Final Summary
    header("Summary")
    
    if all_passed:
        print(color_text("\n✅ All checks passed! Ready to run:", "green"))
        print(color_text("\n  uvicorn main:app --reload --host 0.0.0.0 --port 8000\n", "green"))
    else:
        print(color_text("\n⚠️  Some checks failed. Please fix the issues above.", "yellow"))
        print(color_text("\nRefer to SETUP_GUIDE.md for detailed instructions.\n", "yellow"))
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(color_text("\n\nDiagnostic cancelled by user.", "yellow"))
        sys.exit(1)
    except Exception as e:
        print(color_text(f"\n\n❌ Unexpected error: {e}", "red"))
        sys.exit(1)