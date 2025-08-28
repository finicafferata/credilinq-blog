#!/usr/bin/env python3
"""
Quick script to switch CrediLinq Agent to use Gemini AI.
"""

import os
import sys


def main():
    """Switch to Gemini AI provider."""
    print("🔄 Switching CrediLinq Agent to Gemini AI")
    print("=" * 45)
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        print(f"❌ {env_file} not found!")
        print("📝 Creating new .env file from example...")
        
        # Copy from example if it exists
        if os.path.exists(".env.gemini.example"):
            import shutil
            shutil.copy(".env.gemini.example", ".env")
            print("✅ Created .env from .env.gemini.example")
        else:
            # Create basic .env file
            with open(".env", "w") as f:
                f.write("# CrediLinq Agent Configuration\n")
                f.write("PRIMARY_AI_PROVIDER=gemini\n")
                f.write("GEMINI_API_KEY=your-gemini-api-key-here\n")
                f.write("\n")
                f.write("# Database (required)\n")
                f.write("DATABASE_URL=postgresql://postgres@localhost:5432/credilinq_dev_postgres\n")
                f.write("DATABASE_URL_DIRECT=postgresql://postgres@localhost:5432/credilinq_dev_postgres\n")
            print("✅ Created basic .env file")
    
    # Read existing .env
    env_lines = []
    with open(".env", "r") as f:
        env_lines = f.readlines()
    
    # Update or add PRIMARY_AI_PROVIDER
    updated_lines = []
    found_provider = False
    
    for line in env_lines:
        if line.startswith("PRIMARY_AI_PROVIDER="):
            updated_lines.append("PRIMARY_AI_PROVIDER=gemini\n")
            found_provider = True
        else:
            updated_lines.append(line)
    
    # Add if not found
    if not found_provider:
        updated_lines.append("\n# AI Provider Configuration\n")
        updated_lines.append("PRIMARY_AI_PROVIDER=gemini\n")
    
    # Write back to .env
    with open(".env", "w") as f:
        f.writelines(updated_lines)
    
    print("✅ Updated PRIMARY_AI_PROVIDER=gemini")
    
    # Check if Gemini API key is configured
    gemini_key_found = False
    for line in updated_lines:
        if line.startswith("GEMINI_API_KEY=") and "your-" not in line:
            gemini_key_found = True
            break
    
    if not gemini_key_found:
        print("\n⚠️  IMPORTANT: You need to configure your Gemini API key!")
        print("1. Get your API key from: https://makersuite.google.com/app/apikey")
        print("2. Edit .env file and replace: GEMINI_API_KEY=your-gemini-api-key-here")
        print("3. Or set environment variable: export GEMINI_API_KEY='your-key-here'")
        print()
        
        # Interactive API key setup
        if "--interactive" in sys.argv:
            api_key = input("🔑 Enter your Gemini API key (or press Enter to skip): ").strip()
            if api_key and len(api_key) > 10:
                # Update the API key in .env
                final_lines = []
                for line in updated_lines:
                    if line.startswith("GEMINI_API_KEY="):
                        final_lines.append(f"GEMINI_API_KEY={api_key}\n")
                    else:
                        final_lines.append(line)
                
                # Add if not found
                if not any(line.startswith("GEMINI_API_KEY=") for line in updated_lines):
                    final_lines.append(f"GEMINI_API_KEY={api_key}\n")
                
                with open(".env", "w") as f:
                    f.writelines(final_lines)
                
                print("✅ API key configured!")
                gemini_key_found = True
    
    print("\n🧪 Testing Gemini configuration...")
    
    # Test the configuration
    try:
        import subprocess
        result = subprocess.run([sys.executable, "test_gemini_setup.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and "All tests completed successfully" in result.stdout:
            print("✅ Gemini integration test passed!")
            print("\n🎉 Successfully switched to Gemini AI!")
            print("\n📋 Next steps:")
            print("1. Restart your application: python3 -m src.main")
            print("2. Your agents will now use Gemini AI")
            print("3. Monitor performance and costs")
            
            if "--start" in sys.argv:
                print("\n🚀 Starting application...")
                os.system("python3 -m src.main")
            
        else:
            print("❌ Gemini test failed!")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            if not gemini_key_found:
                print("\n💡 Make sure to configure your GEMINI_API_KEY")
                
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out - this might be normal")
        print("✅ Configuration updated. Try running manually: python3 test_gemini_setup.py")
    except FileNotFoundError:
        print("❌ Cannot run test - test_gemini_setup.py not found")
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print(f"\n📄 Configuration saved to: {os.path.abspath('.env')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)