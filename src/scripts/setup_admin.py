#!/usr/bin/env python3
"""
Secure admin setup utility for CrediLinq Content Agent.
This script helps initialize admin credentials securely.
"""

import os
import sys
import secrets
import string
import getpass
import hashlib
from pathlib import Path

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_secure_password(length: int = 16) -> str:
    """Generate a cryptographically secure password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    
    # Ensure password has required character types
    if not any(c.isupper() for c in password):
        password = password[:-1] + secrets.choice(string.ascii_uppercase)
    if not any(c.islower() for c in password):
        password = password[:-2] + secrets.choice(string.ascii_lowercase) + password[-1]
    if not any(c.isdigit() for c in password):
        password = password[:-3] + secrets.choice(string.digits) + password[-2:]
    if not any(c in "!@#$%^&*" for c in password):
        password = password[:-4] + secrets.choice("!@#$%^&*") + password[-3:]
    
    return password

def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """Validate password meets security requirements."""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    # Check for character variety
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    missing_types = []
    if not has_upper:
        missing_types.append("uppercase letter")
    if not has_lower:
        missing_types.append("lowercase letter")
    if not has_digit:
        missing_types.append("number")
    if not has_special:
        missing_types.append("special character")
    
    if missing_types:
        errors.append(f"Password must contain at least one: {', '.join(missing_types)}")
    
    # Check for common weak passwords
    weak_patterns = [
        "password", "admin", "123456", "qwerty", "letmein",
        "welcome", "monkey", "dragon", "master", "secret", "admin123"
    ]
    
    password_lower = password.lower()
    for pattern in weak_patterns:
        if pattern in password_lower:
            errors.append(f"Password cannot contain common pattern: {pattern}")
    
    return len(errors) == 0, errors

def write_env_file(admin_email: str, admin_password: str, env_file: str = ".env"):
    """Write or update .env file with admin credentials."""
    env_path = Path(env_file)
    
    # Read existing .env content if it exists
    existing_content = []
    admin_email_set = False
    admin_password_set = False
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('ADMIN_EMAIL='):
                    existing_content.append(f'ADMIN_EMAIL={admin_email}')
                    admin_email_set = True
                elif line.startswith('ADMIN_PASSWORD='):
                    existing_content.append(f'ADMIN_PASSWORD={admin_password}')
                    admin_password_set = True
                else:
                    existing_content.append(line)
    
    # Add new entries if they weren't in the existing file
    if not admin_email_set:
        existing_content.append(f'ADMIN_EMAIL={admin_email}')
    if not admin_password_set:
        existing_content.append(f'ADMIN_PASSWORD={admin_password}')
    
    # Write updated content
    with open(env_path, 'w') as f:
        for line in existing_content:
            f.write(line + '\n')
    
    print(f"✅ Admin credentials written to {env_file}")

def main():
    """Main setup function."""
    print("🔐 CrediLinq Admin Setup Utility")
    print("=" * 40)
    
    # Get admin email
    default_email = "admin@credilinq.com"
    admin_email = input(f"Admin email [{default_email}]: ").strip()
    if not admin_email:
        admin_email = default_email
    
    # Validate email format
    if '@' not in admin_email or '.' not in admin_email.split('@')[1]:
        print("❌ Invalid email format")
        return 1
    
    # Get password preference
    print("\nPassword options:")
    print("1. Generate secure password automatically (recommended)")
    print("2. Set custom password")
    
    choice = input("Choose option [1]: ").strip()
    if not choice:
        choice = "1"
    
    if choice == "1":
        # Generate secure password
        admin_password = generate_secure_password(20)
        print(f"\n🔑 Generated secure password: {admin_password}")
        print("⚠️  Please save this password securely - it won't be shown again!")
        
    elif choice == "2":
        # Custom password
        while True:
            admin_password = getpass.getpass("Enter admin password: ")
            confirm_password = getpass.getpass("Confirm admin password: ")
            
            if admin_password != confirm_password:
                print("❌ Passwords don't match. Please try again.")
                continue
            
            # Validate password strength
            is_valid, errors = validate_password_strength(admin_password)
            if not is_valid:
                print("❌ Password validation failed:")
                for error in errors:
                    print(f"   - {error}")
                continue
            
            break
    else:
        print("❌ Invalid option")
        return 1
    
    # Confirm setup
    print(f"\nSetup Summary:")
    print(f"  Email: {admin_email}")
    print(f"  Password: {'*' * len(admin_password)} ({len(admin_password)} characters)")
    
    confirm = input("\nProceed with setup? [y/N]: ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Setup cancelled")
        return 1
    
    # Write to .env file
    try:
        write_env_file(admin_email, admin_password)
        print("\n✅ Admin setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the application: python -m src.main")
        print("2. Login with the admin credentials")
        print("3. Consider changing the password after first login if required")
        
        # Generate additional security keys if needed
        if not os.getenv('SECRET_KEY'):
            secret_key = secrets.token_hex(32)
            with open('.env', 'a') as f:
                f.write(f'\nSECRET_KEY={secret_key}\n')
            print("🔐 Generated SECRET_KEY")
        
        if not os.getenv('JWT_SECRET'):
            jwt_secret = secrets.token_hex(32)
            with open('.env', 'a') as f:
                f.write(f'JWT_SECRET={jwt_secret}\n')
            print("🔐 Generated JWT_SECRET")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return 1

if __name__ == "__main__":
    exit(main())