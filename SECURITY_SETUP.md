# Security Setup Guide

## Overview

This guide explains how to securely configure authentication credentials for the CrediLinq Content Agent platform.

## 🔐 Secure Admin Setup

### Quick Setup (Recommended)

Use the provided setup utility to generate secure credentials:

```bash
# Run the secure setup utility
python src/scripts/setup_admin.py
```

This will:
- Generate a cryptographically secure admin password (20 characters)
- Create or update your `.env` file with credentials
- Generate additional security keys if needed
- Validate password strength requirements

### Manual Setup

If you prefer to set credentials manually:

1. **Set environment variables** in your `.env` file:

```env
# Admin Account (Required for first-time setup)
ADMIN_EMAIL=admin@yourcompany.com
ADMIN_PASSWORD=YourSecurePassword123!

# Security Keys (Auto-generated if not provided)
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Optional: Require password change on first login
REQUIRE_ADMIN_PASSWORD_CHANGE=true
```

2. **Password Requirements:**

   **Development:**
   - Minimum 8 characters
   - Must contain: uppercase, lowercase, number, special character
   - Cannot contain common weak patterns

   **Production:**
   - Minimum 12 characters
   - Must contain: uppercase, lowercase, number, special character
   - Cannot contain common weak patterns
   - Additional strength validation

## 🛡️ Security Best Practices

### For Development

1. **Use the setup utility** to generate secure credentials
2. **Never commit** `.env` files to version control
3. **Change default credentials** immediately after setup
4. **Use unique passwords** for each environment

### For Production

1. **Set explicit environment variables** instead of relying on auto-generation:
   ```bash
   export ADMIN_EMAIL="admin@yourcompany.com"
   export ADMIN_PASSWORD="YourVerySecurePassword123!"
   export SECRET_KEY="your-64-character-secret-key"
   export JWT_SECRET="your-64-character-jwt-secret"
   ```

2. **Use secrets management** services:
   - AWS Secrets Manager
   - HashiCorp Vault
   - Azure Key Vault
   - Google Secret Manager

3. **Enable additional security features**:
   ```env
   REQUIRE_ADMIN_PASSWORD_CHANGE=true
   JWT_EXPIRATION_HOURS=8
   RATE_LIMIT_PER_MINUTE=30
   ```

## 🚨 Migration from Hardcoded Credentials

If you're upgrading from a version with hardcoded credentials:

1. **Run the setup utility** to generate new secure credentials
2. **Update any existing API clients** with new authentication
3. **Verify the old hardcoded credentials are removed** from the codebase
4. **Test login functionality** with new credentials

## 🔍 Security Validation

The system automatically validates:

- ✅ **Password strength** (length, character variety, weak patterns)
- ✅ **Secret key security** (auto-generation, minimum length)
- ✅ **Production readiness** (explicit credentials, warnings for auto-generation)
- ✅ **Environment-specific requirements** (stricter rules for production)

## 🆘 Troubleshooting

### "Authentication failed" errors
- Verify `.env` file exists and contains `ADMIN_EMAIL` and `ADMIN_PASSWORD`
- Check password meets requirements (run setup utility to validate)
- Ensure application restarted after credential changes

### "Invalid or expired token" errors
- Check `JWT_SECRET` is set consistently
- Verify `JWT_EXPIRATION_HOURS` setting
- Clear browser cookies/localStorage and re-login

### Auto-generated credentials not showing
- Check console output during application startup
- Verify `ENVIRONMENT=development` (credentials only shown in dev mode)
- Look for log files if credentials were auto-generated

## 📝 Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ADMIN_EMAIL` | No | `admin@credilinq.com` | Admin user email |
| `ADMIN_PASSWORD` | No | Auto-generated | Admin user password |
| `SECRET_KEY` | No | Auto-generated | Application secret key |
| `JWT_SECRET` | No | Auto-generated | JWT signing secret |
| `JWT_EXPIRATION_HOURS` | No | `24` | JWT token lifetime |
| `REQUIRE_ADMIN_PASSWORD_CHANGE` | No | `true` | Force password change on first login |

## 🔄 Regular Security Maintenance

1. **Rotate credentials** periodically (every 90 days recommended)
2. **Monitor failed login attempts** in application logs
3. **Update password policies** as needed
4. **Review and revoke unused API keys** regularly

---

For additional security questions, please refer to the main documentation or create an issue in the repository.