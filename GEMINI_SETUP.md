# 🚀 Gemini AI Integration Guide

CrediLinq Agent now supports **Google Gemini AI** as an alternative to OpenAI! This guide will help you set up and migrate to Gemini.

## 🎯 Quick Setup

### 1. Get Your Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key for configuration

### 2. Configure Environment Variables

**Option A: Use the example file**
```bash
cp .env.gemini.example .env
# Edit .env with your actual API key
```

**Option B: Set variables manually**
```bash
export PRIMARY_AI_PROVIDER=gemini
export GEMINI_API_KEY=your-gemini-api-key-here
```

### 3. Test Your Setup
```bash
python test_gemini_setup.py
```

For interactive testing:
```bash
python test_gemini_setup.py --interactive
```

---

## 🔧 Configuration Options

### Primary Provider Selection
```bash
# Use Gemini as primary AI provider
PRIMARY_AI_PROVIDER=gemini

# Or use OpenAI (default)
PRIMARY_AI_PROVIDER=openai
```

### Gemini-Specific Settings
```bash
# Required: Your Gemini API key
GEMINI_API_KEY=your-api-key-here

# Optional: Model selection
GEMINI_MODEL=gemini-1.5-flash        # Default, fast and efficient
# GEMINI_MODEL=gemini-1.5-pro        # More capable, slower

# Optional: Generation parameters
GEMINI_TEMPERATURE=0.7              # Creativity level (0.0-2.0)
GEMINI_MAX_TOKENS=4000              # Response length limit
```

### Backup Provider (Recommended)
Keep OpenAI as a backup:
```bash
OPENAI_API_KEY=your-openai-key-here
```

---

## 🎮 Available Models

### Gemini Models
| Model | Description | Best For |
|-------|-------------|----------|
| `gemini-1.5-flash` | ⚡ **Default** - Fast, efficient | General content, social posts |
| `gemini-1.5-pro` | 🧠 More capable, detailed | Complex content, analysis |

### Model Selection
```bash
# In your .env file
GEMINI_MODEL=gemini-1.5-flash

# Or programmatically
PRIMARY_AI_PROVIDER=gemini
```

---

## 🔄 Migration from OpenAI

### Automatic Migration
The system is designed for **zero-code migration**:

1. **Update Environment Variables**
   ```bash
   PRIMARY_AI_PROVIDER=gemini
   GEMINI_API_KEY=your-key-here
   ```

2. **Restart Application**
   ```bash
   python3 -m src.main
   ```

3. **All agents automatically use Gemini** ✅

### Manual Per-Agent Control
You can also specify providers per-agent:

```python
# In your code
from src.core.ai_utils import get_langchain_llm

# Force specific provider
gemini_llm = get_langchain_llm("gemini")
openai_llm = get_langchain_llm("openai")
```

---

## 📊 Performance Comparison

| Feature | OpenAI GPT-3.5 | Gemini 1.5 Flash | Gemini 1.5 Pro |
|---------|----------------|-------------------|-----------------|
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Cost** | $$$ | $ | $$ |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Context** | 4K tokens | 1M tokens | 1M tokens |

**Recommendation**: Start with `gemini-1.5-flash` for best performance/cost ratio.

---

## 🧪 Testing Your Setup

### Quick Test
```bash
python test_gemini_setup.py
```

Expected output:
```
🧪 Testing Gemini AI Integration
==================================================
Primary AI Provider: gemini
Gemini API Key configured: ✅
OpenAI API Key configured: ✅

📋 Provider Availability Check:
  GEMINI: ✅ Available
  OPENAI: ✅ Available

✅ Successfully created GEMINI client
🚀 Testing Text Generation:
✅ Text generation successful!
🎉 Gemini Integration Test Complete!
```

### Interactive Testing
```bash
python test_gemini_setup.py --interactive
```

Try prompts like:
- "Write a blog post title about AI marketing"
- "Create 5 LinkedIn post ideas"
- "Explain SEO in simple terms"

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. API Key Not Working
```
❌ Failed to create AI client: Gemini client initialization failed
```
**Solution**: 
- Verify your API key at [Google AI Studio](https://makersuite.google.com/app/apikey)
- Ensure key has proper permissions
- Check for typos in environment variable

#### 2. Missing Dependencies
```
❌ Google Generative AI library not available
```
**Solution**:
```bash
pip install google-generativeai
# OR
pip install langchain-google-genai
```

#### 3. Rate Limits
```
❌ Quota exceeded
```
**Solution**:
- Check your usage at [Google Cloud Console](https://console.cloud.google.com/)
- Upgrade your plan or wait for quota reset
- Use OpenAI as backup temporarily

#### 4. Model Not Found
```
❌ Model not found: gemini-1.5-flash
```
**Solution**:
- Update to latest google-generativeai library
- Try `gemini-pro` as fallback model

### Debug Mode
Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python3 -m src.main
```

---

## 🔄 Switching Between Providers

### Runtime Switching
Change providers without restarting:
```python
from src.core.ai_client_factory import AIClientFactory

# Clear cache and switch
AIClientFactory.clear_cache()

# Update settings
import os
os.environ['PRIMARY_AI_PROVIDER'] = 'gemini'
```

### Per-Request Provider
```python
from src.core.ai_utils import call_ai_model

# Use Gemini for this request
response = await call_ai_model("Write a title", provider="gemini")

# Use OpenAI for this request  
response = await call_ai_model("Write a title", provider="openai")
```

---

## 📈 Cost Optimization

### Smart Provider Selection
Use different providers for different tasks:

```python
# Fast tasks: Use Gemini Flash (cheaper)
social_posts = await call_ai_model(prompt, provider="gemini")

# Complex tasks: Use Gemini Pro or OpenAI
analysis = await call_ai_model(complex_prompt, provider="openai")
```

### Batch Processing
Process multiple requests efficiently:
```python
tasks = [
    call_ai_model(prompt1, provider="gemini"),
    call_ai_model(prompt2, provider="gemini"),
    call_ai_model(prompt3, provider="gemini")
]
results = await asyncio.gather(*tasks)
```

---

## 🎯 Next Steps

1. **✅ Test your setup** with the test script
2. **🔧 Configure your environment** variables
3. **🚀 Restart your application**
4. **📊 Monitor performance** and costs
5. **🔄 Optimize** provider usage based on your needs

---

## 📞 Support

- **Issues**: Create an issue on GitHub
- **Questions**: Check the troubleshooting section above
- **Documentation**: Refer to [Gemini API docs](https://ai.google.dev/docs)

Happy content creation with Gemini! 🎉