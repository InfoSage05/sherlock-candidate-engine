# Model Comparison

## Current Default: Llama 3.1 70B

**Model ID:** `meta-llama/llama-3.1-70b-instruct:free`

### Why 70B over 8B?
- **9x more parameters** = significantly better reasoning and instruction following
- **Better at structured output** - more reliable JSON/tool calling
- **Better at nuanced role classification** - understands context better
- **Still free** on OpenRouter

### Tradeoffs
- Slightly slower than 8B (but still fast enough for 30s batching)
- Same rate limits as 8B on free tier

---

## Alternative Models

### Qwen 2.5 72B
**Model ID:** `qwen/qwen-2.5-72b-instruct:free`

**Best for:** Structured output and JSON generation

**Strengths:**
- Excellent at following complex schemas
- Very reliable tool calling
- Strong multilingual support

**When to use:** If you're getting JSON parsing errors with Llama

---

### NVIDIA Nemotron 70B
**Model ID:** `nvidia/llama-3.1-nemotron-70b-instruct:free`

**Best for:** Reasoning-heavy tasks

**Strengths:**
- NVIDIA-optimized architecture
- Good at multi-step reasoning
- Strong at understanding context

**When to use:** If you need better reasoning about complex interview dynamics

---

### Llama 3.1 8B (Original)
**Model ID:** `meta-llama/llama-3.1-8b-instruct:free`

**Best for:** Speed and low latency

**Strengths:**
- Fastest inference
- Lowest resource usage
- Good enough for simple classification

**When to use:** If you're hitting rate limits or need maximum speed

---

## Switching Models

```python
from sherlock.llm_client import OpenRouterClient, LLMConfig

# Option 1: Qwen for better structured output
config = LLMConfig(model="qwen/qwen-2.5-72b-instruct:free")
client = OpenRouterClient(config)

# Option 2: Nemotron for better reasoning
config = LLMConfig(model="nvidia/llama-3.1-nemotron-70b-instruct:free")
client = OpenRouterClient(config)

# Option 3: 8B for maximum speed
config = LLMConfig(model="meta-llama/llama-3.1-8b-instruct:free")
client = OpenRouterClient(config)
```

---

## Rate Limits (Free Tier)

All free models on OpenRouter have the same limits:
- ~20 requests/minute
- ~10,000 tokens/minute

The 30s batching window keeps you well within these limits for typical interviews.

---

## Recommendation

**Use Llama 3.1 70B** (current default) unless:
- You're hitting rate limits → switch to 8B
- You're getting JSON errors → switch to Qwen 2.5 72B
- You need deeper reasoning → switch to Nemotron 70B
