# LLM Provider Notes

## Gemini
Free tier with 1500 RPD on Flash-Lite, 250 RPD on Flash. Quota resets at
Pacific midnight. Best price-performance ratio in 2026 at $0.10/M input
tokens on Flash-Lite.

## Claude
Anthropic API requires paid credit; claude.ai subscription does not include
API access. Strongest tool-calling reliability among major providers.

## Ollama
Local model hosting. Single-instance Ollama serializes requests, so it
doesn't parallelize even when LangGraph dispatches concurrent calls.
Useful for development, weak choice for production parallelism.