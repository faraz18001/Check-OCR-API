import os

# API key configuration
# In a real-world scenario, you should access this via os.getenv("OPENAI_API_KEY")
# The original code explicitly set it to empty string/placeholder.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""

# Corrected model pricing (cost per 1K tokens)
MODEL_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},  # $5/1M input, $15/1M output
    "gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.0006,
    },  # $0.15/1M input, $0.60/1M output
    "gpt-4-vision-preview": {
        "input": 0.01,
        "output": 0.03,
    },  # Assuming GPT-4 Turbo pricing
    "gpt-4": {"input": 0.03, "output": 0.06},  # Original GPT-4 pricing
    "gpt-3.5-turbo": {
        "input": 0.0005,
        "output": 0.0015,
    },  # $0.50/1M input, $1.50/1M output
}
