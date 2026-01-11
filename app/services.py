import os
import base64
import json
import tiktoken
from typing import Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .config import MODEL_PRICING


def encode_image(image_path: str) -> str:
    """Encode image to base64 string format for API submission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> Tuple[float, float, float]:
    """Calculate cost based on token usage and model pricing."""
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")

    input_cost = (input_tokens / 1000) * MODEL_PRICING[model]["input"]
    output_cost = (output_tokens / 1000) * MODEL_PRICING[model]["output"]
    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def initialize_agent(
    image_path: str, model: str = "gpt-4o"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Initialize the OCR agent and process the check image."""
    system_prompt = """You are a specialized OCR system designed to extract specific information from bank check images. Your sole purpose is to analyze check images and extract the following details:
    1. Name (payee name)
    2. Date
    3. Amount in words (first line)
    4. Amount in words (second line, if applicable)
    5. Amount in digits
    6. Presence of signature (true/false)
    
    You must provide the extracted information in valid JSON format as follows:
    
    {
      "chequeOcrResponse": {
        "name": [extracted name or null],
        "date": [extracted date or null],
        "amountInWords1": [first line of amount in words or null],
        "amountInWords2": [second line of amount in words or null],
        "amount_digits": [extracted amount in digits or null],
        "signature": [true or false]
      }
    }
    
    IMPORTANT:
    - Do not provide any explanations, greetings, additional text, or code blocks
    - Return ONLY the valid JSON object with no markdown formatting
    - If any information cannot be extracted, return null for that field
    - For signature, return true if a signature is detected, false otherwise
    - Process only check images; for any other input, return the JSON with all null values
    - Do not ask for clarification or engage in conversation
    """

    # Check if image path exists
    if not os.path.exists(image_path):
        return {
            "chequeOcrResponse": {
                "name": None,
                "date": None,
                "amountInWords1": None,
                "amountInWords2": None,
                "amount_digits": None,
                "signature": False,
            }
        }, {
            "model": model,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    # Encode image to base64
    base64_image = encode_image(image_path)

    # Create vision model client
    chat = ChatOpenAI(model=model, temperature=0, max_tokens=1000)

    # Create messages with image content
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": "Extract information from this check image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        ),
    ]

    # Get response from the model
    response = chat.invoke(messages)
    output_text = response.content

    # Extract token usage from the API response
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = response.usage_metadata[
            "input_tokens"
        ]  # Includes text and image tokens
        output_tokens = response.usage_metadata["output_tokens"]
    else:
        # Fallback: estimate tokens (rare case)
        input_text = system_prompt + "Extract information from this check image."
        input_tokens = count_tokens(input_text) + 170  # Rough estimate for image
        output_tokens = count_tokens(output_text)

    # Calculate cost using actual token counts
    input_cost, output_cost, total_cost = calculate_cost(
        model, input_tokens, output_tokens
    )

    cost_info = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
    }

    # Clean up the response content
    cleaned_response = output_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.replace("```json", "", 1)
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response.replace("```", "", 1)
    cleaned_response = cleaned_response.strip()

    # Ensure valid JSON
    if not cleaned_response.startswith("{"):
        cleaned_response = "{" + cleaned_response
    if not cleaned_response.endswith("}"):
        cleaned_response = cleaned_response + "}"

    try:
        parsed_response = json.loads(cleaned_response)
        return parsed_response, cost_info
    except json.JSONDecodeError:
        default_response = {
            "chequeOcrResponse": {
                "name": None,
                "date": None,
                "amountInWords1": None,
                "amountInWords2": None,
                "amount_digits": None,
                "signature": False,
            }
        }
        return default_response, cost_info
