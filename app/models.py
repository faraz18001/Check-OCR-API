from typing import Optional
from pydantic import BaseModel


class Base64ImageRequest(BaseModel):
    image: str  # Base64 encoded image string
    model: str = "gpt-4o-mini"  # Default model


class ChequeOcrResponse(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    amountInWords1: Optional[str] = None
    amountInWords2: Optional[str] = None
    amount_digits: Optional[str] = None
    signature: bool = False


class CostInfo(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


class OcrResultResponse(BaseModel):
    chequeOcrResponse: ChequeOcrResponse
    cost_info: CostInfo
