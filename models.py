from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

# 1. Individual Line Item Schema [cite: 20-39]
class BillItem(BaseModel):
    item_name: str = Field(..., description="Exactly as mentioned in the bill")
    item_amount: float = Field(..., description="Net Amount of the item post discounts")
    item_rate: float = Field(..., description="Exactly as mentioned in the bill")
    item_quantity: float = Field(..., description="Exactly as mentioned in the bill")

# 2. Page Level Schema [cite: 17-19]
class PageLevelData(BaseModel):
    page_no: str
    page_type: Literal["Bill Detail", "Final Bill", "Pharmacy"]
    bill_items: List[BillItem]

# 3. Token Usage Stats [cite: 15-16]
class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

# 4. Final Data Payload [cite: 16-97]
class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLevelData]
    total_item_count: int

# 5. The API Response Schema 
class ExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractionData] = None
    error: Optional[str] = None # Helper field for debugging

# 6. The Input Request [cite: 9-12]
class ExtractionRequest(BaseModel):
    document: str