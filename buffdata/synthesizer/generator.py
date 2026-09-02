import asyncio
from typing import List
from pydantic import BaseModel, Field
from buffdata.engine.client import LLMClient

class SyntheticQA(BaseModel):
    instruction: str = Field(description="A user instruction or question based on the text.")
    response: str = Field(description="A highly accurate, detailed response based on the text.")

class SyntheticDataset(BaseModel):
    items: List[SyntheticQA] = Field(description="List of extracted QA pairs.")

async def extract_text_from_pdf(pdf_path: str) -> str:
    import fitz # PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

async def extract_text_from_url(url: str) -> str:
    import aiohttp
    from bs4 import BeautifulSoup
    from buffdata.security.validator import is_safe_url
    
    if not is_safe_url(url):
        raise ValueError(f"Security Policy Violation: URL is not safe or allowed: {url}")
        
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={'User-Agent': 'Mozilla/5.0 BuffData/2.0'}) as response:
            response.raise_for_status()
            html = await response.text()
            
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

class DataSynthesizer:
    def __init__(self, client: LLMClient):
        self.client = client
        
    async def synthesize_from_text(self, text: str, max_pairs: int = 10) -> List[dict]:
        prompt = f"""
        Analyze the following text and generate {max_pairs} high-quality instruction-response pairs for fine-tuning an AI.
        The instruction should be a realistic user query, and the response should be derived from the text.
        
        Text:
        {text[:50000]} # Limit for now
        """
        
        result = await self.client.generate_structured_async(prompt, SyntheticDataset)
        if not result:
            return []
            
        return [
            {
                "instruction": qa.instruction,
                "input": "",
                "output": qa.response,
                "metadata": {"source": "synthetic"}
            }
            for qa in result.items
        ]
