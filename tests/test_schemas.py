import pytest
from buffdata.models.schemas import DatasetItem, DatasetFormat, ChatMessage, QualityScore

def test_alpaca_schema():
    item = DatasetItem(
        format=DatasetFormat.ALPACA,
        instruction="Write a function",
        input="arg1",
        output="def fn(): pass"
    )
    p, r = item.get_prompt_and_response()
    assert "Write a function" in p
    assert "arg1" in p
    assert r == "def fn(): pass"

    item.update_content(new_prompt="Updated prompt", new_response="Updated response")
    assert item.instruction == "Updated prompt"
    assert item.output == "Updated response"

def test_chat_schema():
    item = DatasetItem(
        format=DatasetFormat.CHAT,
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!")
        ]
    )
    p, r = item.get_prompt_and_response()
    assert p == "Hello"
    assert r == "Hi there!"

    item.update_content(new_response="New answer")
    _, new_r = item.get_prompt_and_response()
    assert new_r == "New answer"

def test_dpo_schema():
    item = DatasetItem(
        format=DatasetFormat.DPO,
        prompt="Tell a joke",
        chosen="Good joke",
        rejected="Bad joke"
    )
    p, r = item.get_prompt_and_response()
    assert p == "Tell a joke"
    assert r == "Good joke"
    d = item.to_dict()
    assert d["chosen"] == "Good joke"
    assert d["rejected"] == "Bad joke"
