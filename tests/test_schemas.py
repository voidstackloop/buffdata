import pytest
from buffdata.models.schemas import DatasetItem, DatasetFormat, ChatMessage, PipelineConfig, QualityScore


@pytest.mark.parametrize(
    "provider", ["gemini", "openai", "anthropic", "azure_openai", "bedrock_anthropic", "openai_compatible"],
)
def test_pipeline_config_accepts_every_llm_provider(provider):
    # PipelineConfig.validate_provider checks against buffdata.engine.client.LLMProvider
    # directly rather than its own hardcoded list, specifically so the two can't drift out
    # of sync the way they once did when private-endpoint providers were added to the client
    # but not here.
    config = PipelineConfig(provider=provider)
    assert config.provider == provider


def test_pipeline_config_rejects_unknown_provider():
    with pytest.raises(ValueError, match="provider must be one of"):
        PipelineConfig(provider="not-a-real-provider")


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


def test_huggingface_integer_label_is_preserved():
    item = DatasetItem.from_dict({"text": "A market headline", "label": 2})

    assert item.labels == 2
    assert item.to_dict()["labels"] == 2
