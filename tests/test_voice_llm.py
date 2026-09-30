"""Unit tests for Voice LLM generator, prompt formatting, and guardrails."""
import re
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from config import Settings
from schemas import VoiceDirective


# -----------------------------------------------------------------------------
# 1. Prompt Formatting & Section 5 Specification Tests
# -----------------------------------------------------------------------------


def test_system_prompt_exact_match_spec_section_5():
    """Verify VOICE_SYSTEM_PROMPT_TEMPLATE matches Section 5 of spec.md exactly."""
    from engine.voice_llm import VOICE_SYSTEM_PROMPT_TEMPLATE

    expected_template = (
        "Ты — жесткий, прагматичный венчурный инвестор. Твоя задача — озвучить решение, которое УЖЕ ПРИНЯТО аналитическим отделом.\n"
        "Ты не имеешь права соглашаться, если решение REJECT или COUNTER.\n"
        "Ты не имеешь права предлагать другие суммы и проценты, кроме указанных в директиве.\n\n"
        "ДИРЕКТИВА:\n"
        "- Действие: {reaction}\n"
        "- Готовность принять сделку: {accept_score}/5\n"
        "- Выгодность предложения: {good_deal_status}\n"
        "- Встречные условия (если action=counter): {counter_offer_investment}€ за {counter_offer_equity}%\n\n"
        "Сгенерируй только прямую речь персонажа на русском языке (до 3 предложений). Не добавляй вводных слов автора."
    )

    assert VOICE_SYSTEM_PROMPT_TEMPLATE == expected_template


def test_format_voice_prompt_counter_reaction():
    """Verify prompt formatting when reaction is counter properly injects terms."""
    from engine.voice_llm import format_voice_prompt

    directive = VoiceDirective(
        reaction="counter",
        accept_score=3,
        good_deal_status="выгодная сделка",
        counter_offer_investment=500_000,
        counter_offer_equity=25.0,
    )

    prompt = format_voice_prompt(directive)

    assert "- Действие: counter" in prompt
    assert "- Готовность принять сделку: 3/5" in prompt
    assert "- Выгодность предложения: выгодная сделка" in prompt
    assert "500000€ за 25.0%" in prompt or "500000€ за 25%" in prompt
    # Ensure no unresolved template placeholders remain
    assert "{" not in prompt and "}" not in prompt


def test_format_voice_prompt_non_counter_reactions():
    """Verify non-counter reactions replace counter terms with '—' or 'нет'."""
    from engine.voice_llm import format_voice_prompt

    for reaction in ("accept", "reject", "walk_away"):
        directive = VoiceDirective(
            reaction=reaction,
            accept_score=4 if reaction == "accept" else 1,
            good_deal_status="выгодная сделка" if reaction == "accept" else "невыгодная сделка",
            counter_offer_investment=None,
            counter_offer_equity=None,
        )

        prompt = format_voice_prompt(directive)

        assert f"- Действие: {reaction}" in prompt
        # Must contain "нет" or "—" as counter-offer terms placeholder
        assert ("нет" in prompt or "—" in prompt)
        # Must not contain unformatted placeholder
        assert "{counter_offer_investment}" not in prompt
        assert "{counter_offer_equity}" not in prompt
        assert "{" not in prompt and "}" not in prompt


# -----------------------------------------------------------------------------
# 2. MockVoiceLLM Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_voice_llm_all_reactions_return_valid_russian_dialogue():
    """MockVoiceLLM returns non-empty Russian dialogue for all reactions."""
    from engine.voice_llm import MockVoiceLLM

    mock_llm = MockVoiceLLM()
    reactions = ("accept", "counter", "reject", "walk_away")

    for reaction in reactions:
        directive = VoiceDirective(
            reaction=reaction,
            accept_score=4 if reaction == "accept" else 2,
            good_deal_status="выгодная сделка" if reaction == "accept" else "невыгодная сделка",
            counter_offer_investment=300_000 if reaction == "counter" else None,
            counter_offer_equity=20.0 if reaction == "counter" else None,
        )

        speech = await mock_llm.generate_speech(directive)

        assert isinstance(speech, str)
        assert len(speech.strip()) > 10
        # Ensure Cyrillic characters are present
        assert bool(re.search(r"[а-яА-ЯёЁ]", speech)), f"Dialogue should be in Russian: {speech}"


@pytest.mark.asyncio
async def test_mock_voice_llm_counter_contains_investment_and_equity():
    """MockVoiceLLM on counter reaction includes counter investment and equity."""
    from engine.voice_llm import MockVoiceLLM

    mock_llm = MockVoiceLLM()
    directive = VoiceDirective(
        reaction="counter",
        accept_score=3,
        good_deal_status="выгодная сделка",
        counter_offer_investment=450_000,
        counter_offer_equity=22.5,
    )

    speech = await mock_llm.generate_speech(directive)

    assert "450000€" in speech or "450 000€" in speech or "450000 €" in speech
    assert "22.5%" in speech or "22.5 %" in speech


@pytest.mark.asyncio
async def test_mock_voice_llm_sentence_count_concise():
    """MockVoiceLLM returns concise dialogue (1-3 sentences)."""
    from engine.voice_llm import MockVoiceLLM

    mock_llm = MockVoiceLLM()
    for reaction in ("accept", "counter", "reject", "walk_away"):
        directive = VoiceDirective(
            reaction=reaction,
            accept_score=3,
            good_deal_status="выгодная сделка",
            counter_offer_investment=500_000 if reaction == "counter" else None,
            counter_offer_equity=25.0 if reaction == "counter" else None,
        )
        speech = await mock_llm.generate_speech(directive)

        # Count sentences by splitting on sentence-ending punctuation followed by whitespace
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", speech) if s.strip()]
        assert 1 <= len(sentences) <= 3, f"Expected 1-3 sentences, got {len(sentences)} in: {speech}"


@pytest.mark.asyncio
async def test_mock_voice_llm_no_author_introductions():
    """MockVoiceLLM does not include author introductions."""
    from engine.voice_llm import MockVoiceLLM

    mock_llm = MockVoiceLLM()
    for reaction in ("accept", "counter", "reject", "walk_away"):
        directive = VoiceDirective(
            reaction=reaction,
            accept_score=3,
            good_deal_status="выгодная сделка",
            counter_offer_investment=500_000 if reaction == "counter" else None,
            counter_offer_equity=25.0 if reaction == "counter" else None,
        )
        speech = await mock_llm.generate_speech(directive)

        assert not speech.startswith("Инвестор сказал")
        assert not speech.startswith("Инвестор:")
        assert not speech.startswith("Автор:")


# -----------------------------------------------------------------------------
# 3. validate_dialogue_guardrails Tests
# -----------------------------------------------------------------------------


def test_guardrail_intercepts_agreement_on_reject():
    """Guardrail intercepts false agreement phrases when directive reaction is reject."""
    from engine.voice_llm import validate_dialogue_guardrails

    directive = VoiceDirective(
        reaction="reject",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )

    hallucinated_speeches = [
        "Отлично, я согласен на ваши условия, оформляем!",
        "Договорились, переводим деньги на ваш расчетный счет.",
        "По рукам, добро пожаловать в клуб!",
        "Я принимаю ваше предложение, отличный стартап.",
        "Мы договорились, подписываем term sheet.",
    ]

    for speech in hallucinated_speeches:
        guarded = validate_dialogue_guardrails(speech, directive)
        assert guarded != speech
        assert "я согласен" not in guarded.lower()
        assert "договорились" not in guarded.lower()
        assert "по рукам" not in guarded.lower()
        # Verify it expresses rejection
        assert any(word in guarded.lower() for word in ("отклон", "отказ", "не готов", "не будет", "нет"))


def test_guardrail_intercepts_agreement_on_walk_away():
    """Guardrail intercepts false agreement phrases when directive reaction is walk_away."""
    from engine.voice_llm import validate_dialogue_guardrails

    directive = VoiceDirective(
        reaction="walk_away",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )

    speech = "Я согласен инвестировать, по рукам!"
    guarded = validate_dialogue_guardrails(speech, directive)

    assert guarded != speech
    assert "я согласен" not in guarded.lower()
    assert "по рукам" not in guarded.lower()
    assert any(word in guarded.lower() for word in ("выхожу", "окончен", "ухожу", "конец"))


def test_guardrail_intercepts_agreement_on_counter():
    """Guardrail intercepts unconditional agreement when directive reaction is counter."""
    from engine.voice_llm import validate_dialogue_guardrails

    directive = VoiceDirective(
        reaction="counter",
        accept_score=3,
        good_deal_status="выгодная сделка",
        counter_offer_investment=400_000,
        counter_offer_equity=20.0,
    )

    speech = "Договорились, я согласен на все ваши условия!"
    guarded = validate_dialogue_guardrails(speech, directive)

    assert guarded != speech
    assert "400000€" in guarded
    assert "20" in guarded


def test_guardrail_preserves_clean_compliant_speech():
    """Guardrail preserves compliant speech without altering it."""
    from engine.voice_llm import validate_dialogue_guardrails

    directive_reject = VoiceDirective(
        reaction="reject",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )
    compliant_reject = "Ваша оценка необоснованно завышена. Я отклоняю это предложение."
    assert validate_dialogue_guardrails(compliant_reject, directive_reject) == compliant_reject

    # Crucial test: "я не согласен" is negation of agreement, must NOT be intercepted
    compliant_negation = "Я не согласен на такие грабительские условия."
    assert validate_dialogue_guardrails(compliant_negation, directive_reject) == compliant_negation

    compliant_negation_2 = "Мы не договорились, возвращайтесь с реальными цифрами."
    assert validate_dialogue_guardrails(compliant_negation_2, directive_reject) == compliant_negation_2

    # Compliant accept
    directive_accept = VoiceDirective(
        reaction="accept",
        accept_score=5,
        good_deal_status="выгодная сделка",
    )
    compliant_accept = "Отличная сделка, я согласен. По рукам!"
    assert validate_dialogue_guardrails(compliant_accept, directive_accept) == compliant_accept


def test_guardrail_strips_author_introductions_and_quotes():
    """Guardrail strips author prefixes like 'Инвестор сказал:' and enclosing quotes."""
    from engine.voice_llm import validate_dialogue_guardrails

    directive = VoiceDirective(
        reaction="reject",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )

    raw_speech = 'Инвестор сказал: "Я отклоняю это предложение."'
    cleaned = validate_dialogue_guardrails(raw_speech, directive)
    assert not cleaned.startswith("Инвестор сказал")
    assert not cleaned.startswith('"')
    assert cleaned == "Я отклоняю это предложение."


# -----------------------------------------------------------------------------
# 4. HttpVoiceLLM Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_voice_llm_openai_chat_completions_parsing():
    """HttpVoiceLLM parses standard OpenAI chat completions payload."""
    from engine.voice_llm import HttpVoiceLLM

    mock_response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Оценка завышена. Мое предложение: 400000€ за 20.0%.",
                }
            }
        ]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data
    mock_resp.raise_for_status = lambda: None
    mock_client.post.return_value = mock_resp

    llm = HttpVoiceLLM(
        llm_url="http://test-llm:11434/v1",
        api_key="sk-test",
        model="llama3:latest",
        client=mock_client,
    )

    directive = VoiceDirective(
        reaction="counter",
        accept_score=3,
        good_deal_status="выгодная сделка",
        counter_offer_investment=400_000,
        counter_offer_equity=20.0,
    )

    speech = await llm.generate_speech(directive, history=["Founder: 500k for 10%"])

    assert "400000€" in speech
    assert "20.0%" in speech

    # Verify HTTP POST call parameters
    mock_client.post.assert_awaited_once()
    called_url, called_kwargs = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
    assert called_url == "http://test-llm:11434/v1/chat/completions"
    assert "Bearer sk-test" in called_kwargs["headers"]["Authorization"]
    messages = called_kwargs["json"]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "ДИРЕКТИВА:" in messages[0]["content"]
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_http_voice_llm_ollama_format_parsing():
    """HttpVoiceLLM parses alternative Ollama chat completion format."""
    from engine.voice_llm import HttpVoiceLLM

    mock_response_data = {
        "message": {
            "role": "assistant",
            "content": "Я отказываюсь от сделки. Оценка не соответствует рынку.",
        }
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data
    mock_resp.raise_for_status = lambda: None
    mock_client.post.return_value = mock_resp

    llm = HttpVoiceLLM(
        llm_url="http://test-llm:11434/v1",
        client=mock_client,
    )

    directive = VoiceDirective(
        reaction="reject",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )

    speech = await llm.generate_speech(directive)
    assert speech == "Я отказываюсь от сделки. Оценка не соответствует рынку."


@pytest.mark.asyncio
async def test_http_voice_llm_applies_guardrails_to_hallucinated_response():
    """HttpVoiceLLM applies safety guardrails if raw response violates directive."""
    from engine.voice_llm import HttpVoiceLLM

    # LLM hallucinates agreement on a reject directive
    mock_response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Отлично, я согласен на все ваши условия, по рукам!",
                }
            }
        ]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data
    mock_resp.raise_for_status = lambda: None
    mock_client.post.return_value = mock_resp

    llm = HttpVoiceLLM(client=mock_client)
    directive = VoiceDirective(
        reaction="reject",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )

    speech = await llm.generate_speech(directive)

    # Must be intercepted by safety guardrail
    assert "я согласен" not in speech.lower()
    assert "по рукам" not in speech.lower()
    assert any(word in speech.lower() for word in ("отклон", "отказ", "не готов", "не будет", "нет"))


@pytest.mark.asyncio
async def test_http_voice_llm_http_500_raises():
    """HttpVoiceLLM propagates httpx.HTTPStatusError on server errors."""
    from engine.voice_llm import HttpVoiceLLM

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    req = httpx.Request("POST", "http://test-llm:11434/v1/chat/completions")
    resp = httpx.Response(status_code=500, request=req)
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "Internal Server Error", request=req, response=resp
    )

    llm = HttpVoiceLLM(client=mock_client)
    directive = VoiceDirective(
        reaction="reject",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )

    with pytest.raises(httpx.HTTPStatusError):
        await llm.generate_speech(directive)


@pytest.mark.asyncio
async def test_http_voice_llm_timeout_raises():
    """HttpVoiceLLM propagates httpx.TimeoutException on connection timeouts."""
    from engine.voice_llm import HttpVoiceLLM

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")

    llm = HttpVoiceLLM(client=mock_client)
    directive = VoiceDirective(
        reaction="reject",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )

    with pytest.raises(httpx.HTTPError):
        await llm.generate_speech(directive)


@pytest.mark.asyncio
async def test_http_voice_llm_malformed_response_raises():
    """HttpVoiceLLM raises ValueError on unrecognized response format."""
    from engine.voice_llm import HttpVoiceLLM

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"unexpected_key": 123}
    mock_resp.raise_for_status = lambda: None
    mock_client.post.return_value = mock_resp

    llm = HttpVoiceLLM(client=mock_client)
    directive = VoiceDirective(
        reaction="reject",
        accept_score=1,
        good_deal_status="невыгодная сделка",
    )

    with pytest.raises(ValueError):
        await llm.generate_speech(directive)


@pytest.mark.asyncio
async def test_http_voice_llm_default_client_lifecycle():
    """HttpVoiceLLM manages internal httpx.AsyncClient when none is injected."""
    from engine.voice_llm import HttpVoiceLLM

    directive = VoiceDirective(
        reaction="accept",
        accept_score=5,
        good_deal_status="выгодная сделка",
    )
    mock_response = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "Мы договорились!"}}]},
        request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        llm = HttpVoiceLLM()
        speech = await llm.generate_speech(directive)
        assert speech == "Мы договорились!"
        mock_post.assert_awaited_once()


def test_http_voice_llm_endpoint_normalization():
    """Verify endpoint normalization handles trailing slashes, /v1, and /chat/completions."""
    from engine.voice_llm import HttpVoiceLLM

    llm1 = HttpVoiceLLM(llm_url="http://localhost:11434")
    assert llm1.endpoint == "http://localhost:11434/v1/chat/completions"

    llm2 = HttpVoiceLLM(llm_url="http://localhost:11434/v1")
    assert llm2.endpoint == "http://localhost:11434/v1/chat/completions"

    llm3 = HttpVoiceLLM(llm_url="http://localhost:11434/v1/")
    assert llm3.endpoint == "http://localhost:11434/v1/chat/completions"

    llm4 = HttpVoiceLLM(llm_url="http://custom-api:8080/v1/chat/completions")
    assert llm4.endpoint == "http://custom-api:8080/v1/chat/completions"


# -----------------------------------------------------------------------------
# 5. Factory Function & Base Voice LLM Tests
# -----------------------------------------------------------------------------


def test_factory_returns_mock_when_mock_mode_true():
    """get_voice_llm returns MockVoiceLLM when settings.MOCK_MODE=True."""
    from engine.voice_llm import BaseVoiceLLM, MockVoiceLLM, get_voice_llm

    cfg = Settings(MOCK_MODE=True)
    llm = get_voice_llm(cfg)
    assert isinstance(llm, MockVoiceLLM)
    assert isinstance(llm, BaseVoiceLLM)


def test_factory_returns_http_when_mock_mode_false():
    """get_voice_llm returns HttpVoiceLLM when settings.MOCK_MODE=False."""
    from engine.voice_llm import BaseVoiceLLM, HttpVoiceLLM, get_voice_llm

    cfg = Settings(
        MOCK_MODE=False,
        LLM_URL="http://remote-llm:8000/v1",
        LLM_API_KEY="secret-key",
        LLM_MODEL="gpt-4o",
    )
    llm = get_voice_llm(cfg)
    assert isinstance(llm, HttpVoiceLLM)
    assert isinstance(llm, BaseVoiceLLM)
    assert llm.llm_url == "http://remote-llm:8000/v1"
    assert llm.api_key == "secret-key"
    assert llm.model == "gpt-4o"


def test_factory_uses_default_settings_when_none_provided():
    """get_voice_llm uses default settings when none provided."""
    from engine.voice_llm import BaseVoiceLLM, get_voice_llm

    llm = get_voice_llm()
    assert isinstance(llm, BaseVoiceLLM)
