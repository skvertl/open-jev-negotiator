"""Voice LLM generator, prompt formatting, and guardrails for NPC investor dialogue."""
from abc import ABC, abstractmethod
import re
from typing import Any
import httpx

from config import Settings, settings
from schemas import ReactionType, VoiceDirective

VOICE_SYSTEM_PROMPT_TEMPLATE = (
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


def format_voice_prompt(directive: VoiceDirective, no_counter_text: str = "нет") -> str:
    """Format system prompt template strictly populating directive data.

    When reaction is 'counter', injects calculated counter investment and equity.
    For non-counter reactions, replaces counter-offer placeholder with no_counter_text.
    """
    if (
        directive.reaction == "counter"
        and directive.counter_offer_investment is not None
        and directive.counter_offer_equity is not None
    ):
        counter_terms = (
            f"{directive.counter_offer_investment}€ за {directive.counter_offer_equity}%"
        )
    else:
        counter_terms = no_counter_text

    prompt = VOICE_SYSTEM_PROMPT_TEMPLATE.replace(
        "{counter_offer_investment}€ за {counter_offer_equity}%",
        counter_terms,
    )
    return prompt.format(
        reaction=directive.reaction,
        accept_score=directive.accept_score,
        good_deal_status=directive.good_deal_status,
    )


def clean_speech_text(speech: str) -> str:
    """Clean LLM output of dialogue tags, author introductions, and outer quotes."""
    cleaned = speech.strip()
    cleaned = re.sub(
        r"^(?:инвестор(?:\s+сказал|\s+ответил)?\s*:|ответ(?:\s+инвестора)?\s*:|голос\s+инвестора\s*:|автор\s*:)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("«") and cleaned.endswith("»")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def contains_hallucinated_agreement(speech: str) -> bool:
    """Check if speech contains affirmative agreement expressions, ignoring negations."""
    lower = speech.lower()

    # Mask out explicit negations of agreement so they don't trigger false positives
    # e.g., "не согласен", "мы не договорились", "не по рукам"
    sanitized = re.sub(
        r"\b(?:не|нет|никогда\s+не|не\s+буду)\s+(?:соглас(?:ен|н[аыое])|договорились|по\s+рукам|принима[юе]т?|в\s+деле)\b",
        " ",
        lower,
    )
    sanitized = re.sub(r"\bникаких\s+по\s+рукам\b", " ", sanitized)
    sanitized = re.sub(r"\bне\s+готов\s+инвестировать\b", " ", sanitized)

    agreement_patterns = [
        r"\b(?:я\s+)?соглас(?:ен|н[аыое])\b",
        r"\bдоговорились\b",
        r"\bпо\s+рукам\b",
        r"\bпринима[юе]т?\s+(?:ваше\s+|эти\s+|ваши\s+)?(?:предложение|условия|сделку)\b",
        r"\bсделка\s+заключена\b",
        r"\bмы\s+в\s+деле\b",
        r"\bготов\s+инвестировать\s+на\s+(?:этих|ваших)\s+условиях\b",
        r"\b(?:i\s+)?accept\b",
        r"\bdeal\b",
        r"\bagreed\b",
    ]

    for pattern in agreement_patterns:
        if re.search(pattern, sanitized):
            return True
    return False


def validate_dialogue_guardrails(speech: str, directive: VoiceDirective) -> str:
    """Validate dialogue against arbiter directive to prevent contradictory commitments.

    If reaction is 'reject', 'walk_away', or 'counter', prevents hallucinated agreements.
    Strips author introductions and quotes.
    """
    cleaned = clean_speech_text(speech)

    if directive.reaction in ("reject", "walk_away", "counter"):
        if contains_hallucinated_agreement(cleaned):
            if directive.reaction == "walk_away":
                return "Переговоры окончены, я выхожу из сделки."
            elif directive.reaction == "reject":
                return "Я отклоняю это предложение. На таких условиях сделки не будет."
            elif directive.reaction == "counter":
                inv = directive.counter_offer_investment
                eq = directive.counter_offer_equity
                if inv is not None and eq is not None:
                    return f"Ваши условия меня не устраивают. Мое встречное предложение: {inv}€ за {eq}%."
                return "Ваши текущие условия неприемлемы, пересмотрите параметры сделки."

    return cleaned


class BaseVoiceLLM(ABC):
    """Abstract base class for Voice LLM dialogue generation."""

    @abstractmethod
    async def generate_speech(
        self, directive: VoiceDirective, history: list[str] | None = None
    ) -> str:
        """Generate formatted spoken response from investor adhering to directive."""
        pass


class MockVoiceLLM(BaseVoiceLLM):
    """Deterministic offline mock voice generator providing Russian venture investor dialogue."""

    MOCK_SPEECHES: dict[ReactionType, str] = {
        "accept": (
            "Отличное предложение, мы ударили по рукам. "
            "Моя команда юристов уже готовит term sheet для закрытия раунда. "
            "Добро пожаловать в портфель!"
        ),
        "counter": (
            "Ваша текущая оценка завышена для этой стадии, но проект мне интересен. "
            "Мое встречное предложение: {investment}€ за {equity}%. "
            "Если согласны — оформляем сделку, иначе переговоры окончены."
        ),
        "reject": (
            "На таких условиях я заходить в раунд не готов, оценка абсолютно не оправдана метриками. "
            "Пересмотрите свое предложение и аппетиты, если хотите продолжить разговор. "
            "Пока это категорический отказ."
        ),
        "walk_away": (
            "Мы ходим по кругу, а мое время стоит слишком дорого. "
            "Не вижу смысла продолжать эти переговоры, я выхожу из сделки. "
            "Всего хорошего и удачи в поисках менее требовательных инвесторов."
        ),
    }

    async def generate_speech(
        self, directive: VoiceDirective, history: list[str] | None = None
    ) -> str:
        """Generate tailored Russian VC dialogue for the given directive."""
        template = self.MOCK_SPEECHES.get(directive.reaction)
        if not template:
            template = self.MOCK_SPEECHES["reject"]

        if directive.reaction == "counter":
            inv = directive.counter_offer_investment or 0
            eq = directive.counter_offer_equity or 0.0
            speech = template.format(investment=inv, equity=eq)
        else:
            speech = template

        return validate_dialogue_guardrails(speech, directive)


def _normalize_chat_endpoint(url: str) -> str:
    """Normalize base LLM URL to OpenAI-compatible /chat/completions endpoint."""
    cleaned = url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


class HttpVoiceLLM(BaseVoiceLLM):
    """Asynchronous HTTP client calling remote LLM endpoint via OpenAI / Ollama / vLLM API."""

    def __init__(
        self,
        llm_url: str = "http://localhost:11434/v1",
        api_key: str = "dummy",
        model: str = "llama3:latest",
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.llm_url = llm_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = client
        self.endpoint = _normalize_chat_endpoint(llm_url)

    def _build_payload(
        self, directive: VoiceDirective, history: list[str] | None = None
    ) -> dict[str, Any]:
        """Build chat completions payload with system prompt and history context."""
        system_content = format_voice_prompt(directive)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        if history:
            history_text = "\n".join(history[-4:])
            user_content = f"История переговоров:\n{history_text}\n\nОзвучь решение инвестора."
        else:
            user_content = "Озвучь решение инвестора в соответствии с директивой."

        messages.append({"role": "user", "content": user_content})

        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200,
        }

    def _extract_speech(self, data: dict[str, Any]) -> str:
        """Extract assistant message content from API response."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict response from LLM, got {type(data).__name__}: {data}")

        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if content:
                        return str(content)
                elif "text" in choice:
                    return str(choice["text"])

        if "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content", "")
            if content:
                return str(content)

        if "response" in data and isinstance(data["response"], str):
            return data["response"]

        raise ValueError(f"Unrecognized chat completion response format: {data}")

    async def generate_speech(
        self, directive: VoiceDirective, history: list[str] | None = None
    ) -> str:
        """Call LLM endpoint, parse speech, and apply safety dialogue guardrails."""
        payload = self._build_payload(directive, history)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self._client is not None:
            response = await self._client.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                )

        response.raise_for_status()
        data = response.json()
        raw_speech = self._extract_speech(data)
        return validate_dialogue_guardrails(raw_speech, directive)


def get_voice_llm(settings_obj: Settings | None = None) -> BaseVoiceLLM:
    """Factory creating configured Voice LLM generator based on application settings."""
    cfg = settings_obj or settings
    if cfg.MOCK_MODE:
        return MockVoiceLLM()
    return HttpVoiceLLM(
        llm_url=cfg.LLM_URL,
        api_key=cfg.LLM_API_KEY,
        model=cfg.LLM_MODEL,
    )


__all__ = [
    "VOICE_SYSTEM_PROMPT_TEMPLATE",
    "format_voice_prompt",
    "clean_speech_text",
    "contains_hallucinated_agreement",
    "validate_dialogue_guardrails",
    "BaseVoiceLLM",
    "MockVoiceLLM",
    "HttpVoiceLLM",
    "get_voice_llm",
]
