from __future__ import annotations

from openai import AsyncOpenAI


async def translate_to_english(
    afrikaans_text: str,
    *,
    context: str,
    cheap: bool = False,
) -> tuple[str, str]:
    """Translate Afrikaans text to idiomatic English via OpenAI chat completion."""
    model = "gpt-4o-mini" if cheap else "gpt-4o"
    system_prompt = (
        "You are a professional translator converting Afrikaans speech transcripts into English.\n"
        f"Domain context: {context}\n"
        "The speaker may mix in English words mid-sentence (code-switching); preserve their "
        "meaning naturally in English without flagging the switch.\n"
        "Produce idiomatic, natural English — never a literal word-for-word translation. "
        "Match the tone of the source: casual stays casual, formal stays formal.\n"
        "Output ONLY the English translation. No preamble, no surrounding quotes, no notes, "
        "no explanations."
    )

    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": afrikaans_text},
        ],
    )
    english = response.choices[0].message.content or ""
    return english, model
