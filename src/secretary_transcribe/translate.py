from __future__ import annotations

import json

from openai import AsyncOpenAI


def _model_name(*, cheap: bool) -> str:
    return "gpt-4o-mini" if cheap else "gpt-4o"


async def fix_afrikaans_construction(
    raw_text: str,
    *,
    context: str,
    cheap: bool = False,
) -> tuple[str, str]:
    """Clean up a raw Afrikaans speech transcript into well-formed Grade 3 Afrikaans.

    Fixes sentence construction (run-ons, false starts, repetitions, missing punctuation
    that come from speech-to-text) while keeping the language at a Graad 3 reading level.

    Returns (cleaned_afrikaans, model_name).
    """
    model = _model_name(cheap=cheap)
    system_prompt = (
        "You receive a raw Afrikaans speech-to-text transcript. It may contain run-on "
        "sentences, false starts, repeated words, missing punctuation, and other speech "
        "disfluencies. Rewrite it into clean, well-constructed Afrikaans at a Graad 3 "
        "reading level (about 8 to 9 years old).\n"
        "\n"
        "Rules:\n"
        "   - Fix sentence construction: split run-ons, complete fragments, add proper "
        "punctuation, remove repeated words and 'uhm'-type fillers.\n"
        "   - Keep the language simple. Replace big or uncommon words with everyday "
        "equivalents (e.g. 'bemarkingsbestuurder' -> 'bemarking se baas', 'inkomste' -> "
        "'geld wat inkom').\n"
        "   - Short sentences. One idea per sentence.\n"
        "   - Natural, conversational, kid-friendly Afrikaans. No jargon.\n"
        "   - Preserve EVERY fact and idea from the original. Do NOT summarize. Do NOT "
        "skip details.\n"
        "   - If the speaker code-switched into English, render that part in simple "
        "Afrikaans where it fits naturally.\n"
        "   - Keep proper nouns (names of people, companies, places) as they are.\n"
        "\n"
        f"Domain context: {context}\n"
        "\n"
        "Output ONLY a JSON object with EXACTLY this key: "
        "{\"afrikaans\": \"...\"}. "
        "No preamble, no explanations, no extra keys, no markdown fences."
    )

    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    afrikaans = (data.get("afrikaans") or "").strip()
    return afrikaans, model


async def translate_to_english(
    afrikaans_text: str,
    *,
    context: str,
    cheap: bool = False,
) -> tuple[str, str]:
    """Translate cleaned Afrikaans into idiomatic English.

    Returns (english, model_name).
    """
    model = _model_name(cheap=cheap)
    system_prompt = (
        "You receive an Afrikaans transcript that has already been cleaned up and "
        "simplified. Produce an idiomatic English translation. Natural English, not "
        "literal. Match the source tone (casual stays casual). Preserve every fact "
        "and idea. Keep proper nouns as they are.\n"
        "\n"
        f"Domain context: {context}\n"
        "\n"
        "Output ONLY a JSON object with EXACTLY this key: "
        "{\"english\": \"...\"}. "
        "No preamble, no explanations, no extra keys, no markdown fences."
    )

    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": afrikaans_text},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    english = (data.get("english") or "").strip()
    return english, model
