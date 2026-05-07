from __future__ import annotations

import json

from openai import AsyncOpenAI


async def simplify_and_translate(
    afrikaans_text: str,
    *,
    context: str,
    cheap: bool = False,
) -> tuple[str, str, str]:
    """Simplify Afrikaans to a Grade 3 reading level AND translate to English in one call.

    Returns (simplified_afrikaans, english, model_name).
    """
    model = "gpt-4o-mini" if cheap else "gpt-4o"
    system_prompt = (
        "You receive a raw Afrikaans speech transcript. Produce TWO outputs as a single JSON object.\n"
        "\n"
        "1. \"afrikaans\": Rewrite the SAME content in very simple Afrikaans, easy for a Graad 3 leerder "
        "(about 8 to 9 years old) to understand. Rules:\n"
        "   - Replace every big or uncommon word with an everyday equivalent (e.g. 'bemarkingsbestuurder' -> 'bemarking se baas', 'inkomste' -> 'geld wat inkom').\n"
        "   - Break long sentences into short ones. One idea per sentence.\n"
        "   - Use natural, conversational, kid-friendly Afrikaans. Avoid jargon.\n"
        "   - Preserve every fact and idea from the original. Do NOT summarize. Do NOT skip details.\n"
        "   - If the speaker code-switched into English, render that part in simple Afrikaans where it fits naturally.\n"
        "   - Keep proper nouns (names of people, companies, places) as they are.\n"
        "\n"
        "2. \"english\": An idiomatic English translation of the original meaning. Natural English, not literal. "
        "Match the source tone (casual stays casual).\n"
        "\n"
        f"Domain context for both outputs: {context}\n"
        "\n"
        "Output ONLY a JSON object with EXACTLY these two keys: "
        "{\"afrikaans\": \"...\", \"english\": \"...\"}. "
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
    afrikaans = (data.get("afrikaans") or "").strip()
    english = (data.get("english") or "").strip()
    return afrikaans, english, model
