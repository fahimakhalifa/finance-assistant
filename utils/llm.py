import os

import requests


def call_groq_llm(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "AI assistant is not configured. Please set the GROQ_API_KEY environment variable."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as err:
        return f"HTTP Error: {err.response.status_code} — {err.response.text}"
    except Exception as exc:
        return f"LLM call failed: {exc}"
