"""
Terminal Chat Client — Personal Digital Assistant

This is the chat interface for the RAG pipeline. It is provided
complete — you do not need to modify this file.

Usage:
    python chat.py
"""

import os
import subprocess

from rag import Assistant

from dotenv import load_dotenv

WELCOME = """
╔══════════════════════════════════════════════════════╗
║           Asistente Personal Digital                 ║
║                                                      ║
║  Pregúúúntame sobre emails, notas, SMS y calendario. ║
║  Tipea '/clear' para borrar el historial.            ║
║  Cálale: ¿En qué dirección festejaremos a Laura?     ║
║  Tipea '/exit' para salir a afuera desde dentro.     ║
╚══════════════════════════════════════════════════════╝
"""


def load_config_from_env() -> dict[str, str | None]:
    """Load raw RAG configuration values from environment variables."""
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "top_k": os.getenv("TOP_K"),
        "chunk_size": os.getenv("CHUNK_SIZE"),
        "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
    }


def main():
    print("Iniciando...")
    config = load_config_from_env()
    assistant = Assistant.from_config(config)
    subprocess.run('cls' if os.name == 'nt' else 'clear')

    print(WELCOME)

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSi te vi... ¡Ni me acuerdo!")
            break

        if not question:
            continue

        if question.lower() == "/exit":
            print("Si te vi... ¡Ni me acuerdo!")
            break

        if question.lower() == "/clear":
            assistant.clear_history()
            subprocess.run('cls' if os.name == 'nt' else 'clear')
            print("\nSe borró todo el chat.\n")
            continue

        response = assistant.ask(question)
        print(f"\nAsistente: {response}\n")


if __name__ == "__main__":
    load_dotenv()
    main()
