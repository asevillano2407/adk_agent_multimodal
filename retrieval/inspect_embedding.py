import math
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from retrieval.embeddings import embed_text

def main() -> None:
    """Comprueba una llamada al modelo de embeddings."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / "corporate_reports/.env")

    with genai.Client(vertexai=False) as client:
        vector = embed_text(
            client,
            "Compute & Networking revenue was 116.2 billion USD.",
            task_type="RETRIEVAL_DOCUMENT",
        )

    print("Dimensiones:", len(vector))
    print("Primeros cinco valores:", vector[:5])
    print("Norma:", math.hypot(*vector))


if __name__ == "__main__":
    main()
