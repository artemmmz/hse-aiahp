import os

import pandas as pd
from dotenv import load_dotenv

from app.models.mistral import Mistral
from app.utils.submit import generate_submit

if __name__ == "__main__":
    load_dotenv()

    system_prompt = """
    Ты - профессиональный программист и ментор. Давай очень короткие ответы о синтаксических ошибках в коде, если они есть.
    """

    mistral = Mistral(
        token=os.getenv("MISTRAL_API_KEY"),
        system_prompt=system_prompt,
    )


    def predict(row: pd.Series) -> str:
        return mistral.ask(row["student_solution"])


    generate_submit(
        test_solutions_path="data/raw/test/solutions.xlsx",
        predict_func=predict,
        save_path="data/processed/submission.csv",
        use_tqdm=True,
    )
