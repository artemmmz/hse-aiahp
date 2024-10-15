import os

import pandas as pd
from dotenv import load_dotenv

from app.models.gigachat import Gigachat
from app.utils.submit import generate_submit

if __name__ == "__main__":
    load_dotenv()

    system_prompt = """
    Ты - профессиональный программист и ментор. Давай очень короткие ответы о синтаксических ошибках в коде, если они есть.
    """

    gigachat = Gigachat(
        token=os.environ["GIGACHAT_API_TOKEN"],
        client_id=os.environ["GIGACHAT_API_CLIENT_ID"],
        system_prompt=system_prompt,
    )


    def predict(row: pd.Series) -> str:
        return gigachat.ask(row["student_solution"])


    generate_submit(
        test_solutions_path="../data/raw/test/solutions.xlsx",
        predict_func=predict,
        save_path="../data/processed/submission.csv",
        use_tqdm=True,
    )
