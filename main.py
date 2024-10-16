import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from app.models.mistral import Mistral
from app.utils.submit import generate_submit

if __name__ == "__main__":
    load_dotenv()
    ROOT_DIR = Path(__file__).resolve().parent

    system_prompt = """
    Ты — ментор, помогающий студентам, изучающим программирование на Python, справляться с логическими ошибками в их коде. Твоя задача — анализировать предоставленный код и давать полезные рекомендации по его улучшению.
    Не пиши код за студента и не предоставляй прямых решений.
    Направляй студента, задавая вопросы и предлагая пересмотреть отдельные участки кода, где могут быть логические ошибки.
    Указывай на типичные ошибки, которые могут возникнуть при работе с алгоритмами и структурами данных, и помогай студенту увидеть их самостоятельно.
    Придерживайся строгих ограничений: помогай выявить ошибки и объясняй, как проверить правильность, но не предоставляй готовый код.
    Студенты могут не иметь доступа к закрытым тестам, поэтому обрати внимание на проверку граничных случаев и крайних значений входных данных.
    Если требуется, напомни о принципах отладки и тестирования кода, предложи проверить работу программы на дополнительных наборах данных.
    """
    
    mistral = Mistral(
        token=os.getenv("MISTRAL_API_KEY"),
        system_prompt=system_prompt,
    )


    def predict(row: pd.Series) -> str:
        return mistral.ask(row["student_solution"])


    generate_submit(
        test_solutions_path=(ROOT_DIR / "data/raw/test/solutions.xlsx").as_posix(),
        predict_func=predict,
        save_path=(ROOT_DIR / "data/processed/submission.csv").as_posix(),
        use_tqdm=True,
    )
