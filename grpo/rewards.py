import re
import evaluate

from typing import List


def extract_answer_from_completiom(text: str):
    """
    Extract the text between <answer> and </answer> tags.

    Args:
        text (str): Input string containing <answer> tags

    Returns:
        str: The content between the tags, or None if tags aren't found
    """

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1)
    else:
        return ""


def structure_reward(completions: List[str], answers: List[str] = None):
    """
    Given a set of model completions for a given prompt, this function assigns
    a reward score based on if the model follows the instruction correctly.

    For us, structure here focuses on following instructions specified in the system prompt
    to generate COT in <thinking> tags and the final answer in <answer> tags.

    We assign a reward of 0.25 for each tag that is present in the completion.
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count

    rewards = list(map(count_tags, completions))

    return rewards


def translation_reward(completions: List[str], answers: List[str]):
    """
    Given a set of model completions for a given prompt, this function assigns
    a reward score based on the accuracy of the answer.

    We assign a reward of 1.0 if the answer is correct and 0.0 otherwise.
    """

    def check_answer(completion: str, answer: str) -> float:
        completion_answer = extract_answer_from_completiom(completion)

        if not completion_answer or not completion_answer.strip():
            return 0.0

        chrf_metric = evaluate.load("chrf")
        result = chrf_metric.compute(
            predictions=[completion_answer.strip()], references=[answer.strip()]
        )
        return result["score"] / 100.0

    rewards = list(map(check_answer, completions, answers))

    return rewards
