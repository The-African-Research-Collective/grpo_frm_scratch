import re

def extract_answer_from_completiom(text):
    """
    Extract the text between <answer> and </answer> tags.
    
    Args:
        text (str): Input string containing <answer> tags
        
    Returns:
        str: The content between the tags, or None if tags aren't found
    """
    import re
    
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1)
    else:
        return ""

def structure_reward(completions):
    """
    Given a set of model completions for a given prompt, this function assigns
    a reward score based on if the model follows the instruction correctly.

    For us, structure here focuses on following instructions specified in the system prompt
    to generate COT in <thinking> tags and the final answer in <answer> tags.

    We assign a reward of 0.25 for each tag that is present in the completion.
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<thinking>") == 1:
            count += 0.25
        if text.count("</thinking>") == 1:
            count += 0.25
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count
    
    completions = [completion[0]["content"] for completion in completions]
    rewards = list(map(count_tags, completions))

    return rewards

def accuracy_reward(completions, answers):
    """
    Given a set of model completions for a given prompt, this function assigns
    a reward score based on the accuracy of the answer.

    We assign a reward of 1.0 if the answer is correct and 0.0 otherwise.
    """

    def check_answer(completion: str, answer: str) -> float:
        completion_answer = extract_answer_from_completiom(completion)

        if str(answer) in completion_answer:
            return 1.0
        else:
            return 0.0

    completions = [completion[0]['content'] for completion in completions]
    rewards = list(map(check_answer, completions, answers))

    return rewards

def reward_function(completions, answers):
    """
    Given a set of model completions for a given prompt, this function assigns
    a reward score based on the accuracy of the answer and the structure of the response.
    """

    # Calculate structure rewards
    structure_rewards = structure_reward(completions)

    # Calculate accuracy rewards
    accuracy_rewards = accuracy_reward(completions, answers)

    # Combine the rewards
    combined_rewards = [structure + accuracy for structure, accuracy in zip(structure_rewards, accuracy_rewards)]

    return combined_rewards