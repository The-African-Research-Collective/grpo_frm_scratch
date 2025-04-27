import pytest

from grpo_tutorial.rewards import structure_reward, accuracy_reward

sample_completions = [
    ([[{"role": "assistant", "content": "<thinking>Here's what i'm thinking</thinking>\n<answer>\n42\n</answer>"}]], 42, 1.0, 1.0),
    ([[{"role": "assistant", "content": "So here's how to solve this</thinking>\n<answer>\nThe answer is 43\n</answer>"}]], 30, 0.75, 0.0),
]

@pytest.mark.parametrize("completions, answer, expected_structure_reward, expected_accuracy_reward", sample_completions)
def test_structure_accuracy_reward(completions, answer, expected_structure_reward, expected_accuracy_reward):
    """
    Test the structure and accuracy rewards.
    """

    print(completions)

    # Calculate the structure reward
    structure_rewards = structure_reward(completions)
    
    # Calculate the accuracy reward
    accuracy_rewards = accuracy_reward(completions, [answer])
    
    # Check if the calculated rewards match the expected rewards
    assert structure_rewards[0] == expected_structure_reward, f"Expected {expected_structure_reward}, but got {structure_rewards[0]}"
    assert accuracy_rewards[0] == expected_accuracy_reward, f"Expected {expected_accuracy_reward}, but got {accuracy_rewards[0]}"