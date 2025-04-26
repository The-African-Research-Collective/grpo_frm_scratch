import re

from datasets import load_dataset

from grpo_tutorial.data.prompt import SYSTEM_PROMPT

def get_openmath_instruct_answer(response_text: str):

    """
    This is an utility function for extracting the final answer from the OpenMath instruction response text.
    It uses regular expressions to find the answer in the text.

    A sample response from the OpenMath Dataset (https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
    ```
    ...Since the trip is 3 days long, each person will need 3 granola bars/day * 3 days = 9 granola bars.
    So for 5 people, Ava will need 5 * 9 = 45 granola bars.
    Thus, Ava will need to pack \boxed{45} granola bars in total for the entire trip.
    ```
    """

    # The regex pattern looks for \\boxed{ followed by any characters (non-greedy) until }
    pattern = r"\\boxed\{([^}]*)\}"

    # Search for the pattern in the response text
    match = re.search(pattern, response_text)
    if match:
        # If a match is found, extract the answer from the first capturing group
        answer = match.group(1)
        return answer
    else:
        # If no match is found, return None or an appropriate message
        return None

def process_datapoint(example):
    """
    Format each datapoint in the instruction format.
    """

    prompt_messages = [
        {"role": "system","content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
    ]

    return {
        "messages": prompt_messages,
    }



def load_and_preprocess_openinstruct(problem_source: str, num_samples: int, split: str = "train"):

    """
    This is an utility function for loading and preprocessing the OpenInstructV2 dataset.

    OpenMath Dataset (https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)

    We want to load the dataset, filter and subset it, and extract the required fields and format in the instruction format.
    """

    # load the train split of the dataset
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split=split)

    # Filter the dataset to only include the specified problem source
    filtered_dataset = dataset.filter(lambda x: x["problem_source"] == problem_source)

    # Subset the dataset to the specified number of samples
    subset_dataset = filtered_dataset.select(range(num_samples))

    dataset = subset_dataset.map(
        process_datapoint,
        remove_columns=[
                name
                for name in subset_dataset.column_names
                if name not in ["messages", "answer"]
            ],
        desc="Processing individual datapoints",
    )

    return dataset



