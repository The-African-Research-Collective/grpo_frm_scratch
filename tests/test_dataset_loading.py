import pytest

from grpo_tutorial.data_utils import load_and_preprocess_openinstruct


def test_loading_dataset():
    """
    Test the loading and preprocessing of the OpenInstructV2 dataset.
    """

    # Load and preprocess the OpenInstructV2 dataset
    dataset = load_and_preprocess_openinstruct(
        problem_source="augmented_gsm8k",
        num_samples=10,
        split="train_1M"
    )

    print(dataset[0])

    # Check if the dataset is loaded correctly
    assert len(dataset) == 10, "Dataset length mismatch"
    assert "messages" in dataset[0], "Messages field missing in the first example"
    assert len(dataset[0]["messages"]) == 2, "Messages field should have 2 elements"