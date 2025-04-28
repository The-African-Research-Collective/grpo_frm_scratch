def generate_batch_rollout_data(batch_data,
            model,
            reference_model,
            tokenizer,
            num_generations,
            max_generation_length ):
    """
    This function is used to generate number of generations for each prompt in a given batch of data
    We also generate the log probabilities for each token using the current policy and reference policy

    Steps:
    ---------------------------------------------------------------------------------------------------
    1. Generate `num_generations` samples for each prompt in the batch using the model
    2. Concatenate the prompt ids with each generated sample ids
    3. Pass each concatenated prompt_ids + sample_ids to the model to get the log probabilities (explore chunked softmax)
    4. Return input_ids, masks and log_probs and generations

    batch_data : batch_prompts, answers

    """

    