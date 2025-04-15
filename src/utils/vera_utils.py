from prompts.vera_prompts import get_vera_prompt, VERA_ASK_FOR_APPROVAL_ONLY_PROMPT, VERA_ANSWER_SYMBOL
from termcolor import colored
from utils.loading_utils import load_domain_specific_verifiers
from vllm import LLM
def generate_verifier_approvals(
    config: dict,
    batch_data: list,
    verifier_llm: LLM,
    verifier_sampling_params: dict,
    strict_verifier_sampling_params: dict
) -> list:
    """
    Generate the verifier approvals for a batch of answers and update the batch data.

    Args:
        config: Configuration dictionary
        batch_data: A list of data dictionaries with generated solutions
        verifier_llm: The model to use for generation
        verifier_sampling_params: The sampling parameters for the verifier model
        strict_verifier_sampling_params: The sampling parameters for the strict verifier model
    Returns:
        list: Updated batch data with verifier approvals
    """
    user_prompts = []
    veras = load_domain_specific_verifiers(config['dataset'])
    num_veras = len(veras)
    direct_veras_idx = [i for i, v in enumerate(veras) if "direct" in v]

    for data in batch_data:
        generated_solutions = data["generated_solutions"]

        for solution in generated_solutions:
            for vera_name in veras:
                vera_prompt = get_vera_prompt(config['dataset'], vera_name, data['problem'], solution)
                user_prompts.append(vera_prompt)

    assert len(user_prompts) == num_veras * config['num_generations'] * len(batch_data), colored(f"Number of user prompts is not equal to the number of veras * number of generated solutions * batch size. Double check!", "red") 
    inputs = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            },
        ] for prompt in user_prompts
    ]
    print(f"\nGenerating {num_veras} different verifier approvals for each of the {config['num_generations']} generated solutions in the batch...\n")
    outputs = verifier_llm.chat(inputs, verifier_sampling_params, use_tqdm=False)
    verifier_responses = [(output.outputs[0].text).split("<end_of_turn>")[0] for output in outputs]
    # SECOND PASS TO GET STRICT TRUE/FALSE APPROVAL
    strict_prompts = [
        [
            inputs[i][0],
            {
                "role": "assistant",
                "content": [{"type": "text", "text": verifier_responses[i]}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": VERA_ASK_FOR_APPROVAL_ONLY_PROMPT}]
            }
        ]
        for i in range(0, len(verifier_responses)) if i % num_veras not in direct_veras_idx
    ]
    print(f"\nGetting strict True/False approvals for verifier responses in the batch...\n")
    strict_outputs = verifier_llm.chat(strict_prompts, strict_verifier_sampling_params, use_tqdm=False)
    strict_decoded_outputs = [(output.outputs[0].text).split("<end_of_turn>")[0] for output in strict_outputs]
    final_outputs = []
    strict_idx = 0
    for idx, output in enumerate(verifier_responses):
        if (idx % num_veras) in direct_veras_idx:
            final_outputs.append(output)
        else:
            final_outputs.append(strict_decoded_outputs[strict_idx])
            strict_idx += 1

    assert len(final_outputs) == num_veras * config['num_generations'] * len(batch_data), colored(f"Number of final outputs is not equal to the number of veras * number of generated solutions * batch size. Double check!", "red") 

    # Update each data entry in the batch with verifier approvals
    for data_idx, data in enumerate(batch_data):
        start_idx = data_idx * num_veras * config['num_generations']
        end_idx = start_idx + num_veras * config['num_generations']
        verifier_final_solutions = [final_outputs[j:j + num_veras] for j in range(start_idx, end_idx, num_veras)]
        verifier_approval_bools = [[extract_verifier_approval(output) for output in sublist] for sublist in verifier_final_solutions]
        data['verifier_responses'] = [verifier_responses[i:i + num_veras] for i in range(start_idx, end_idx, num_veras)]
        data['verifier_final_solutions'] = verifier_final_solutions
        data['verifier_approval_bools'] = verifier_approval_bools
    return batch_data 
    
def extract_verifier_approval(verifier_response: str) -> bool:
    """Extract the verifier's approval from the response."""
    # Get the last answer
    vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
    last_index = verifier_response.lower().rfind(vera_answer_symbol)
    answer = verifier_response[last_index + len(vera_answer_symbol):].strip() if last_index != -1 else None
    
    if not answer:
        print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                      f"and full verifier_response (length {len(verifier_response)}): "
                      f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
        return False
    
    answer = answer.replace("*", "")  # Remove any asterisks (bolding)
    answer = answer.strip().lower()
    if answer == "true" or answer == "true.":
        return True
    elif answer == "false" or answer == "false.":
        return False
    else:
        # Check if 'true' or 'false' is in the first word
        print(colored(f"NOTICE in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                      f"checking if the FIRST WORK contains 'true' or 'false'...", "magenta"))
        first_word = answer.split()[0]
        if "true" in first_word:
            print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
            return True
        elif "false" in first_word:
            print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
            return False
        else:
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                          f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                          f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            return False