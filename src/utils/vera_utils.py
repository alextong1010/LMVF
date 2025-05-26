from prompts.vera_prompts import get_vera_prompt, VERA_ASK_FOR_APPROVAL_ONLY_PROMPT, VERA_ANSWER_SYMBOL, get_borda_prompt, VERA_RANKING_SYMBOL
from termcolor import colored
from utils.loading_utils import load_domain_specific_verifiers
from vllm import LLM
from typing import List, Dict, Any, Optional
import re

_RANK_RE = re.compile(r"[\[\(]?\s*(\d+(?:\s*,\s*\d+)*)\s*[\]\)]?")

def generate_borda_responses(
    config: dict,
    verifier_model_config: dict,
    batch_data: List[Dict[str, Any]],
    verifier_llm: LLM,
    verifier_sampling_params: dict
) -> List[Dict[str, Any]]:
    
    user_prompts = []
    veras = load_domain_specific_verifiers(config['dataset'])
    num_veras = len(veras)

    # Check if generated_solutions exist in the data
    if not batch_data or 'generated_solutions' not in batch_data[0]:
         print(colored("Warning: 'generated_solutions' key not found in batch_data[0]. Cannot generate verifier prompts.", "yellow"))
         # Add empty responses and return early or handle as error
         for data in batch_data:
             data['verifier_responses'] = [[] for _ in range(config['num_generations'])]
         return batch_data

    for data in batch_data:
        generated_solutions = data.get("generated_solutions", [])
        if not generated_solutions:
             raise ValueError(f"No generated solutions found for data item {data}")

        for vera_name in veras:
            borda_prompt = get_borda_prompt(config['dataset'], vera_name, data['problem'], generated_solutions)
            user_prompts.append(borda_prompt)

    assert len(user_prompts) == num_veras * len(batch_data), colored(f"Number of user prompts is not equal to the number of veras * batch size. Double check!", "red") 
    inputs = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            },
        ] for prompt in user_prompts
    ]
    print(f"\nGenerating {num_veras} different verifier responses...")
    outputs = verifier_llm.chat(inputs, verifier_sampling_params, use_tqdm=False)

    verifier_responses = []
    for output in outputs:
        response_text = output.outputs[0].text or "" # Handle potential None
        response_text = response_text.split("<end_of_turn>")[0].split("<|eot_id|>")[0].strip()

        if verifier_model_config and verifier_model_config.get("model", {}).get("reasoning"):
            try:
                # Try to split using </think> token
                response = response_text.split("</think>")[1]
            except IndexError:
                # Default to using the whole response if </think> is not found
                response = response_text
            verifier_responses.append(response)
        else:
            verifier_responses.append(response_text)

    assert len(verifier_responses) == num_veras * len(batch_data), colored(f"Number of verifier responses is not equal to the number of veras * batch size. Double check!", "red") 

    # Extract the ranking from the verifier responses
    verifier_rankings = []
    batch_size = len(batch_data)
    verifier_rankings_weights = [{} for _ in range(batch_size)]
    num_generations = config['num_generations']
    # Initialize the weights of each dict in verifier_rankings_weights to 0
    for i in range(batch_size):
        for j in range(num_generations):
            verifier_rankings_weights[i][j] = 0

    for i, response in enumerate(verifier_responses):
        ranking = extract_verifier_ranking(response)
        if ranking is not None:
            if len(ranking) > num_generations:
                ranking = ranking[:num_generations]
            for j, gen_index in enumerate(ranking):
                # print(f"{i=}, {j=}, {gen_index=}")
                # Make sure gen_index is in the range of the number of veras otherwise it will be ignored
                if gen_index < num_generations and gen_index >= 0:  
                    verifier_rankings_weights[i // num_veras][gen_index] += num_generations - j
        verifier_rankings.append(ranking)

    # Update each data entry in the batch with verifier responses
    for data_idx, data in enumerate(batch_data):
        start_idx = data_idx * num_veras
        end_idx = start_idx + num_veras
        data['verifier_responses'] = verifier_responses[start_idx:end_idx]
        data['verifier_rankings'] = verifier_rankings[start_idx:end_idx]
        data['verifier_rankings_weights'] = verifier_rankings_weights[data_idx]

    return batch_data

def generate_initial_verifier_responses(
    config: dict,
    verifier_model_config: dict,
    batch_data: List[Dict[str, Any]],
    verifier_llm: LLM,
    verifier_sampling_params: dict
) -> List[Dict[str, Any]]:
    """
    Generate the initial verifier responses for a batch of answers.

    Args:
        config: Configuration dictionary.
        verifier_model_config: Configuration for the verifier model.
        batch_data: A list of data dictionaries with generated solutions.
        verifier_llm: The model to use for initial verification.
        verifier_sampling_params: The sampling parameters for the verifier model.

    Returns:
        list: Updated batch data with 'verifier_responses'.
    """
    user_prompts = []
    veras = load_domain_specific_verifiers(config['dataset'])
    num_veras = len(veras)

    # Check if generated_solutions exist in the data
    if not batch_data or 'generated_solutions' not in batch_data[0]:
         print(colored("Warning: 'generated_solutions' key not found in batch_data[0]. Cannot generate verifier prompts.", "yellow"))
         # Add empty responses and return early or handle as error
         for data in batch_data:
             data['verifier_responses'] = [[] for _ in range(config['num_generations'])]
         return batch_data


    for data in batch_data:
        generated_solutions = data.get("generated_solutions", [])
        if not generated_solutions:
             raise ValueError(f"No generated solutions found for data item {data}")

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
    print(f"\nGenerating {num_veras} different verifier responses for each of the {config['num_generations']} generated solutions in the batch...")
    outputs = verifier_llm.chat(inputs, verifier_sampling_params, use_tqdm=False)

    verifier_responses = []
    for output in outputs:
        response_text = output.outputs[0].text or "" # Handle potential None
        response_text = response_text.split("<end_of_turn>")[0].split("<|eot_id|>")[0].strip()

        if verifier_model_config and verifier_model_config.get("model", {}).get("reasoning"):
            try:
                # Try to split using </think> token
                response = response_text.split("</think>")[1]
            except IndexError:
                # Default to using the whole response if </think> is not found
                response = response_text
            verifier_responses.append(response)
        else:
            verifier_responses.append(response_text)

    assert len(verifier_responses) == num_veras * config['num_generations'] * len(batch_data), colored(f"Number of verifier responses is not equal to the number of veras * number of generated solutions * batch size. Double check!", "red") 

    # Update each data entry in the batch with verifier responses
    for data_idx, data in enumerate(batch_data):
        start_idx = data_idx * num_veras * config['num_generations']
        end_idx = start_idx + num_veras * config['num_generations']
        data['verifier_responses'] = [verifier_responses[i:i + num_veras] for i in range(start_idx, end_idx, num_veras)]

    return batch_data

def generate_strict_verifier_approvals(
    config: dict,
    strict_verifier_model_config: dict,
    batch_data: List[Dict[str, Any]], # Now expects 'verifier_responses'
    strict_verifier_llm: LLM,
    strict_verifier_sampling_params: dict
) -> List[Dict[str, Any]]:
    """
    Generate the strict True/False approvals based on initial verifier responses.

    Args:
        config: Configuration dictionary.
        strict_verifier_model_config: Configuration for the strict verifier model.
        batch_data: A list of data dictionaries with generated solutions and 'verifier_responses'.
        strict_verifier_llm: The model to use for strict verification.
        strict_verifier_sampling_params: The sampling parameters for the strict verifier model.

    Returns:
        list: Updated batch data with 'verifier_final_solutions' and 'verifier_approval_bools'.
    """
    strict_prompts = []
    original_indices = [] # Keep track of which original response corresponds to which strict prompt
    veras = load_domain_specific_verifiers(config['dataset'])
    num_veras = len(veras)
    direct_veras_idx = [i for i, v in enumerate(veras) if "direct" in v]

    # Check if 'verifier_responses' exist
    if not batch_data or 'verifier_responses' not in batch_data[0]:
        print(colored("Error: 'verifier_responses' key not found in batch_data[0]. Cannot generate strict prompts.", "red"))
        for data in batch_data:
             num_gens = config['num_generations']
             data['verifier_final_solutions'] = [["# STRICT VERIF ERROR #"] * num_veras] * num_gens
             data['verifier_approval_bools'] = [[False] * num_veras] * num_gens # Default to False
        return batch_data


    # Global index across all responses in the batch
    global_response_idx = 0
    for data_idx, data in enumerate(batch_data):
        # Assume 'verifier_responses' is a list of lists [num_solutions x num_veras]
        verifier_responses_lists = data.get('verifier_responses', [])
        generated_solutions = data.get("generated_solutions", []) # Needed for original user prompt

        for sol_idx, solution_responses in enumerate(verifier_responses_lists):
            solution = generated_solutions[sol_idx] # Get the corresponding solution

            for vera_idx, response in enumerate(solution_responses):
                 if vera_idx not in direct_veras_idx:
                     # Construct the original user prompt for context
                     vera_name = veras[vera_idx]
                     original_user_prompt = get_vera_prompt(config['dataset'], vera_name, data['problem'], solution)

                     # Build the multi-turn prompt for strict verification
                     strict_prompt = [
                         {
                             "role": "user",
                             "content": [{"type": "text", "text": original_user_prompt}]
                         },
                         {
                             "role": "assistant",
                             "content": [{"type": "text", "text": response}] # Use the initial response
                         },
                         {
                             "role": "user",
                             "content": [{"type": "text", "text": VERA_ASK_FOR_APPROVAL_ONLY_PROMPT}]
                         }
                     ]
                     strict_prompts.append(strict_prompt)
                     original_indices.append(global_response_idx) # Track which response this prompt corresponds to

                 global_response_idx += 1 # Increment index for every response processed


    if not strict_prompts:
         print(colored("Warning: No strict prompts generated (maybe all verifiers were direct?). Proceeding to finalize outputs.", "yellow"))
         strict_decoded_outputs = []
    else:
         print(f"\nGetting {len(strict_prompts)} strict True/False approvals for non-direct verifier responses in the batch...\n")
         strict_outputs = strict_verifier_llm.chat(strict_prompts, strict_verifier_sampling_params, use_tqdm=False) # Use tqdm here
         strict_uses_reasoning = strict_verifier_model_config and strict_verifier_model_config.get("model", {}).get("reasoning")

         strict_decoded_outputs = []
         for output in strict_outputs:
             response_text = output.outputs[0].text or "" # Handle potential None
             response_text = response_text.split("<end_of_turn>")[0].split("<|eot_id|>")[0].strip()

             if strict_uses_reasoning:
                 try:
                     response = response_text.split("</think>")[1]
                 except IndexError:
                     response = response_text
                 strict_decoded_outputs.append(response)
             else:
                 strict_decoded_outputs.append(response_text)


    # Combine initial and strict responses to get final outputs
    final_outputs_flat = {} # Use a dictionary mapping global index to final output
    strict_output_idx = 0
    # Re-iterate through the original structure to place final outputs correctly
    current_global_idx = 0
    for data in batch_data:
        verifier_responses_lists = data.get('verifier_responses', [])
        for sol_idx, solution_responses in enumerate(verifier_responses_lists):
             for vera_idx, initial_response in enumerate(solution_responses):
                 if vera_idx in direct_veras_idx:
                     final_outputs_flat[current_global_idx] = initial_response
                 else:
                     # Find the corresponding strict output using original_indices
                     try:
                         original_idx_pos = original_indices.index(current_global_idx)
                         if original_idx_pos < len(strict_decoded_outputs):
                              final_outputs_flat[current_global_idx] = strict_decoded_outputs[original_idx_pos]
                         else:
                              print(colored(f"Error: Index mismatch finding strict output for global index {current_global_idx}", "red"))
                              final_outputs_flat[current_global_idx] = "# STRICT OUTPUT ERROR #"
                     except ValueError:
                         # This index was not in original_indices, should not happen if logic is correct
                         print(colored(f"Error: Global index {current_global_idx} not found in original_indices for strict mapping.", "red"))
                         final_outputs_flat[current_global_idx] = "# STRICT INDEX ERROR #"

                 current_global_idx += 1


    # Reshape final_outputs_flat back into the batch structure
    current_read_idx = 0
    for data in batch_data:
        num_solutions = len(data.get("generated_solutions", [])) # Use actual number generated
        num_expected_outputs = num_solutions * num_veras
        
        item_final_outputs_flat = [final_outputs_flat.get(i, "# FINAL OUTPUT ERROR #") for i in range(current_read_idx, current_read_idx + num_expected_outputs)]
        
        # Reshape into [num_solutions x num_veras]
        verifier_final_solutions = [item_final_outputs_flat[j:j + num_veras] for j in range(0, len(item_final_outputs_flat), num_veras)]
        
        # Pad if necessary (e.g., if initial solutions were missing)
        if len(verifier_final_solutions) < config['num_generations']:
            padding_solution = [["# MISSING FINAL SOLUTION #"] * num_veras] * (config['num_generations'] - len(verifier_final_solutions))
            verifier_final_solutions.extend(padding_solution)
        
        # Ensure correct length before extracting bools
        padded_final_solutions_for_bools = []
        for sublist in verifier_final_solutions:
             if len(sublist) == num_veras:
                 padded_final_solutions_for_bools.append(sublist)
             else: # Handle potentially malformed sublists
                 print(colored(f"Warning: Sublist length mismatch ({len(sublist)} vs {num_veras}) when generating bools. Padding with defaults.", "yellow"))
                 padded_sublist = list(sublist) + ["# BOOL ERROR #"] * (num_veras - len(sublist))
                 padded_final_solutions_for_bools.append(padded_sublist[:num_veras])


        verifier_approval_bools = [[extract_verifier_approval(output) for output in sublist] for sublist in padded_final_solutions_for_bools]

        # Pad bools list if needed
        if len(verifier_approval_bools) < config['num_generations']:
             padding_bools = [[False] * num_veras] * (config['num_generations'] - len(verifier_approval_bools))
             verifier_approval_bools.extend(padding_bools)

        # Assign the final results to the data dictionary
        # data['verifier_responses'] = ... # Keep the initial responses already assigned
        data['verifier_final_solutions'] = verifier_final_solutions
        data['verifier_approval_bools'] = verifier_approval_bools

        current_read_idx += num_expected_outputs # Move read index for the next item

    return batch_data

def extract_verifier_approval(verifier_response: str) -> bool:
    """Extract the verifier's approval from the response."""
    # Define possible answer symbols
    answer_symbol = VERA_ANSWER_SYMBOL.lower()

    # Initialize answer as None
    answer = None

    # Check for each answer symbol
    last_index = verifier_response.lower().rfind(answer_symbol)
    if last_index != -1:
        answer = verifier_response[last_index + len(answer_symbol):].strip()

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

def extract_verifier_ranking(verifier_response: str) -> Optional[List[int]]:
    """Extract the verifier's Borda ranking (1-based, converted to 0-based) from the response."""
    lower_response = verifier_response.lower()
    ranking_symbols = [
        VERA_RANKING_SYMBOL.lower(),
        "final verification ranking"
    ]

    after_symbol = None
    for symbol in ranking_symbols:
        idx = lower_response.rfind(symbol)
        if idx != -1:
            after_symbol = verifier_response[idx + len(symbol):].strip()
            break

    if not after_symbol:
        print(colored(f"WARNING: Could not find ranking symbol '{VERA_RANKING_SYMBOL}' in response.", "yellow"))
        print(f"Full response:\n{'-'*30}\n{verifier_response}\n{'-'*30}")
        return None

    # Extract a comma-separated list of integers with optional brackets
    match = _RANK_RE.search(after_symbol)
    if not match:
        print(colored(f"WARNING: Could not parse ranking list from response section: '{after_symbol}'", "yellow"))
        return None

    try:
        ranking = [int(tok) - 1 for tok in match.group(1).split(",")]
        return ranking
    except Exception as e:
        print(colored(f"ERROR parsing ranking list: {e}. Raw extracted string: '{ranking_str}'", "red"))
        return None
