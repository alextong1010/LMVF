VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER:"
VERA_RANKING_SYMBOL = "FINAL VERIFICATION RANKING:"

# For verifiers other than direct approval, we ask a follow up message since it is sometimes unclear what the verifier decided
VERA_ASK_FOR_APPROVAL_ONLY_PROMPT = f"To clarify, based on the above analysis, reply with ONLY '{VERA_ANSWER_SYMBOL}True' or ONLY '{VERA_ANSWER_SYMBOL}False'. Do not include any other text in your response."


def is_not_direct_approval(vera_name: str) -> bool:
    return "direct" not in vera_name


def get_vera_prompt(dataset_name, vera_name, question, solution):
    # system string should be a single line (no newlines)
    system_str_math = (
        "You are a critical verifier tasked with evaluating mathematical problem-solving. "
        "You will be presented with a question and a proposed solution. "
        "Your job is to carefully go over and analyze the solution. Follow the instructions."
    )
    #TODO: Add prompts for other datasets
    if dataset_name == "math":
        system_str = system_str_math
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    prefix = f"""{system_str}\n\n
    QUESTION:
    {question}\n\n
    PROPOSED SOLUTION:
    {solution}\n\n"""

    vera_names_to_prompts = {
        "math_steps": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Go over each step in the proposed solution and check whether it is mathematically correct. Think out load. "
            f"If you reach a step that is incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'."
            f"If you get to the end of all the steps and each step was correct, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "units_steps": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Check if the units are handled correctly in each step of the solution. Think out loud. "
            f"If you find any issues with the units, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If all units are handled correctly, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_summarize": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Summarize the solution in your own words, explore anything you think may be incorrect. Think out loud. "
            f"If you find something that's incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If you've gone over the solution and everything seems correct, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_edge": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Check if the solution handles edge cases and boundary conditions, test extreme values or special cases. Think out loud. "
            f"If any boundary conditions or edge cases fail, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If all boundary conditions and edge cases are handled correctly, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_mistakes": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Check if the solution has any common mistakes, calculation errors, or misconceptions that typically found in this type of problem. Think out loud. "
            f"If you find any common mistakes, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If no common mistakes are found, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_domain": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Check if the solution correctly applies relevant domain-knowledge, established theories, and standard practices for this type of problem. Think out loud. "
            f"If any domain knowledge is misapplied or violated, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If all domain-specific knowledge is correctly applied, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
    }
    return vera_names_to_prompts[vera_name]

def get_borda_prompt(dataset_name, vera_name, question, solutions_list):
    # system string should be a single line (no newlines)
    system_str_math = (
        "You are a critical verifier tasked with evaluating mathematical problem-solving. "
        "You will be presented with a question and a proposed solution. "
        "Your job is to carefully go over and analyze the solution. Follow the instructions."
    )
    #TODO: Add prompts for other datasets
    if dataset_name == "math":
        system_str = system_str_math
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Iterate through the solutions and format them as a numbered list
    solutions_str = ""
    for i, solution in enumerate(solutions_list, 1):
        solutions_str += f"Solution {i}:\n{solution}\n\n"

    prefix = f"""{system_str}\n\n
    QUESTION:
    {question}\n\n
    PROPOSED SOLUTIONS:
    {solutions_str}\n\n"""

    vera_names_to_prompts = {
        "math_steps": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "You are given multiple proposed solutions to the same math problem above.\n"
            "Go over each solution and evaluate the mathematical correctness of the steps. Think out loud. \n"
            "Then rank all of the solutions from most to least mathematically correct.\n"
            f"Reply with '{VERA_RANKING_SYMBOL} [1, 2, ...]' where each number corresponds to the solution number, from best to worst. "
            "Make sure to include the square brackets and commas. Make sure to not include more numbers than the number of solutions."
        ),
        "units_steps": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "You are given multiple proposed solutions to the same problem above.\n"
            "For each solution, evaluate how well units are handled at every step. Think out loud. \n"
            "Rank all of the solutions from best to worst based on correctness and consistency of units.\n"
            f"Reply with '{VERA_RANKING_SYMBOL} [1, 2, ...]' where each number corresponds to the solution number, from best to worst. "
            "Make sure to include the square brackets and commas. Make sure to not include more numbers than the number of solutions."
        ),
        "general_summarize": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "You are given multiple proposed solutions to the same problem above.\n"
            "Review each solution and summarize its approach in your own words, explore anything you think may be incorrect.\n"
            "Then evaluate the overall correctness and clarity of reasoning. Think out loud. \n"
            "Rank all of the solutions from most to least correct overall.\n"
            f"Reply with '{VERA_RANKING_SYMBOL} [1, 2, ...]' where each number corresponds to the solution number, from best to worst. "
            "Make sure to include the square brackets and commas. Make sure to not include more numbers than the number of solutions."
        ),
        "general_edge": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "You are given multiple proposed solutions to the same problem above.\n"
            "Test each solution for how well it handles edge cases and boundary conditions.\n"
            "Consider extreme values and special inputs where appropriate. Think out loud. \n"
            "Rank all of the solutions from best to worst based on how well they address these cases.\n"
            f"Reply with '{VERA_RANKING_SYMBOL} [1, 2, ...]' where each number corresponds to the solution number, from best to worst. "
            "Make sure to include the square brackets and commas. Make sure to not include more numbers than the number of solutions."
        ),
        "general_mistakes": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "You are given multiple proposed solutions to the same problem above.\n"
            "Check each solution for common mistakes, such as calculation errors or misconceptions. Think out loud. \n"
            "Rank all of the solutions from most to least accurate (i.e., fewest mistakes to most mistakes).\n"
            f"Reply with '{VERA_RANKING_SYMBOL} [1, 2, ...]' where each number corresponds to the solution number, from best to worst. "
            "Make sure to include the square brackets and commas. Make sure to not include more numbers than the number of solutions."
        ),
        "general_domain": (
            f"{prefix}"
            "INSTRUCTIONS:\n"
            "You are given multiple proposed solutions to the same problem above.\n"
            "Evaluate each solution's use of domain-specific knowledge and standard methods. Think out loud. \n"
            "Rank all of the solutions from best to worst based on how well they apply relevant domain knowledge.\n"
            f"Reply with '{VERA_RANKING_SYMBOL} [1, 2, ...]' where each number corresponds to the solution number, from best to worst. "
            "Make sure to include the square brackets and commas. Make sure to not include more numbers than the number of solutions."
        ),
    }
    return vera_names_to_prompts[vera_name]
