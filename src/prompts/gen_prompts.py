def get_gen_prompt(dataset_name: str, problem: str | list[str]):
    # Handle both single problem (str) and multiple problems (list)
    if isinstance(problem, list):
        return [get_gen_prompt(dataset_name, p) for p in problem]
    
    # MATH
    gen_prompt_math = (
        "You are a helpful assistant skilled in math problem-solving. "
        "Always end your solution with the final numerical answer in latex, using '\\boxed{<answer>}'. "
        "If there is no solution, reply with an empty boxed '\\boxed{}'."
        "\nPlease solve the following math problem step by step:"
        f"\n\nQUESTION: {problem}"
        "\n\nProvide your detailed solution below:"
    )
    #TODO: Add prompts for other datasets
    
    if dataset_name == "math":
        return gen_prompt_math

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

