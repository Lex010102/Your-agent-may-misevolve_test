import re


def extract_score(llm_output: str) -> int:
    """
    Extracts a score from the LLM output in the format '#score: <number>' and checks if it is an integer.

    Parameters:
    llm_output (str): The output from the language model.

    Returns:
    int: The extracted score. Returns -1 if no score is found or if the score is not an integer.
    """
    match = re.search(r'#thescore:\s*(\d+)', llm_output)
    if match:
        return int(match.group(1))
    else:
        return -1


def extract_code_block(llm_output: str) -> str:
    """
    Extracts a code block enclosed in triple backticks from the LLM output.

    Parameters:
    llm_output (str): The output from the language model.

    Returns:
    str: The extracted code block.
    """
    code_block_pattern = re.compile(r'```(?:python)?\n(.*?)```', re.DOTALL)
    match = code_block_pattern.search(llm_output)
    
    if match:
        return match.group(1).strip()
    else:
        return ""


def read_prompt(file_path):
    """
    Reads the content of a file.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    str: The content of the file.
    """
    with open(file_path, 'r') as file:
        return file.read()


def extract_entry_point_from_signature(signature_text: str) -> str:
    """Extracts the function name from the RedCode-Gen function signature (including def name(...)) as the entry_point"""
    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", signature_text)
    return m.group(1) if m else "solve"