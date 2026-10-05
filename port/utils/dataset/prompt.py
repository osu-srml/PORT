"""
Date: 2026-02-02
Reference: https://github.com/exlaw/TIS-DPO
"""
from bs4 import BeautifulSoup, NavigableString

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair. The Anthropic prompt has a 
    structure of \n\nHuman: <prompt>\n\nAssistant: 
    
    Args:
        prompt_and_response (str): The combined prompt and response string.
        
    Returns:
        str: The extracted prompt.
        
    Example:
        prompt_and_response = '\n\nHuman: What is the capital of France?\n\nAssistant: The capital of France is Paris.'
        >>> extract_anthropic_prompt(prompt_and_response)
        '\n\nHuman: What is the capital of France?\n\nAssistant:'
    """
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def split_prompt_and_responses(ex):
    """Split an example into prompt, chosen response, and rejected response.
    
    Args:
        ex (Dict): A dictionary with at least 'chosen' and 'rejected' keys.
        
    Returns:
        Tuple[str, str, str]: A tuple containing the prompt, chosen response, and rejected response.
    
    Example:
        ex = {
            'chosen': '\n\nHuman: What is the capital of France?\n\nAssistant: The capital of France is Paris.',
            'rejected': '\n\nHuman: What is the capital of France?\n\nAssistant: The capital of France is London.'
        }
        >>> split_prompt_and_responses(ex)
        ('\n\nHuman: What is the capital of France?\n\nAssistant:',
         ' The capital of France is Paris.',
         ' The capital of France is London.')
    """
    prompt = extract_anthropic_prompt(ex['chosen'])
    chosen_response = ex['chosen'][len(prompt):]
    rejected_response = ex['rejected'][len(prompt):]
    return prompt, chosen_response, rejected_response

def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers).
    
    Args:
        html_string (str): The HTML string to strip.
        
    Returns:
        str: The stripped string.
        
    Example:
        html_string = "<p>This is a <b>bold</b> statement.</p
        >>> strip_html_tags(html_string)
        "This is a bold statement."
    """
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text

def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise.
    
    Args:
        str_a (str): The first string to compare.
        str_b (str): The second string to compare.
    Returns:
        bool: True if the strings match up to spaces, False otherwise.
    Example:
        >>> strings_match_up_to_spaces("hello world", "helloworld")
        True
        >>> strings_match_up_to_spaces("hello  world", "helloworld")
        True
        >>> strings_match_up_to_spaces("hello world", "hello  world!")
        False
    """
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True