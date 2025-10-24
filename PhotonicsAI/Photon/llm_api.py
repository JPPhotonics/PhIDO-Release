# Standard library imports
import ast
import json
import os
import re

# Third-party imports
import anthropic
import backoff
import google.generativeai as genai
import tiktoken
import yaml
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel
from deepseek_tokenizer import ds_token

# Local imports
from PhotonicsAI.config import CONF, PATH

load_dotenv()

# Session-specific token tracking for Google Gemini models
def get_session_token_usage():
    """Get token usage from current session state."""
    import streamlit as st
    if 'token_usage' not in st.session_state:
        st.session_state.token_usage = {
            "non_cached_input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0
        }
    return st.session_state.token_usage

def reset_token_usage():
    """Reset token usage counters for current session."""
    import streamlit as st
    st.session_state.token_usage = {
        "non_cached_input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0
    }

def get_token_usage():
    """Get current token usage for current session."""
    return get_session_token_usage().copy()

def add_token_usage(input_tokens, output_tokens, is_cached=False):
    """Add token usage to the current session counter."""
    token_usage = get_session_token_usage()
    if is_cached:
        token_usage["cached_input_tokens"] += input_tokens
    else:
        token_usage["non_cached_input_tokens"] += input_tokens
    token_usage["output_tokens"] += output_tokens

def debug_token_usage():
    """Debug function to print current session token usage."""
    import streamlit as st
    token_usage = get_token_usage()
    print(f"Current session token usage: {token_usage}")
    if hasattr(st.session_state, '_session_id'):
        print(f"Session ID: {st.session_state._session_id}")

def debug_anthropic_response(message):
    """Debug function to inspect Anthropic response structure."""
    print("=== Anthropic Response Debug ===")
    print(f"Response type: {type(message)}")
    print(f"Response attributes: {dir(message)}")
    
    # Check for usage information
    if hasattr(message, 'usage'):
        print(f"Usage object: {message.usage}")
        print(f"Usage type: {type(message.usage)}")
        if message.usage:
            print(f"Usage attributes: {dir(message.usage)}")
            if hasattr(message.usage, 'input_tokens'):
                print(f"Input tokens: {message.usage.input_tokens}")
            if hasattr(message.usage, 'output_tokens'):
                print(f"Output tokens: {message.usage.output_tokens}")
    else:
        print("No usage object found in response")
    
    # Check for content blocks (thinking and text)
    if hasattr(message, 'content'):
        print(f"Content blocks: {len(message.content)}")
        thinking_blocks = 0
        text_blocks = 0
        
        for i, block in enumerate(message.content):
            print(f"Block {i}: type={block.type}")
            if block.type == "thinking":
                thinking_blocks += 1
                print(f"  Thinking summary: {block.thinking[:100]}...")
            elif block.type == "text":
                text_blocks += 1
                print(f"  Text length: {len(block.text)} characters")
        
        print(f"Total thinking blocks: {thinking_blocks}")
        print(f"Total text blocks: {text_blocks}")
    
    print("================================")

try:
    with open(PATH.prompts) as file:
        prompts = yaml.safe_load(file)
except FileNotFoundError:
    print(f"No {PATH.prompts} file found.")
    pass

LOCATION='us-east5'

def call_anthropic(prompt, sys_prompt, model='claude-3-7-sonnet-20250219'):

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    message = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        # temperature=0.1,
        system=sys_prompt,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    # Track token usage
    try:
        # Debug the response structure to understand what's available
        debug_anthropic_response(message)
        
        # Check if the response has usage information (including thinking tokens)
        if hasattr(message, 'usage') and message.usage:
            # Use the usage information from the response if available
            input_tokens = message.usage.input_tokens if hasattr(message.usage, 'input_tokens') else 0
            output_tokens = message.usage.output_tokens if hasattr(message.usage, 'output_tokens') else 0
            
            # If thinking was enabled, the output_tokens should include thinking tokens
            # according to Anthropic's API documentation
            print(f"Anthropic response usage - Input: {input_tokens}, Output: {output_tokens}")
            print("Note: If thinking was enabled, output_tokens includes thinking tokens")
            
        else:
            # Fallback to manual token counting
            # Count input tokens (system prompt + user prompt)
            count_response = client.messages.count_tokens(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            input_tokens = count_response.input_tokens
            
            # Add system prompt tokens
            if sys_prompt:
                sys_count_response = client.messages.count_tokens(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt}
                    ]
                )
                input_tokens += sys_count_response.input_tokens
            
            # For output tokens, count the response content
            # Extract all text blocks from the response
            response_text = ""
            for block in message.content:
                if block.type == "text":
                    response_text += block.text
            
            output_tokens = client.messages.count_tokens(
                model=model,
                messages=[
                    {"role": "assistant", "content": response_text}
                ]
            ).input_tokens
            
            # Note: This doesn't include thinking tokens since we can't access them
            # when using manual counting. The thinking tokens are only available
            # in the response.usage object when the API provides it.
            print(f"Manual token counting - Input: {input_tokens}, Output: {output_tokens}")
            print("Warning: Thinking tokens not included in manual counting")
        
        # Add to session token usage (assuming not cached for now)
        add_token_usage(input_tokens, output_tokens, is_cached=False)
        
    except Exception as e:
        # If token tracking fails, continue without it
        print(f"Token tracking error in call_anthropic: {e}")
        # Fallback to tiktoken estimation
        try:
            input_tokens = len(tokenizer.encode(prompt + sys_prompt))
            
            # Extract all text blocks from the response
            response_text = ""
            for block in message.content:
                if block.type == "text":
                    response_text += block.text
            
            output_tokens = len(tokenizer.encode(response_text))
            add_token_usage(input_tokens, output_tokens, is_cached=False)
            print(f"Fallback tiktoken counting - Input: {input_tokens}, Output: {output_tokens}")
            print("Warning: Thinking tokens not included in fallback counting")
        except Exception as e2:
            print(f"Fallback token tracking also failed: {e2}")
    
    # Summary of token counting approach:
    # 1. If response.usage is available: Uses Anthropic's official token counts (includes thinking tokens)
    # 2. If not available: Manual counting of system+user prompts and response text (excludes thinking tokens)
    # 3. Fallback: tiktoken estimation (excludes thinking tokens)
    
    # Extract all text blocks for the final response
    response_text = ""
    for block in message.content:
        if block.type == "text":
            response_text += block.text
    
    with open('anthropic_response.yml', 'w') as outfile:
        yaml.dump(message.content, outfile)
    return response_text

def call_google(prompt, sys_prompt, model='gemini-2.5-pro'):
    """Calling google API using GenerativeModel with enhanced thinking support.

    Args:
        prompt: The prompt to send to the model.
        sys_prompt: The system prompt to send to the model.
        model: The model to use for the completion.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        print(f"Google Generative AI library not available: {e}")
        print("Falling back to a simple response indicating the issue.")
        return "Error: Google Generative AI library not properly installed or configured."
    
    try:
        # Configure the API key
        genai.configure(api_key=os.getenv("GOOGLEGENAI_API_KEY"))
    except Exception as e:
        print(f"Failed to configure Google Generative AI: {e}")
        return "Error: Failed to configure Google Generative AI API."
    
    prompt = truncate_prompt(prompt)
    
    try:
        # Create model with system instruction
        model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=sys_prompt
        )
    except Exception as e:
        print(f"Failed to create Google Generative AI model: {e}")
        return "Error: Failed to create Google Generative AI model."
    
    # Try to enable thinking with budget off (-1 means unlimited)
    try:
        # Try with ThinkingConfig if available
        response = model_instance.generate_content(
            prompt,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=-1),  # Unlimited thinking
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0.1
            ),
            safety_settings={
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        )
        print("Using ThinkingConfig with unlimited budget and disabled safety filters")
    except Exception as e:
        print(f"ThinkingConfig failed: {e}")
        try:
            # Try alternative thinking approaches
            response = model_instance.generate_content(
                prompt,
                thinking=True,  # Try direct thinking parameter
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0.1
                ),
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )
            print("Using thinking=True parameter with disabled safety filters")
        except Exception as e2:
            print(f"Direct thinking parameter failed: {e2}")
            try:
                # Fallback to basic generation - thinking might be enabled by default
                response = model_instance.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        temperature=0.1
                    ),
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                print("Using basic generation with disabled safety filters")
            except Exception as e3:
                print(f"All approaches failed: {e3}")
                return "Error: Failed to generate content with Google Generative AI."
    
    # Track token usage with detailed thinking token tracking
    try:
        # Get token counts from response metadata
        input_tokens = 0
        output_tokens = 0
        thoughts_tokens = 0
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            # Access token counts from usage metadata
            usage_metadata = response.usage_metadata
            
            # Get input tokens (prompt tokens)
            if hasattr(usage_metadata, 'prompt_token_count'):
                input_tokens = usage_metadata.prompt_token_count
            elif hasattr(usage_metadata, 'input_token_count'):
                input_tokens = usage_metadata.input_token_count
            
            # Get output tokens (candidate tokens)
            if hasattr(usage_metadata, 'candidates_token_count'):
                output_tokens = usage_metadata.candidates_token_count
            elif hasattr(usage_metadata, 'output_token_count'):
                output_tokens = usage_metadata.output_token_count
            
            # Get thinking tokens if available
            if hasattr(usage_metadata, 'thoughts_token_count'):
                thoughts_tokens = usage_metadata.thoughts_token_count
            elif hasattr(usage_metadata, 'thinking_token_count'):
                thoughts_tokens = usage_metadata.thinking_token_count
            
            # Debug: print all available attributes in usage_metadata
            print(f"Usage metadata attributes: {[x for x in dir(usage_metadata) if not x.startswith('_')]}")
        
        print(f"Google response usage - Input: {input_tokens}, Output: {output_tokens}, Thoughts: {thoughts_tokens}")
        print(f"Total tokens (including thinking): {input_tokens + output_tokens + thoughts_tokens}")
        
        # Add thoughts tokens to output tokens for total tracking
        total_output_tokens = output_tokens + thoughts_tokens
        
        # If we can't get token counts from response, estimate them
        if input_tokens == 0:
            # Estimate input tokens (prompt + system prompt)
            input_tokens = len(tokenizer.encode(prompt + sys_prompt))
        if total_output_tokens == 0:
            # Estimate output tokens
            total_output_tokens = len(tokenizer.encode(response.text))
        
        # Add to global token usage (assuming not cached for now)
        add_token_usage(input_tokens, total_output_tokens, is_cached=False)
        
    except Exception as e:
        # If token tracking fails, continue without it
        print(f"Token tracking error: {e}")
    
    # Check if response has valid content
    if not response or not hasattr(response, 'text') or not response.text:
        print("Warning: Google API returned empty or invalid response")
        return "Error: No valid response from Google API. The request may have been blocked or filtered."
    
    # Check for finish reason indicating blocked content
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 1:
                print("Warning: Google API response was blocked/filtered (finish_reason=1)")
                return "Error: Response was blocked by Google API safety filters. Please try rephrasing your request."
    
    with open('google_response.yml', 'w') as outfile:
        yaml.dump(response.text.replace("```", "").replace("yaml", "").replace("dot\n", ""), outfile)
    return response.text.replace("```", "").replace("yaml", "").replace("dot\n", "")

tokenizer = tiktoken.get_encoding("o200k_base")

def count_deepseek_tokens(text):
    """Count tokens using the fast DeepSeek tokenizer."""
    try:
        return len(ds_token.encode(text))
    except Exception as e:
        print(f"DeepSeek tokenizer error: {e}")
        return None


def truncate_prompt(prompt, max_tokens=120000):
    """Truncate the input prompt to the maximum allowed tokens.

    Args:
        prompt (str): The input prompt to truncate.
        max_tokens (int): The maximum number of tokens allowed.
    """
    # Tokenize the input prompt
    tokens = tokenizer.encode(prompt)

    # Check if the prompt exceeds the maximum allowed tokens
    if len(tokens) > max_tokens:
        # Truncate the prompt by keeping only the last `max_tokens` tokens
        tokens = tokens[-max_tokens:]

        # Decode tokens back to string
        truncated_prompt = tokenizer.decode(tokens)
        return truncated_prompt
    return prompt


def call_nvidia(prompt, sys_prompt="", model="nvidia/llama-3.1-nemotron-ultra-253b-v1", n_completion=1):
    """Calling openai API.

    Args:
        prompt: The prompt to send to the model.
        sys_prompt: The system prompt to send to the model.
        model: The model to use for the completion.
    """
    prompt = truncate_prompt(prompt)

    client = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key=os.getenv("NVIDIA_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        top_p = 0.7,
        n=n_completion,
        stream = False,
        messages=[
            {"role": "system", "content": "detailed thinking off"},
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    
    # Track token usage
    try:
        # Get token counts from response
        input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else 0
        output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else 0
        
        # If we can't get token counts from response, estimate them
        if input_tokens == 0:
            # Estimate input tokens (prompt + system prompt)
            input_tokens = len(tokenizer.encode(prompt + sys_prompt))
        if output_tokens == 0:
            # Estimate output tokens
            if n_completion == 1:
                output_tokens = len(tokenizer.encode(response.choices[0].message.content))
            else:
                # For multiple completions, sum all output tokens
                output_tokens = sum(len(tokenizer.encode(choice.message.content)) for choice in response.choices)
        
        # Add to session token usage (assuming not cached for now)
        add_token_usage(input_tokens, output_tokens, is_cached=False)
        
    except Exception as e:
        # If token tracking fails, continue without it
        print(f"Token tracking error in call_nvidia: {e}")

    # function to remove COT outputs in Nemotron API calls
    splice = lambda x : re.sub(r'<think>.*?</think>', '', x, flags=re.DOTALL)
    
    with open('nvidia_response.yml', 'w') as outfile:
        yaml.dump(response.choices[0].message.content, outfile)

    if n_completion == 1:
        return splice(response.choices[0].message.content)
    else:
        return [splice(r.message.content) for r in response.choices]

def call_openai(prompt, sys_prompt="", model="gpt-4o", n_completion=1):
    """Calling openai API.

    Args:
        prompt: The prompt to send to the model.
        sys_prompt: The system prompt to send to the model.
        model: The model to use for the completion.
    """
    prompt = truncate_prompt(prompt)

    client = OpenAI(api_key=CONF.openai_api_key or os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        n=n_completion,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    
    # Track token usage
    try:
        # Get token counts from response
        input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else 0
        output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else 0
        
        # If we can't get token counts from response, estimate them
        if input_tokens == 0:
            # Estimate input tokens (prompt + system prompt)
            input_tokens = len(tokenizer.encode(prompt + sys_prompt))
        if output_tokens == 0:
            # Estimate output tokens
            if n_completion == 1:
                output_tokens = len(tokenizer.encode(response.choices[0].message.content))
            else:
                # For multiple completions, sum all output tokens
                output_tokens = sum(len(tokenizer.encode(choice.message.content)) for choice in response.choices)
        
        # Add to session token usage (assuming not cached for now)
        add_token_usage(input_tokens, output_tokens, is_cached=False)
        
    except Exception as e:
        # If token tracking fails, continue without it
        print(f"Token tracking error in call_openai: {e}")

    if n_completion == 1:
        return response.choices[0].message.content
    else:
        return [r.message.content for r in response.choices]


def call_openai_reasoning(prompt, model="o1-preview"):
    """Calling openai o1 model.

    Args:
        prompt: The prompt to send to the model.
        model: The model to use for the completion.
    """
    # prompt = truncate_prompt(prompt)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    
    # Track token usage
    try:
        # Get token counts from response
        input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else 0
        output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else 0
        
        # If we can't get token counts from response, estimate them
        if input_tokens == 0:
            # Estimate input tokens
            input_tokens = len(tokenizer.encode(prompt))
        if output_tokens == 0:
            # Estimate output tokens
            output_tokens = len(tokenizer.encode(response.choices[0].message.content))
        
        # Add to session token usage (assuming not cached for now)
        add_token_usage(input_tokens, output_tokens, is_cached=False)
        
    except Exception as e:
        # If token tracking fails, continue without it
        print(f"Token tracking error in call_openai_reasoning: {e}")

    return response.choices[0].message.content


def callgpt_pydantic(prompt, sys_prompt, pydantic_model):
    """Calling openai with pydantic model.

    Args:
        prompt: The prompt to send to the model.
        sys_prompt: The system prompt to send to the model.
        pydantic_model: The pydantic model to use for the completion.
    """
    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format=pydantic_model,
    )

    message = completion.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        print(message.refusal)
        return message.refusal

def calldeepseek_pydantic(prompt, sys_prompt, pydantic_model):
    """Calling openai with pydantic model.

    Args:
        prompt: The prompt to send to the model.
        sys_prompt: The system prompt to send to the model.
        pydantic_model: The pydantic model to use for the completion.
    """
    client = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key=os.getenv("DEEPSEEK_API_KEY"))

    completion = client.beta.chat.completions.parse(
        model="deepseek-ai/deepseek-r1",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format=pydantic_model,
    )
    
    message = completion.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        print(message.refusal)
        return message.refusal

def callgoogle_pydantic(prompt, sys_prompt, pydantic_model):
    genai.configure(api_key=os.getenv("GOOGLEGENAI_API_KEY"))
    prompt = truncate_prompt(prompt)
    model=genai.GenerativeModel(
    model_name='gemini-1.5-pro',
    system_instruction=sys_prompt)

    response = model.generate_content(prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.5,
            response_mime_type='application/json',
            response_schema=pydantic_model)
    )
    
    with open('google_response.yml', 'w') as outfile:
        yaml.dump(response.text, outfile)
    
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    response_dict = json.loads(response.text)
    s = Struct(**response_dict)
    return s

def parse_and_validate_list(string):
    """Parse and validate a list from a string.

    Args:
        string: The string to parse and validate.
    """
    try:
        # Clean up the string - remove any markdown formatting
        cleaned_string = string.strip()
        if cleaned_string.startswith('```'):
            # Remove markdown code blocks
            lines = cleaned_string.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines[-1].startswith('```'):
                lines = lines[:-1]
            cleaned_string = '\n'.join(lines).strip()
        
        # Remove language identifiers at the beginning (like "python", "yaml", etc.)
        lines = cleaned_string.split('\n')
        if lines and lines[0].strip().lower() in ['python', 'yaml', 'json']:
            lines = lines[1:]
            cleaned_string = '\n'.join(lines).strip()
        
        # Step 1: Parse the string
        parsed_list = ast.literal_eval(cleaned_string)

        # Step 2: Check if the parsed result is a list
        if not isinstance(parsed_list, list):
            raise ValueError(f"Parsed result is not a list, got {type(parsed_list)}: {parsed_list}")

        # Step 3: Verify that all elements in the list are integers
        if all(isinstance(item, int) for item in parsed_list):
            return parsed_list
        else:
            non_integers = [item for item in parsed_list if not isinstance(item, int)]
            raise ValueError(f"Not all elements in the list are integers. Non-integers: {non_integers}")

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing list from string: {e}")
        print(f"Original string: {string}")
        return None

# @backoff.on_exception(backoff.expo, RateLimitError)
def call_deepseek(prompt, sys_prompt="", model="deepseek-reasoner", n_completion=1):
    """Calling openai API.

    Args:
        prompt: The prompt to send to the model.
        sys_prompt: The system prompt to send to the model.
        model: The model to use for the completion.
    """
    prompt = truncate_prompt(prompt)

    client = OpenAI(base_url='https://api.deepseek.com/v1', api_key=os.getenv("DEEPSEEK_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=0.6,
        n=n_completion,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    
    while isinstance(response, str):
        response = client.chat.completions.create(
        model=model,
        temperature=0.6,
        n=n_completion,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    
    # Track token usage
    try:
        # Get token counts from response
        input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else 0
        output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else 0
        
        # If we can't get token counts from response, use fast DeepSeek tokenizer
        if input_tokens == 0 or output_tokens == 0:
            # Use fast DeepSeek tokenizer for accurate counting
            if input_tokens == 0:
                # Count input tokens (prompt + system prompt)
                input_text = prompt + sys_prompt
                input_tokens = count_deepseek_tokens(input_text)
                if input_tokens is None:
                    # Fallback to tiktoken
                    input_tokens = len(tokenizer.encode(input_text))
            
            if output_tokens == 0:
                # Count output tokens
                if n_completion == 1:
                    output_tokens = count_deepseek_tokens(response.choices[0].message.content)
                    if output_tokens is None:
                        output_tokens = len(tokenizer.encode(response.choices[0].message.content))
                else:
                    # For multiple completions, sum all output tokens
                    output_tokens = 0
                    for choice in response.choices:
                        token_count = count_deepseek_tokens(choice.message.content)
                        if token_count is None:
                            token_count = len(tokenizer.encode(choice.message.content))
                        output_tokens += token_count
        
        # Add to session token usage (assuming not cached for now)
        add_token_usage(input_tokens, output_tokens, is_cached=False)
        
    except Exception as e:
        # If token tracking fails, continue without it
        print(f"Token tracking error in call_deepseek: {e}")
        # Fallback to tiktoken estimation
        try:
            input_tokens = len(tokenizer.encode(prompt + sys_prompt))
            if n_completion == 1:
                output_tokens = len(tokenizer.encode(response.choices[0].message.content))
            else:
                output_tokens = sum(len(tokenizer.encode(choice.message.content)) for choice in response.choices)
            add_token_usage(input_tokens, output_tokens, is_cached=False)
        except Exception as e2:
            print(f"Fallback token tracking also failed: {e2}")
    
    # function to remove COT outputs in DeepSeek API calls
    splice = lambda x : re.sub(r'<think>.*?</think>', '', x, flags=re.DOTALL)
    
        # yaml.dump(response.text, outfile)

    if n_completion == 1:
        with open('deepseek_response.yml', 'w') as outfile:
            yaml.dump(splice(response.choices[0].message.content), outfile)
        return splice(response.choices[0].message.content)
    else:
        with open('deepseek_response.yml', 'w') as outfile:
            yaml.dump(splice(response.choices[0].message.content), outfile)
        return [splice(r.message.content) for r in response.choices]
        
def call_llm(prompt, sys_prompt,llm_api_selection="nvidia/nemotron-4-340b-instruct"):
    """Call the LLM API.

    Args:
        prompt: The prompt to send to the model.
        sys_prompt: The system prompt to send to the model.
        llm_api_selection: The API to use for the completion.
    """
    if llm_api_selection[:4] == "gpt-":
        return call_openai(prompt, sys_prompt, llm_api_selection)
    if llm_api_selection[:4] == "nvid":
        print("NVIDIA")
        return call_nvidia(prompt,sys_prompt, llm_api_selection)
    elif llm_api_selection[:2] == "o1" or llm_api_selection[:2] == "o3":
        return call_openai_reasoning(
            f"{prompt} \n {sys_prompt}", model=llm_api_selection
        )
    elif llm_api_selection[:8] == "deepseek":
        return call_deepseek(
            prompt, sys_prompt, llm_api_selection
            )
    elif llm_api_selection == 'gemini-1.5-flash':
        return call_google(prompt, sys_prompt, model='gemini-1.5-flash')
    elif llm_api_selection == 'gemini-2.0-flash':
        return call_google(prompt, sys_prompt, model='gemini-2.0-flash')
    elif llm_api_selection == 'gemini-1.5-pro':
        return call_google(prompt, sys_prompt, model='gemini-1.5-pro')
    elif llm_api_selection == "gemini-2.5-pro-preview-03-25":
        return call_google(prompt, sys_prompt, model="gemini-2.5-pro-preview-03-25")
    elif llm_api_selection == "gemini-2.5-pro":
        return call_google(prompt, sys_prompt, model="gemini-2.5-pro")
    elif llm_api_selection[:6] == "claude":
        return call_anthropic(prompt, sys_prompt, model=llm_api_selection)

def llm_retrieve(query, contexts, llm_api_selection):
    """Retrieve the best matched photonic components based on the query.

    Args:
        query: the query to search for.
        contexts: a list of photonic components to search from.
        llm_api_selection: the API to use for the search.
    """
    desc_ = dict(enumerate(contexts))
    desc_json = json.dumps(desc_, indent=2)
    # print(desc_json)

    sys_prompt = f"""Instructions: You are a photonic chip layout developer.
    The following JSON contains all available {len(contexts)} photonic devices, including their description and properties.
    When your are asked about a photonic component, you can only look at items listed in this JSON
    to find the best matched item.
    One important property that needs to be matched is the number of input and output ports.
    We use this notion to specify number of ports: [input]x[output]. For example a 1x2 device has one input and two output ports.
    In your answer, do not explain details of the process.
    Do not preamble your answer with quotes or other strings.
    Only output a python list of integers, with the ID of item(s) you find suitable for the given input text.

    JSON: \n\n{desc_json}\n\n
    """

    r = call_llm(query, sys_prompt, llm_api_selection)

    _indices = parse_and_validate_list(r)

    # Handle case where parsing failed
    if _indices is None:
        print(f"Warning: Failed to parse LLM response as list. Raw response: {r}")
        return []

    # remove duplicate IDs:
    seen = set()
    indices = [x for x in _indices if not (x in seen or seen.add(x))]

    return indices


class MatchedComponents(BaseModel):
    """Pydantic model for matched components.

    Args:
        match_list: The list of matched component IDs.
        match_scores: The list of match scores.
        match_comment: The match comment.
    """

    match_list: list[int]
    match_scores: list[str]
    match_comment: str


def llm_search(query, contexts):
    """Search for the best matched photonic components based on the query.

    Args:
        query: the query to search for.
        contexts: a list of photonic components to search from.
    """
    desc_ = dict(enumerate(contexts))
    desc_json = json.dumps(desc_, indent=2)

    sys_prompt = f"""You are a photonic chip layout developer.
    You have access to {len(contexts)} photonic devices/components, provided in the JSON below.
    Your task is to find the best-matched component(s) based on the described functionality and port configuration.

    Key matching criteria:
    1. Often a component is described with many specifications and modifiers.
       Identify the main component and functionality and search for a match to that.
       (e.g. a coupler with 10 nm bandwidth and with s-bend; the coupler is the main component and not the s-bend).
    2. Functionality is the highest priority.
    3. Match optical port configuration (e.g., [input]x[output] such as 1x2) when possible.
    4. If no exact match exists, prioritize functionality, then select the closest port configuration.
    5. If multiple close matches are found, rank them by the number of ports first, then by functionality closeness.
    6. If the query is ambiguous (missing function or port count details), make reasonable assumptions and provide a note in 'match_comment'.
    7. If no match is found, output the nearest match. Never output an empty list.

    For each matched item, return a qualitative score:
    - exact: Exactly matches both functionality and port configuration.
    - partial: A partial match with some differences in functionality or port configuration.
    - poor: Weak match or significantly different.

    Output the following:
    - match_list: List of matched item IDs.
    - match_scores: Corresponding qualitative scores.

    JSON of available components:

    {desc_json}
    """

    r = callgpt_pydantic(query, sys_prompt, MatchedComponents)

    if len(r.match_list) != len(r.match_scores):
        print(
            "Error: match_list and match_scores have different lengths.... trying again"
        )
        print(r.match_list, r.match_scores)
        r = callgpt_pydantic(query, sys_prompt, MatchedComponents)

    # It is also important to match the functionality of the seeking component.
    # For example, when looking for a high-speed resonant modulator all these functionalities should be matched.
    # If ports is a match but functionalities is a partial match, output the ID of found items in python list: partial_match_list.

    # print('=======')
    # print(r.match_list)
    # print(r.match_scores)
    # print(r.comment_str)
    # print('=======')

    return r


def dot_add_edges(session):
    _prompt = f"""You are an assistant to a photonic engineer.
You have two input DOT graphs:
- Graph1 has the correct definition of nodes including their ports, but the edges are missing.
- Graph2 has the correct definition of edges, but the nodes definition is incomplete.

Follow these instructions:
- add the edges from Graph2 to Graph1
- add the port numbers to the edge definitions (e.g. C1:o3 -- C2:o2;). Do not label the edges.
- Do not change the node definitions (the labels and the ports) in Graph1.
- Port are labelled as o1, o2, o3 etc, and they are ordered counter-clockwise around the rectangle node.
  For example a 2x3 node has o1 (left-bottom), o2 (left-top), o3 (right-top), o4 (right-middle), o5 (right-bottom).
- It is important that the edges don't cross. You should reason about the spatial location of ports around the
  rectangular nodes and make sure the edges are not crossing. Add you're reasoning in the comment field.
- Each port can only take one edge.
- Define only one edge between any two nodes, counting all node ports, unless explicitly stated.
- Do not connect a node to itself, unless explicitly stated.
- If there is only one node with no connections output Graph1. 
- Do not under any circumstances add additional nodes.

Do not explain or provide reasoning; only output dot code for a single valid dot graph. Do not preamble with "```dot".

INPUT graph1:
{session['p300_dot_string_draft']}

INPUT graph2:
{session['p200_preschematic']}
"""

    dot_graph_with_edges = call_llm(_prompt, "no prompt", session["p100_llm_api_selection"])
    print(dot_graph_with_edges)
    dot_graph_with_edges = re.sub(r"//.*", "", dot_graph_with_edges)  # remove comments

    return dot_graph_with_edges

def dot_add_edges_errorfunc(session):
    _prompt = f"""You are an assistant to a photonic engineer.
You have three input DOT graphs:
- Graph1 has the correct definition of nodes including their ports, but the edges are missing.
- Graph2 has the correct definition of edges, but the nodes definition is incomplete.
- Graph3 has the correct definition of nodes but a definition of edges that failed a test for crossings.

Follow these instructions:
- add the edges from Graph2 to Graph1. 
- add the port numbers to the edge definitions (e.g. C1:o3 -- C2:o2;). Do not label the edges.
- Do not change the node definitions (the labels and the ports) in Graph1.
- Port are labelled as o1, o2, o3 etc, and they are ordered counter-clockwise around the rectangle node.
  For example a 2x3 node has o1 (left-bottom), o2 (left-top), o3 (right-top), o4 (right-middle), o5 (right-bottom).
- It is important that the edges don't cross. You should reason about the spatial location of ports around the
  rectangular nodes and make sure the edges are not crossing. Add you're reasoning in the comment field.
  Refer to Graph3, which is a failed attempt.
- Each port can only take one edge.
- Define only one edge between any two nodes, counting all node ports, unless explicitly stated.
- Do not connect a node to itself, unless explicitly stated.
- If there is only one node with no connections output Graph1
- Do not under any circumstances add new nodes.

Do not explain or provide reasoning; only output dot code for a single valid dot graph. Do not preamble with "```dot".

INPUT graph1:
{session['p300_dot_string_draft']}

INPUT graph2:
{session['p200_preschematic']}

INPUT graph3:
{session['p300_dot_string']}
"""

    dot_graph_with_edges = call_llm(_prompt, "no prompt", session["p100_llm_api_selection"])
    print(dot_graph_with_edges)
    dot_graph_with_edges = re.sub(r"//.*", "", dot_graph_with_edges)  # remove comments

    return dot_graph_with_edges


def dot_add_edges_templates(session):
    """Add edges to a DOT graph.

    Args:
        session: The session to add edges to.
    """
    _prompt = str(session["p300_circuit_dsl"]["edges"])

    _prompt += "\nDOT Graph:\n" + session["p300_dot_string"]

    yaml_edges = call_llm(
        _prompt, prompts["edges_yaml_to_dot"], session["p100_llm_api_selection"]
    )

    yaml_data = yaml.safe_load(yaml_edges)
    edges_list = yaml_data["edges"]

    dot_graph_lines = session["p300_dot_string"].strip().split("\n")
    closing_bracket_index = dot_graph_lines.index("}")

    # Insert the edges just before the closing bracket
    # Handle both list and string formats
    if isinstance(edges_list, list):
        for edge in edges_list:
            dot_graph_lines.insert(closing_bracket_index, "  " + edge)
    else:
        # Handle string format (backward compatibility)
        for edge in edges_list.strip().split("\n"):
            dot_graph_lines.insert(closing_bracket_index, "  " + edge)

    # Join the lines back into a single string
    dot_graph_with_edges = "\n".join(dot_graph_lines)

    return dot_graph_with_edges


def dot_verify(session):
    """Verify a DOT graph.

    Args:
        session: The session to verify.
    """
    dot_updated = call_llm(
        session.p300_dot_string, prompts["dot_verify"], session.p100_llm_api_selection
    )

    # dot_updated = dot_updated.strip("```dot")
    dot_updated = dot_updated.replace("```dot", "").strip()

    return dot_updated


def netlist_cleanup(yaml_string):
    """Clean up a YAML string.

    Args:
        yaml_string: The YAML string to clean up.
    """
    updated_netlist = call_llm(yaml_string, prompts["yaml_syntax_cleaner"])

    return updated_netlist


class PromptClass(BaseModel):
    """Pydantic model for prompt classification.

    Args:
        category_id: The category ID.
        response: The response.
    """

    category_id: int
    response: str


def intent_classification(input_prompt):
    """Classify the input prompt into one of the categories.

    Args:
        input_prompt: The input prompt to classify.
    """
    sys_prompt = """You are an assistant to a photonic engineer.
Your task is to classify the input text into one of these categories (category_id):
category 1: A description of one or many photonic components/devices potentially forming a circuit.
Or a prompt to design/layout a photonic circuit/GDS.
category 2: A generic question about integrated photonic, or photonic devices/components. Or a prompt to run any type of photonic simulation.
category 3: Not relevant to integrated photonics.
    If the category is 2 or 3, provide a response to the effect: I am only able to help desiging and layouting integrated photonics circuits.
    """

    r = callgpt_pydantic(input_prompt, sys_prompt, PromptClass)
    return r


class InputClarity(BaseModel):
    """Pydantic model for input clarity.

    Args:
        input_clarity: The clarity of the input.
        explain_ambiguity: The explanation of the ambiguity.
    """

    input_clarity: bool
    explain_ambiguity: str


def verify_input_clarity(input_prompt):
    """Verify the clarity of the input prompt.

    Args:
        input_prompt: The input prompt to verify.
    """
    sys_prompt = """You are an assistant to a photonic engineer.
The input is a description of photonic component(s) to be used in a photonic circuit.

- Is it clear from the input what photonic component(s) are being described?
- It is not required that each components have a detail specification. But if there is a specification, is it clear to which component it belongs?
- This is relevant only if more than one component is mentioned:
  Is there at least a hint about how to lay the components out or connect them?
  A sufficient info might be a simple connect A to B, or put A and B in series, or in parallel, etc.
  An insufficient info might be completely or partially missing info about how to the arrangement between all or some components.

If the answer to ALL of these questions is YES, set input_clarity to True.
Otherwise, set input_clarity to False and provide a brief explanation of the ambiguity in explain_ambiguity."""

    r = callgpt_pydantic(input_prompt, sys_prompt, InputClarity)
    return r.model_dump()


class InputEntities(BaseModel):
    """Pydantic model for input entities.

    Args:
        title: The title of the input.
        components_list: The list of components in the input.
        circuit_instructions: The instructions for the circuit.
        brief_summary: The brief summary of the input.
    """

    title: str
    components_list: list[str]
    circuit_instructions: str
    brief_summary: str


def entity_extraction(input_prompt):
    """Extract entities from the input prompt.

    Args:
        input_prompt: The input prompt to extract entities from.
    """
    sp_normal = """You are an assistant to a photonic engineer.
Your task is to extract specific information from the input text and present it in the following structured format:

title: A concise title describing the function of the photonic circuit based on the input text.

components_list: Extract a list of components mentioned in the input text.
This list must contain at least one component. For each component:
- Include all provided specifications and descriptions, if any.
- Include the number of optical input and output ports in the format [input]x[output] (e.g., 1x2),
  but only if explicitly stated. Do not assume numbers that are not provided.
- Do not list specifications or descriptive modifiers as separate components. For example if a phase shifter or heater is integrated into a photonic modulator (MZI) only list the modulator.
- If multiple copies of the same component are described, list each one explicitly.
- If it is implied that a component is used more than once, list each instance separately.

circuit_instructions: Extract any instructions from the input text about how the components
  should be connected or used in the circuit. If none are provided, set this to an empty string.

brief_summary: Provide a summary of the input text in less than 150 words.

Note: All extracted information must be exclusively derived from the input text.
"""

    sp_paper = """You are an assistant for a photonic engineer.
Your task is to extract specific information from the input text and attached figure and present it in the following structured format:

components_list: extract a list of on-chip photonic components, following these guidelines:

(1) For each component, include all provided specifications and descriptions, if any.

(2) For each component, include the number of optical input and output ports in the format [input]x[output] (e.g., 1x2). Make an educated guess from the function if not explicitly stated. For example if the component is in an add-drop configuration it should be 2x2.

(3) Do not list specifications or descriptive modifiers as separate components. For example if a phase shifter or heater is integrated into a photonic modulator only list the modulator.

(4) If multiple copies of the same component are described, explictly state the number of copies of the component. i.e. 4 Germanium photodetectors

(5) Do not group components into "Arrays". Separate arrays into individual components.

(6) Exclude electronic components (e.g., oscilloscope, transimpedance amplifier, DAC, RF source)
and off-chip components (e.g., fiber, free-space lenses/lasers, EDFA).

(7) If the text does not contain any on-chip photonic components, set this field to an empty list.

(8) compose this list in YAML. have each instance labelled as C1, C2, .... Add any
provided specification to each component. make sure multiple copies of components are created as new instances.

circuit_instructions: Extract any instructions from the input text about how the components should be connected or used in the circuit. If none are provided, set this to an empty string.

brief_summary: Provide a summary of the input text in less than 150 words."""

    if len(input_prompt) > 1000:
        sys_prompt = sp_paper
    else:
        sys_prompt = sp_normal

    r = callgpt_pydantic(input_prompt, sys_prompt, InputEntities)

    # if len(r.components_list) == 0:
    #     r = callgpt_pydantic(input_prompt, sys_prompt, InputEntities)

    return r.model_dump()


class PaperEntities1(BaseModel):
    """Pydantic model for paper entities.

    Args:
        topic_photonic: Whether the article is about integrated photonic circuits.
        single_article: Whether the article is a single academic article.
        components_list: The list of components in the article.
        circuit_complete: Whether the article contains enough information to describe a photonic circuit.
    """

    topic_photonic: bool
    single_article: bool
    components_list: list[str]
    circuit_complete: bool


def papers_entity_extraction(input_article):
    """Extract entities from the input article.

    Args:
        input_article: The input article to extract entities from.
    """
    sys_prompt = """You are an assistant for a photonic engineer.

topic_photonic: Determine if the article is about integrated photonic circuits.

single_article: Confirm if this is a single academic article (not a dissertation or a collection of papers).

components_list: If both topic_photonic and single_article are True, extract a list of on-chip photonic components.
Follow these guidelines:
- Exclude electronic components (e.g., oscilloscope, transimpedance amplifier, DAC, RF source)
  and off-chip components (e.g., fiber, free-space lenses/lasers, EDFA).
- For each component, include specifications and descriptions if available.
- Extract the number of optical input and output ports for each component, if specified. Do not infer port counts if not explicitly stated.
- Avoid parsing descriptive modifiers or specifications as separate components. 
- If multiple instances of the same component are mentioned, list each explicitly.
- If the article does not contain any on-chip photonic components, set this field to an empty list.

circuit_complete: Assess if there is enough information to describe how the listed components are
interconnected to form a complete photonic circuit.
    """

    r = callgpt_pydantic(input_article, sys_prompt, PaperEntities1)

    return r.model_dump()


def preschematic(pretemplate, llm_api_selection):
    """Generate a preschematic from the pretemplate.

    Args:
        pretemplate: The pretemplate to generate a preschematic from.
    """
    dot_ = call_llm(yaml.dump(pretemplate), prompts["dot_simple"], llm_api_selection)
    return dot_


try:
    with open(PATH.templates) as file:
        templates_dict = yaml.safe_load(file)
        templates_str = yaml.dump(templates_dict, default_flow_style=False)
except FileNotFoundError:
    print(f"No {PATH.templates} file found.")
    pass

templates_titles = {key: value["doc"]["title"] for key, value in templates_dict.items()}
templates_titles_str = yaml.dump(templates_titles, sort_keys=False)


def parse_user_specs(session):
    """Parse user specifications from the input text.

    Args:
        session: The session to parse specifications from.
    """
    sys_prompt = f"You are an assistant to a photonic circuit engineer.\
            Your task is to parse specifications from the input text according to the template.\
            This is the template: {templates_dict[session['p200_selected_template']]['properties']['specs']}.\
            This is the input text: {session['p200_user_specs']}.\
            You should parse the input text into the template and return that object as a parsable python dict with the original syntax.\
            If user input is not compatible with the template (for example if the list size, the datatype, or the range is not compatible) answer with \
            an error message explaining the issue e.g. {{'Error': 'Four wavelengths are required but only two are provided.'}}.\
            Do no add any additional text or preambles like quotes etc."

    parsed_user_specs = call_llm(
        session["p200_user_specs"], sys_prompt, session["p100_llm_api_selection"]
    )

    # Check if the LLM call returned an error or is None/empty
    if parsed_user_specs is None:
        return {"Error": "No response received from LLM API"}
    
    if not parsed_user_specs.strip():
        return {"Error": "Empty response received from LLM API"}
    
    if parsed_user_specs.startswith("Error:"):
        return {"Error": parsed_user_specs}

    # Clean up the response - remove any markdown formatting and language identifiers
    parsed_user_specs = parsed_user_specs.strip()
    
    # Remove markdown code blocks
    if parsed_user_specs.startswith('```'):
        lines = parsed_user_specs.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines[-1].startswith('```'):
            lines = lines[:-1]
        parsed_user_specs = '\n'.join(lines).strip()
    
    # Remove language identifiers at the beginning (like "python", "yaml", etc.)
    lines = parsed_user_specs.split('\n')
    if lines and lines[0].strip().lower() in ['python', 'yaml', 'json']:
        lines = lines[1:]
        parsed_user_specs = '\n'.join(lines).strip()

    # Try to parse as YAML first, if that fails, try to parse as Python dict
    try:
        # Try YAML parsing first
        return yaml.safe_load(parsed_user_specs)
    except yaml.YAMLError as yaml_error:
        try:
            # If YAML fails, try to parse as Python dict using ast.literal_eval
            import ast
            return ast.literal_eval(parsed_user_specs)
        except (ValueError, SyntaxError) as eval_error:
            # If both fail, try to fix common issues and retry
            try:
                # Try to fix common Python dict formatting issues
                fixed_specs = parsed_user_specs.replace("'", '"')  # Replace single quotes with double quotes
                return yaml.safe_load(fixed_specs)
            except yaml.YAMLError:
                # If all parsing attempts fail, return the original string with an error indicator
                return {"Error": f"Failed to parse LLM response. YAML error: {yaml_error}. Eval error: {eval_error}. Raw response: {parsed_user_specs}"}


def apply_settings(session, llm_api_selection):
    """Apply settings to the circuit DSL.

    Args:
        session: The session to apply settings to.
    """
    y1 = yaml.dump(session.p200_pretemplate_copy["components_list"])
    y2 = yaml.dump(session["p300_circuit_dsl"]["nodes"])
    llm_input = f"INPUT DESCRIPTION: \n{y1} \n\nNETLIST: \n{y2}"
    
    updated_y2 = call_llm(llm_input, prompts["absorb_settings"], llm_api_selection)
    # Pre-process the YAML: wrap unquoted 'comment' values in quotes.
    updated_y2_fixed = re.sub(
        r'^( *comment:\s*)(.+)$', 
        lambda m: m.group(1) + '"' + m.group(2).strip().replace("-", "\n") + '"', 
        updated_y2, 
        flags=re.MULTILINE
    )
    updated_y2 = yaml.safe_load(updated_y2)
    if "comment" in updated_y2:
        del updated_y2["comment"]
    for k, v in updated_y2.items():
        if "comment" in v:
            del updated_y2[k]["comment"]

    session["p300_circuit_dsl"]["nodes"] = updated_y2

    # updated_netlist = verifiers.verify_and_filter_netlist(updated_netlist)
    return session["p300_circuit_dsl"]
