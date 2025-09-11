"""
Unified API wrapper for multiple LLM providers with retry logic and error handling.

This module provides a standardized interface for OpenAI, Google Gemini, and Anthropic Claude APIs
with automatic retry mechanisms, JSON parsing validation, and conditional client initialization.

Features:
- Conditional client initialization based on available API keys
- Automatic retry with exponential backoff for network errors
- JSON response validation and error tolerance
- Unified response format across different providers
- Both synchronous and asynchronous API calls
"""

import asyncio
import json
import os
from typing import Dict, List, Tuple, Any, Union

import anthropic
import openai
from google import genai
from google.genai import types
from openai import OpenAIError as APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_exception_type,
    AsyncRetrying,
)

from errors import OutputPasingError
from utils import WPrinter

# Configuration constants
STOP_AFTER_ATTEMPT = 4
RETRY_WAIT_INITIAL = 3
RETRY_WAIT_FINAL = 5

# Initialize global printer
printer = WPrinter(verbose=True)

# Initialize clients conditionally based on available API keys
anthropic_client = None
gemini_client = None

# Check for Anthropic API key
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
if anthropic_key:
    anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
    print("✓ Anthropic client initialized successfully")
else:
    print("⚠️  Warning: ANTHROPIC_API_KEY not found. Claude models unavailable.")

# Check for Gemini API key
gemini_key = os.environ.get("GEMINI_API_KEY")
if gemini_key:
    gemini_client = genai.Client(api_key=gemini_key)
    print("✓ Gemini client initialized successfully")
else:
    print("⚠️  Warning: GEMINI_API_KEY not found. Gemini models unavailable.")

# Check for OpenAI API key
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key:
    print("✓ OpenAI API key found")
else:
    print("⚠️  Warning: OPENAI_API_KEY not found. OpenAI models unavailable.")


@retry(
    stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
    wait=wait_chain(
        *[wait_fixed(RETRY_WAIT_INITIAL) for _ in range(2)]
        + [wait_fixed(RETRY_WAIT_FINAL) for _ in range(1)]
    ),
    retry=retry_if_exception_type((OutputPasingError, APIError, json.JSONDecodeError)),
)
def completion_with_backoff(**kwargs) -> Dict[str, Any]:
    """
    OpenAI API wrapper with retry logic.

    Args:
        **kwargs: Arguments to pass to OpenAI API including:
            - model: Model name
            - messages: List of message objects
            - json_parsing_check: Boolean for JSON validation
            - verbose: Boolean for debug output

    Returns:
        Dict containing API response

    Raises:
        ValueError: If OpenAI API key is not configured
        APIError: If API call fails after retries
    """
    global printer

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )

    verbose = kwargs.get("verbose", False)
    printer.set_verbose(verbose)
    printer.wprint("Calling OpenAI API")

    checking_json = kwargs.get("json_parsing_check", False)
    if checking_json:
        kwargs["response_format"] = {"type": "json_object"}

    # Clean up kwargs
    kwargs.pop("json_parsing_check", None)
    kwargs.pop("verbose", None)

    try:
        response = openai.ChatCompletion.create(**kwargs)
        printer.wprint(response)

        if checking_json:
            printer.wprint("Validating JSON response")
            _validate_json_response(response)

        return response

    except APIError as e:
        if (
            isinstance(e, APIError)
            and hasattr(e, "http_status")
            and e.http_status == 500
        ):
            printer.wprint(f"500 Error encountered: {e}. Retrying...")
        raise


def completion_with_backoff_gemini(**kwargs) -> Dict[str, Any]:
    """
    Google Gemini API wrapper with retry logic.

    Args:
        **kwargs: Arguments including model, system_instruction, user_text, etc.

    Returns:
        Dict containing standardized API response

    Raises:
        ValueError: If Gemini client is not initialized
    """
    global printer, gemini_client

    if gemini_client is None:
        raise ValueError(
            "Gemini client not initialized. Please set GEMINI_API_KEY environment variable."
        )

    verbose = kwargs.get("verbose", False)
    printer.set_verbose(verbose)
    printer.wprint("Calling Gemini API")

    # Extract parameters
    model = kwargs["model"]
    system_instruction = kwargs.get("system_instruction")
    user_text = kwargs["user_text"]
    checking_json = kwargs.get("json_parsing_check", False)

    # Configure generation
    gen_config = {"candidate_count": kwargs.get("n", 1)}

    if system_instruction:
        gen_config["system_instruction"] = system_instruction

    if checking_json:
        gen_config["response_mime_type"] = "application/json"

    try:
        response = gemini_client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(**gen_config),
            contents=[user_text],
        )

        # Convert to standardized format
        standardized_response = {"choices": []}
        for choice in response.candidates:
            content = choice.content.parts[0].text.replace("json", " ").replace("`", "")
            standardized_response["choices"].append(
                {"message": {"role": "assistant", "content": content}}
            )

        if checking_json:
            printer.wprint("Validating JSON response")
            _validate_json_response(standardized_response)

        return standardized_response

    except Exception as e:
        printer.wprint(f"Error encountered: {e}. Retrying...")
        raise


def completion_with_backoff_claude(**kwargs) -> Dict[str, Any]:
    """
    Anthropic Claude API wrapper with retry logic.

    Args:
        **kwargs: Arguments including model, messages, etc.

    Returns:
        Dict containing standardized API response

    Raises:
        ValueError: If Anthropic client is not initialized
    """
    global printer, anthropic_client

    if anthropic_client is None:
        raise ValueError(
            "Anthropic client not initialized. Please set ANTHROPIC_API_KEY environment variable."
        )

    verbose = kwargs.get("verbose", False)
    printer.set_verbose(verbose)
    printer.wprint("Calling Claude API")

    messages = kwargs["messages"]
    model = kwargs["model"]
    checking_json = kwargs.get("json_parsing_check", False)

    try:
        response = anthropic_client.messages.create(
            model=model, max_tokens=1024, messages=messages
        )

        # Convert to standardized format
        standardized_response = {"choices": []}
        for choice in response.content:
            standardized_response["choices"].append(
                {"message": {"role": "assistant", "content": choice.text}}
            )

        if checking_json:
            printer.wprint("Validating JSON response")
            _validate_json_response(standardized_response)

        return standardized_response

    except Exception as e:
        printer.wprint(f"Error encountered: {e}. Retrying...")
        raise


async def async_openai_call(**kwargs) -> Dict[str, Any]:
    """
    Asynchronous OpenAI API call wrapper.

    Args:
        **kwargs: Arguments to pass to OpenAI API

    Returns:
        Dict containing API response

    Raises:
        ValueError: If OpenAI API key is not configured
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, lambda: openai.ChatCompletion.create(**kwargs)
    )
    return response


@retry(
    stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
    wait=wait_chain(
        *[wait_fixed(RETRY_WAIT_INITIAL) for _ in range(2)]
        + [wait_fixed(RETRY_WAIT_FINAL) for _ in range(1)]
    ),
    retry=retry_if_exception_type((OutputPasingError, APIError, json.JSONDecodeError)),
    reraise=True,
)
async def completion_with_backoff_async(
    json_error_tolerance: int = 1, **kwargs
) -> Union[Dict[str, Any], Any]:
    """
    Asynchronous OpenAI API wrapper with error tolerance.

    Args:
        json_error_tolerance: Number of JSON parsing errors to tolerate
        **kwargs: Arguments to pass to OpenAI API

    Returns:
        API response or processed results if using error tolerance

    Raises:
        ValueError: If OpenAI API key is not configured
    """
    global printer

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )

    json_parsing_check = kwargs.pop("json_parsing_check", False)
    if json_parsing_check:
        kwargs["response_format"] = {"type": "json_object"}

    verbose = kwargs.get("verbose", False)
    printer.set_verbose(verbose)

    try:
        printer.wprint("Calling OpenAI API")
        response = await async_openai_call(**kwargs)

        if json_parsing_check:
            printer.wprint("Validating JSON response")
            error_cnt, processed_results, _ = process_errors(response["choices"])

            if error_cnt > json_error_tolerance:
                printer.wprint(f"Errors encountered: {error_cnt} times. Retrying...")
                raise OutputPasingError("JSON parsing error threshold exceeded")

            if error_cnt > 0:
                printer.wprint("Tolerating error cases")
                return processed_results

        return response

    except APIError as e:
        if hasattr(e, "http_status") and e.http_status == 500:
            printer.wprint(f"500 Error encountered: {e}. Retrying...")
        raise
    except json.JSONDecodeError as e:
        printer.wprint(f"Error decoding JSON: {e}. Retrying...")
        raise
    except Exception as e:
        printer.wprint(f"response: {response}")
        raise OutputPasingError(f"Error parsing response: {e}. Retrying...")


async def completion_with_backoff_async_2(
    json_error_tolerance: int = 1, **kwargs
) -> Union[Dict[str, Any], Any]:
    """
    Alternative asynchronous OpenAI API wrapper using AsyncRetrying.

    Args:
        json_error_tolerance: Number of JSON parsing errors to tolerate
        **kwargs: Arguments to pass to OpenAI API

    Returns:
        API response or processed results if using error tolerance
    """
    global printer

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )

    json_parsing_check = kwargs.pop("json_parsing_check", False)
    if json_parsing_check:
        kwargs["response_format"] = {"type": "json_object"}

    verbose = kwargs.get("verbose", False)
    printer.set_verbose(verbose)

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_chain(
            *[wait_fixed(2) for _ in range(3)]
            + [wait_fixed(RETRY_WAIT_FINAL) for _ in range(1)]
        ),
        retry=retry_if_exception_type(
            (OutputPasingError, APIError, json.JSONDecodeError)
        ),
        reraise=True,
    ):
        with attempt:
            try:
                printer.wprint("Calling OpenAI API")
                response = openai.ChatCompletion.create(**kwargs)

                if json_parsing_check:
                    printer.wprint("Validating JSON response")
                    error_cnt, processed_results, _ = process_errors(
                        response["choices"]
                    )

                    if error_cnt > json_error_tolerance:
                        printer.wprint(
                            f"Errors encountered: {error_cnt} times. Retrying..."
                        )
                        raise OutputPasingError("JSON parsing error threshold exceeded")

                    if error_cnt > 0:
                        printer.wprint("Tolerating error cases")
                        return processed_results

                return response

            except APIError as e:
                if hasattr(e, "http_status") and e.http_status == 500:
                    printer.wprint(f"500 Error encountered: {e}. Retrying...")
                raise
            except (json.JSONDecodeError, OutputPasingError) as e:
                printer.wprint(f"Response content: {response}")
                printer.wprint(f"Exception encountered: {e}. Retrying...")
                raise


async def call_multiple_apis(api_calls: List[Dict[str, Any]]) -> List[Any]:
    """
    Execute multiple API calls concurrently.

    Args:
        api_calls: List of dictionaries containing API call parameters

    Returns:
        List of results from API calls (exceptions are returned as values)
    """

    async def safe_call(params: Dict[str, Any]) -> Any:
        try:
            return await completion_with_backoff_async(**params)
        except Exception as e:
            return e

    tasks = [safe_call(params) for params in api_calls]
    results = await asyncio.gather(*tasks)
    return results


def process_errors(choices: List[Dict[str, Any]]) -> Tuple[int, Dict[str, List], List]:
    """
    Process API response choices and handle JSON parsing errors.

    Args:
        choices: List of choice objects from API response

    Returns:
        Tuple of (error_count, processed_results, error_cases)
    """
    error_count = 0
    results = []
    processed_results = {"choices": []}
    error_cases = []

    for choice in choices:
        try:
            # Attempt to parse JSON content
            parsed_content = json.loads(choice["message"]["content"])
            processed_results["choices"].append(choice)
            results.append(parsed_content)
        except (json.JSONDecodeError, OutputPasingError):
            error_count += 1
            error_cases.append(choice)
            processed_results["choices"].append(None)

    return error_count, processed_results, error_cases


def _validate_json_response(response: Dict[str, Any]) -> None:
    """
    Validate that API response contains valid JSON content.

    Args:
        response: API response dictionary

    Raises:
        json.JSONDecodeError: If JSON parsing fails
        OutputPasingError: If custom parsing validation fails
    """
    try:
        choices = response["choices"]
        for choice in choices:
            json.loads(choice["message"]["content"])
    except json.JSONDecodeError as e:
        printer.wprint(f"Error decoding JSON: {e}")
        printer.wprint(f"Response content: {response}")
        raise
    except OutputPasingError as e:
        printer.wprint(f"Error in parsing response: {e}")
        printer.wprint(f"response: {response}")
        raise


if __name__ == "__main__":
    # Example usage for testing multiple API calls
    api_calls = [
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "verbose": True,
            "json_parsing_check": False,
        },
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Tell me a joke."}],
            "verbose": True,
            "json_parsing_check": False,
        },
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "What is the meaning of life?"}],
            "verbose": True,
            "json_parsing_check": False,
        },
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Write a poem about the ocean."}],
            "verbose": True,
            "json_parsing_check": False,
        },
        {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Explain quantum physics in simple terms."}
            ],
            "verbose": True,
            "json_parsing_check": False,
        },
    ]

    # Run async example
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(call_multiple_apis(api_calls))
    print(results)
