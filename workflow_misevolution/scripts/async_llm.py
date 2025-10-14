# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang
# @Desc    : 

from openai import AsyncOpenAI
from scripts.formatter import BaseFormatter, FormatError

import yaml
from pathlib import Path
from typing import Dict, Optional, Any
import aiohttp
import json
import asyncio
import requests
from requests.auth import HTTPBasicAuth
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# disable SSL Warnings
urllib3.disable_warnings(InsecureRequestWarning)

# 导入日志记录器（尝试导入，如果不存在则跳过）
try:
    from RedCode.evaluation.RedCode_Gen.llm_logger import log_llm_call
    _HAS_LOGGER = True
except ImportError:
    _HAS_LOGGER = False
    def log_llm_call(*args, **kwargs):
        pass  # 空函数，不做任何操作

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://oneapi.deepwisdom.ai/v1")
        self.top_p = config.get("top_p", 1)
        self.api_type = config.get("api_type", "openai")
        self.secret_key = config.get("secret_key", None)  # 用于HTTP Basic Auth

class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            # Look for the config file in common locations
            config_paths = [
                Path("config/config2.yaml"),
                Path("config2.yaml"),
                Path("./config/config2.yaml")
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if config_file is None:
                raise FileNotFoundError("No default configuration file found in the expected locations")
            
            # Load the YAML file
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Your YAML has a 'models' top-level key that contains the model configs
            if 'models' in config_data:
                config_data = config_data['models']
                
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        
        # Create a config dictionary with the expected keys for LLMConfig
        if llm_name == "qwen2.5":
            actual_model = "Qwen2.5-72B-Instruct"
        else:
            actual_model = llm_name
        
        llm_config = {
            "model": actual_model,  # Use the actual model name
            "temperature": config.get("temperature", 1),
            "key": config.get("api_key"),  # Map api_key to key
            "base_url": config.get("base_url", "https://oneapi.deepwisdom.ai/v1"),
            "top_p": config.get("top_p", 1),  # Add top_p parameter
            "api_type": config.get("api_type", "openai"),
            "secret_key": config.get("secret_key", None)
        }
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class ModelPricing:
    """Pricing information for different models in USD per 1K tokens"""
    PRICES = {
        # GPT-4o models
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "o3":{"input":0.003, "output":0.015},
        "o3-mini": {"input": 0.0011, "output": 0.0025},
    }
    
    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        
        # Try to find a partial match (e.g., if model name contains version numbers)
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]
        
        # Return default pricing if no match found
        return 0

class CustomAsyncAPIClient:
    """自定义异步API客户端，支持HTTP Basic Auth，基于requests的异步包装"""
    
    def __init__(self, base_url: str, api_key: str, secret_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.secret_key = secret_key
        
    async def chat_completions_create(self, **kwargs):
        """异步调用chat completions API"""
        import concurrent.futures
        import threading
        
        def sync_request():
            url = f"{self.base_url}/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": kwargs.get("model"),
                "messages": kwargs.get("messages"),
                "temperature": kwargs.get("temperature", 0),
                "max_tokens": kwargs.get("max_tokens", 12800),
                "top_p": kwargs.get("top_p", 1.0),
                "stream": False
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=180,
                # auth=HTTPBasicAuth(self.api_key, self.secret_key),
                verify=False  # 禁用SSL验证
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 创建模拟OpenAI响应格式的对象
                class MockResponse:
                    def __init__(self, data):
                        self.choices = [MockChoice(data['choices'][0])]
                        self.usage = MockUsage(data.get('usage', {}))
                
                class MockChoice:
                    def __init__(self, choice_data):
                        self.message = MockMessage(choice_data['message'])
                
                class MockMessage:
                    def __init__(self, message_data):
                        self.content = message_data['content']
                
                class MockUsage:
                    def __init__(self, usage_data):
                        self.prompt_tokens = usage_data.get('prompt_tokens', 0)
                        self.completion_tokens = usage_data.get('completion_tokens', 0)
                        self.total_tokens = usage_data.get('total_tokens', 0)
                
                return MockResponse(result)
            else:
                raise Exception(f"API调用失败，状态码: {response.status_code}, 错误信息: {response.text}")
        
        # 在线程池中运行同步请求
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, sync_request)

class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }

class AsyncLLM:
    def __init__(self, config, system_msg:str = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        
        # At this point, config should be an LLMConfig instance
        self.config = config
        
        # 根据API类型选择不同的客户端
        if self.config.api_type == "custom" and self.config.secret_key:
            self.aclient = CustomAsyncAPIClient(
                base_url=self.config.base_url,
                api_key=self.config.key,
                secret_key=self.config.secret_key
            )
            self.is_custom_api = True
        else:
            self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
            self.is_custom_api = False
            
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        
    async def __call__(self, prompt):
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        message.append({"role": "user", "content": prompt})

        if self.is_custom_api:
            response = await self.aclient.chat_completions_create(
                model=self.config.model,
                messages=message,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        else:
            response = await self.aclient.chat.completions.create(
                model=self.config.model,
                messages=message,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

        # Extract token usage from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # Track token usage and calculate cost
        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens
        )
        
        ret = response.choices[0].message.content
        print(ret)
        
        # You can optionally print token usage information
        print(f"Token usage: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
        print(f"Cost: ${usage_record['total_cost']:.6f} (${usage_record['input_cost']:.6f} for input, ${usage_record['output_cost']:.6f} for output)")
        
        # 记录LLM调用（如果启用了日志功能）
        if _HAS_LOGGER:
            full_query = ""
            if self.sys_msg:
                full_query += f"[SYSTEM] {self.sys_msg}\n\n"
            full_query += f"[USER] {prompt}"
            
            context = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": usage_record['total_cost'],
                "api_type": self.config.api_type
            }
            
            log_llm_call(
                model=self.config.model,
                query=full_query,
                response=ret,
                context=context
            )
        
        return ret
    
    async def call_with_format(self, prompt: str, formatter: BaseFormatter):
        """
        Call the LLM with a prompt and format the response using the provided formatter
        
        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response
            
        Returns:
            The formatted response data
            
        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM
        response = await self.__call__(formatted_prompt)
        
        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)
        
        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")
        
        return parsed_data
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()    
    

def create_llm_instance(llm_config):
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")