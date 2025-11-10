"""
Data Loader Utilities for RL Training
Supports various dataset formats and preprocessing
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import random


# ============================================================================
# Base Dataset Classes
# ============================================================================

class RLPromptDataset(Dataset):
    """Dataset for RL training with prompts"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        format: str = "jsonl"  # "jsonl", "json", "txt"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path, format)
    
    def _load_data(self, data_path: str, format: str) -> List[Dict]:
        """Load data from various formats"""
        path = Path(data_path)
        
        if format == "jsonl":
            with open(path, 'r') as f:
                data = [json.loads(line) for line in f]
        
        elif format == "json":
            with open(path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        
        elif format == "txt":
            with open(path, 'r') as f:
                lines = f.readlines()
                data = [{"prompt": line.strip()} for line in lines if line.strip()]
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt', item.get('text', item.get('instruction', '')))
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'prompt': prompt,
            'metadata': {k: v for k, v in item.items() if k not in ['prompt', 'text', 'instruction']}
        }


class ConversationDataset(Dataset):
    """Dataset for multi-turn conversation RL training"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        system_prompt: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt or "You are a helpful assistant."
        
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation into a single string"""
        formatted = f"System: {self.system_prompt}\n\n"
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted += f"{role.capitalize()}: {content}\n"
        
        return formatted
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get('messages', item.get('conversation', []))
        
        # Format conversation
        text = self.format_conversation(messages)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text,
            'num_turns': len(messages)
        }


class PromptResponseDataset(Dataset):
    """Dataset with prompt-response pairs (for supervised fine-tuning or preference learning)"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        include_response: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_response = include_response
        
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt', item.get('question', ''))
        response = item.get('response', item.get('answer', ''))
        
        # Concatenate prompt and response if needed
        if self.include_response:
            text = f"{prompt}\n\nResponse: {response}"
        else:
            text = prompt
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'prompt': prompt
        }
        
        if self.include_response:
            result['response'] = response
        
        return result


# ============================================================================
# Specialized Dataset Loaders
# ============================================================================

class MathDataset(Dataset):
    """Dataset for math problems (GSM8K-style)"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        add_reasoning_prompt: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_reasoning_prompt = add_reasoning_prompt
        
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get('question', item.get('problem', ''))
        answer = item.get('answer', '')
        
        # Add reasoning prompt if specified
        if self.add_reasoning_prompt:
            prompt = f"Solve this math problem step by step:\n{question}\n\nSolution:"
        else:
            prompt = question
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'prompt': prompt,
            'question': question,
            'answer': answer
        }


class CodeDataset(Dataset):
    """Dataset for code generation tasks"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        language: str = "python"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language = language
        
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get('instruction', item.get('prompt', ''))
        code = item.get('code', item.get('solution', ''))
        
        # Format prompt
        prompt = f"Write {self.language} code to:\n{instruction}\n\nCode:\n```{self.language}\n"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'prompt': prompt,
            'instruction': instruction,
            'code': code
        }


# ============================================================================
# Data Collators
# ============================================================================

class RLCollator:
    """Custom collator for RL training"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract fields
        prompts = [item['prompt'] for item in batch]
        
        # Dynamic padding
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        # Stack tensors
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt': prompts,
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_dataloader(
    dataset_type: str,
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    max_length: int = 512,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Factory function to create appropriate dataloader
    
    Args:
        dataset_type: Type of dataset ("prompt", "conversation", "prompt_response", "math", "code")
        data_path: Path to data file
        tokenizer: Tokenizer
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for specific dataset types
    
    Returns:
        DataLoader instance
    """
    dataset_map = {
        "prompt": RLPromptDataset,
        "conversation": ConversationDataset,
        "prompt_response": PromptResponseDataset,
        "math": MathDataset,
        "code": CodeDataset,
    }
    
    if dataset_type not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from {list(dataset_map.keys())}")
    
    dataset_class = dataset_map[dataset_type]
    dataset = dataset_class(data_path, tokenizer, max_length=max_length, **kwargs)
    
    collator = RLCollator(tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator
    )
    
    return dataloader


def load_multiple_datasets(
    dataset_configs: List[Dict],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    shuffle: bool = True
) -> DataLoader:
    """
    Load and combine multiple datasets
    
    Args:
        dataset_configs: List of dataset configurations
        tokenizer: Tokenizer
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        Combined DataLoader
    """
    from torch.utils.data import ConcatDataset
    
    datasets = []
    for config in dataset_configs:
        dataset_type = config.pop('type')
        data_path = config.pop('path')
        max_length = config.pop('max_length', 512)
        
        dataset_class = {
            "prompt": RLPromptDataset,
            "conversation": ConversationDataset,
            "prompt_response": PromptResponseDataset,
            "math": MathDataset,
            "code": CodeDataset,
        }[dataset_type]
        
        dataset = dataset_class(data_path, tokenizer, max_length=max_length, **config)
        datasets.append(dataset)
    
    combined_dataset = ConcatDataset(datasets)
    collator = RLCollator(tokenizer)
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )
    
    return dataloader


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Example: Load a simple prompt dataset
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    dataloader = create_dataloader(
        dataset_type="prompt",
        data_path="data.jsonl",
        tokenizer=tokenizer,
        batch_size=4,
        max_length=512
    )
    
    # Test iteration
    for batch in dataloader:
        print(f"Batch size: {batch['input_ids'].shape}")
        print(f"Prompts: {batch['prompt']}")
        break
