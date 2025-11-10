"""
Evaluation Script for RL-trained LLMs
Supports multiple evaluation tasks and metrics
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re


# ============================================================================
# Evaluation Tasks
# ============================================================================

class EvaluationTask:
    """Base class for evaluation tasks"""
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, model, tokenizer, data_path: str) -> Dict[str, float]:
        """Evaluate model on this task"""
        raise NotImplementedError


class GenerationTask(EvaluationTask):
    """General text generation evaluation"""
    
    def __init__(
        self,
        name: str = "generation",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        super().__init__(name)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    def evaluate(self, model, tokenizer, data_path: str) -> Dict[str, float]:
        """Evaluate generation quality"""
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        model.eval()
        device = next(model.parameters()).device
        
        total_length = 0
        num_samples = 0
        
        for item in tqdm(data, desc=f"Evaluating {self.name}"):
            prompt = item.get('prompt', item.get('text', ''))
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):]
            
            total_length += len(response.split())
            num_samples += 1
        
        avg_length = total_length / num_samples if num_samples > 0 else 0
        
        return {
            f"{self.name}/avg_response_length": avg_length,
            f"{self.name}/num_samples": num_samples
        }


class MathTask(EvaluationTask):
    """Math problem evaluation (GSM8K-style)"""
    
    def __init__(self, name: str = "math"):
        super().__init__(name)
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract numerical answer from generated text"""
        # Look for patterns like "#### 42" (GSM8K format)
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            return match.group(1)
        
        # Look for "The answer is X"
        match = re.search(r'(?:answer|result) is[:\s]+(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Extract last number in text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def evaluate(self, model, tokenizer, data_path: str) -> Dict[str, float]:
        """Evaluate math problem solving accuracy"""
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        model.eval()
        device = next(model.parameters()).device
        
        correct = 0
        total = 0
        
        for item in tqdm(data, desc=f"Evaluating {self.name}"):
            question = item.get('question', item.get('problem', ''))
            true_answer = item.get('answer', '')
            
            # Extract numerical answer from ground truth
            true_answer_num = self.extract_answer(true_answer)
            if true_answer_num is None:
                continue
            
            # Generate solution
            prompt = f"Solve this math problem:\n{question}\n\nSolution:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Greedy decoding for evaluation
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_answer = self.extract_answer(generated_text)
            
            if pred_answer is not None and pred_answer == true_answer_num:
                correct += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            f"{self.name}/accuracy": accuracy,
            f"{self.name}/correct": correct,
            f"{self.name}/total": total
        }


class CodeTask(EvaluationTask):
    """Code generation evaluation"""
    
    def __init__(self, name: str = "code", language: str = "python"):
        super().__init__(name)
        self.language = language
    
    def extract_code(self, text: str) -> str:
        """Extract code block from generated text"""
        # Look for code blocks
        pattern = f"```{self.language}\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Look for any code block
        pattern = "```\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return text.strip()
    
    def evaluate(self, model, tokenizer, data_path: str) -> Dict[str, float]:
        """Evaluate code generation"""
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        model.eval()
        device = next(model.parameters()).device
        
        num_samples = 0
        avg_code_length = 0
        
        for item in tqdm(data, desc=f"Evaluating {self.name}"):
            instruction = item.get('instruction', item.get('prompt', ''))
            
            prompt = f"Write {self.language} code to:\n{instruction}\n\nCode:\n```{self.language}\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.2,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = self.extract_code(generated_text)
            
            avg_code_length += len(code.split('\n'))
            num_samples += 1
        
        avg_code_length = avg_code_length / num_samples if num_samples > 0 else 0
        
        return {
            f"{self.name}/avg_code_lines": avg_code_length,
            f"{self.name}/num_samples": num_samples
        }


class PerplexityTask(EvaluationTask):
    """Perplexity evaluation"""
    
    def __init__(self, name: str = "perplexity"):
        super().__init__(name)
    
    def evaluate(self, model, tokenizer, data_path: str) -> Dict[str, float]:
        """Compute perplexity on test set"""
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        model.eval()
        device = next(model.parameters()).device
        
        total_loss = 0
        total_tokens = 0
        
        for item in tqdm(data, desc=f"Evaluating {self.name}"):
            text = item.get('text', item.get('prompt', ''))
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
            
            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        return {
            f"{self.name}/perplexity": perplexity,
            f"{self.name}/avg_loss": avg_loss
        }


# ============================================================================
# Main Evaluator
# ============================================================================

class ModelEvaluator:
    """Main evaluator class"""
    
    def __init__(
        self,
        model_path: str,
        precision: str = "bf16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = model_path
        self.device = device
        
        # Setup precision
        if precision == "bf16":
            self.dtype = torch.bfloat16
        elif precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto"
        )
        self.model.eval()
        
        # Available tasks
        self.tasks = {
            "generation": GenerationTask(),
            "math": MathTask(),
            "code": CodeTask(),
            "perplexity": PerplexityTask(),
        }
    
    def evaluate_task(self, task_name: str, data_path: str) -> Dict[str, float]:
        """Evaluate on a specific task"""
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.tasks.keys())}")
        
        task = self.tasks[task_name]
        metrics = task.evaluate(self.model, self.tokenizer, data_path)
        
        return metrics
    
    def evaluate_all(self, eval_config: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate on multiple tasks
        
        Args:
            eval_config: Dict mapping task names to data paths
                Example: {"math": "math_test.jsonl", "code": "code_test.jsonl"}
        
        Returns:
            Combined metrics dictionary
        """
        all_metrics = {}
        
        for task_name, data_path in eval_config.items():
            print(f"\n{'='*50}")
            print(f"Evaluating task: {task_name}")
            print(f"{'='*50}")
            
            metrics = self.evaluate_task(task_name, data_path)
            all_metrics.update(metrics)
            
            # Print metrics
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
        
        return all_metrics
    
    def save_results(self, metrics: Dict[str, float], output_path: str):
        """Save evaluation results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RL-trained LLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--tasks", type=str, nargs="+", default=["generation"], help="Tasks to evaluate")
    parser.add_argument("--data_paths", type=str, nargs="+", help="Paths to evaluation data")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        precision=args.precision
    )
    
    # Build eval config
    if args.data_paths:
        eval_config = dict(zip(args.tasks, args.data_paths))
    else:
        raise ValueError("Must provide --data_paths")
    
    # Run evaluation
    metrics = evaluator.evaluate_all(eval_config)
    
    # Save results
    evaluator.save_results(metrics, args.output)


if __name__ == "__main__":
    main()
