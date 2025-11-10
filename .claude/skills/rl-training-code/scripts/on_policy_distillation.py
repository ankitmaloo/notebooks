"""
On-Policy Distillation Training Script for LLM RL
Distill a larger teacher model into a smaller student model using on-policy data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import wandb
from pathlib import Path
import json
from typing import Optional, List

# ============================================================================
# Configuration
# ============================================================================

class DistillationConfig:
    """Configuration for on-policy distillation"""
    def __init__(
        self,
        # Model settings
        teacher_model_name: str = "meta-llama/Llama-2-13b-hf",
        student_model_name: str = "meta-llama/Llama-2-7b-hf",
        precision: str = "bf16",  # "bf16", "fp16", "fp32"
        
        # Training settings
        total_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-5,
        max_seq_length: int = 512,
        max_gen_length: int = 100,
        
        # Distillation settings
        temperature: float = 2.0,
        alpha: float = 0.5,  # Weight for distillation loss vs student loss
        kl_weight: float = 1.0,
        
        # Generation settings for on-policy
        num_generations_per_prompt: int = 4,
        generation_temperature: float = 0.8,
        top_p: float = 0.9,
        
        # Logging and checkpointing
        log_interval: int = 10,
        save_interval: int = 500,
        checkpoint_dir: str = "./distillation_checkpoints",
        use_wandb: bool = True,
        wandb_project: str = "llm-distillation",
        
        # Device settings
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.precision = precision
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.alpha = alpha
        self.kl_weight = kl_weight
        self.num_generations_per_prompt = num_generations_per_prompt
        self.generation_temperature = generation_temperature
        self.top_p = top_p
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.device = device
        
        # Setup precision dtype
        if precision == "bf16":
            self.dtype = torch.bfloat16
        elif precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32


# ============================================================================
# On-Policy Distillation Trainer
# ============================================================================

class OnPolicyDistillationTrainer:
    """On-policy distillation trainer"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = config.device
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=vars(config))
        
        # Load tokenizer
        print(f"Loading tokenizer from {config.teacher_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load teacher model
        print(f"Loading teacher model: {config.teacher_model_name}")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=config.dtype,
            device_map="auto"
        )
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Load student model
        print(f"Loading student model: {config.student_model_name}")
        self.student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=config.dtype,
            device_map="auto"
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate
        )
        
        # Checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
    
    def generate_teacher_samples(self, prompts: List[str]):
        """Generate samples from teacher model (on-policy data)"""
        self.teacher_model.eval()
        
        all_generations = []
        
        for prompt in prompts:
            # Generate multiple samples per prompt
            inputs = self.tokenizer(
                [prompt] * self.config.num_generations_per_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_gen_length,
                    do_sample=True,
                    temperature=self.config.generation_temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=self.config.num_generations_per_prompt
                )
            
            generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_generations.extend(generations)
        
        return all_generations
    
    def compute_distillation_loss(self, student_logits, teacher_logits, labels, attention_mask):
        """
        Compute distillation loss combining:
        1. KL divergence between student and teacher
        2. Cross-entropy loss on true labels
        """
        # Shift logits and labels for causal LM
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        # KL divergence loss (distillation loss)
        student_log_probs = F.log_softmax(shift_student_logits / self.config.temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits / self.config.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none'
        ).sum(dim=-1)
        
        # Mask padding tokens
        kl_loss = (kl_loss * shift_attention_mask).sum() / shift_attention_mask.sum()
        kl_loss = kl_loss * (self.config.temperature ** 2)  # Scale by temperature squared
        
        # Cross-entropy loss (standard LM loss)
        ce_loss = F.cross_entropy(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        ce_loss = (ce_loss.view_as(shift_labels) * shift_attention_mask).sum() / shift_attention_mask.sum()
        
        # Combined loss
        total_loss = self.config.alpha * kl_loss + (1 - self.config.alpha) * ce_loss
        
        return total_loss, kl_loss, ce_loss
    
    def train_step(self, batch):
        """Single training step with on-policy distillation"""
        prompts = batch['prompt']
        
        # Generate on-policy samples from teacher
        teacher_generations = self.generate_teacher_samples(prompts)
        
        # Tokenize teacher generations
        encodings = self.tokenizer(
            teacher_generations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # Get teacher logits (for distillation)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        # Get student logits
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs.logits
        
        # Compute loss
        loss, kl_loss, ce_loss = self.compute_distillation_loss(
            student_logits,
            teacher_logits,
            input_ids,
            attention_mask
        )
        
        # Backward pass
        loss.backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        metrics = {
            'loss': loss.item(),
            'kl_loss': kl_loss.item(),
            'ce_loss': ce_loss.item(),
        }
        
        return metrics
    
    def train(self, train_dataloader):
        """Main training loop"""
        print("Starting on-policy distillation training...")
        
        for epoch in range(self.config.total_epochs):
            self.student_model.train()
            
            for batch_idx, batch in enumerate(train_dataloader):
                metrics = self.train_step(batch)
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    print(f"Epoch {epoch}, Step {self.global_step}: {metrics}")
                    if self.config.use_wandb:
                        wandb.log(metrics, step=self.global_step)
                
                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
        
        print("Training complete!")
        self.save_checkpoint(final=True)
    
    def save_checkpoint(self, final=False):
        """Save student model checkpoint"""
        suffix = "final" if final else f"step_{self.global_step}"
        save_path = Path(self.config.checkpoint_dir) / f"checkpoint_{suffix}"
        
        self.student_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Saved checkpoint to {save_path}")


# ============================================================================
# Dataset
# ============================================================================

class PromptDataset(Dataset):
    """Simple dataset of prompts for on-policy generation"""
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {'prompt': item.get('prompt', item.get('text', ''))}


# ============================================================================
# Main
# ============================================================================

def main():
    # Configuration
    config = DistillationConfig(
        teacher_model_name="meta-llama/Llama-2-13b-hf",
        student_model_name="meta-llama/Llama-2-7b-hf",
        precision="bf16",
        total_epochs=3,
        batch_size=4,
        temperature=2.0,
        alpha=0.5,
    )
    
    # Load data
    train_dataset = PromptDataset("path/to/your/prompts.jsonl")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Train
    trainer = OnPolicyDistillationTrainer(config)
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()
