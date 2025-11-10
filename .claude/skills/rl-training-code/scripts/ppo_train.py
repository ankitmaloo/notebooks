"""
PPO Training Script Template for LLM RL Post-training using verl
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import wandb
from pathlib import Path
import json
from typing import Optional, Dict, Any

# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Configuration for PPO training"""
    def __init__(
        self,
        # Model settings
        model_name: str = "meta-llama/Llama-2-7b-hf",
        precision: str = "bf16",  # "bf16", "fp16", "fp32"
        
        # Training settings
        total_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 1e-6,
        max_seq_length: int = 512,
        
        # PPO settings
        ppo_epochs: int = 4,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.1,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,
        
        # Logging and checkpointing
        log_interval: int = 10,
        save_interval: int = 500,
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = True,
        wandb_project: str = "llm-rl-training",
        
        # Device settings
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.precision = precision
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.ppo_epochs = ppo_epochs
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
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
# Data Loading
# ============================================================================

class RLDataset(Dataset):
    """Dataset for RL training"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data (expecting JSONL format)
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt', item.get('text', ''))
        
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
            'prompt': prompt
        }


# ============================================================================
# Reward Function
# ============================================================================

def compute_reward(response: str, prompt: str) -> float:
    """
    Compute reward for a generated response.
    This is a placeholder - replace with your actual reward function.
    
    Args:
        response: Generated text
        prompt: Input prompt
        
    Returns:
        reward: Float reward value
    """
    # Example: Simple length-based reward (REPLACE THIS)
    reward = min(len(response.split()) / 20.0, 1.0)
    
    # TODO: Implement your actual reward function
    # Examples:
    # - Call external reward model
    # - Rule-based scoring
    # - Human feedback
    # - Task-specific metrics (accuracy, correctness, etc.)
    
    return reward


# ============================================================================
# Training Loop
# ============================================================================

class PPOTrainer:
    """PPO Trainer for LLM RL post-training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=vars(config))
        
        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            device_map="auto"
        )
        
        # Create reference model (for KL divergence)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            device_map="auto"
        )
        self.ref_model.eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
    
    def generate_responses(self, prompts, max_new_tokens=100):
        """Generate responses for given prompts"""
        self.model.eval()
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return responses
    
    def compute_log_probs(self, model, input_ids, attention_mask):
        """Compute log probabilities"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get log probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather log probs of actual tokens
        token_log_probs = torch.gather(
            log_probs[:, :-1, :],
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = attention_mask[:, 1:].bool()
        token_log_probs = token_log_probs * mask
        
        return token_log_probs.sum(dim=1)
    
    def train_step(self, batch):
        """Single PPO training step"""
        prompts = batch['prompt']
        
        # Generate responses
        responses = self.generate_responses(prompts)
        
        # Compute rewards
        rewards = torch.tensor([
            compute_reward(resp, prompt)
            for resp, prompt in zip(responses, prompts)
        ]).to(self.device)
        
        # Tokenize full sequences (prompt + response)
        full_texts = responses
        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        # PPO updates
        for _ in range(self.config.ppo_epochs):
            # Current policy log probs
            current_log_probs = self.compute_log_probs(
                self.model,
                encodings['input_ids'],
                encodings['attention_mask']
            )
            
            # Reference policy log probs (for KL)
            with torch.no_grad():
                ref_log_probs = self.compute_log_probs(
                    self.ref_model,
                    encodings['input_ids'],
                    encodings['attention_mask']
                )
            
            # Compute KL divergence
            kl_div = current_log_probs - ref_log_probs
            
            # Compute advantages (simplified - no value function here)
            advantages = rewards - self.config.kl_coef * kl_div
            
            # PPO loss
            ratio = torch.exp(current_log_probs - ref_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_range,
                1 + self.config.clip_range
            )
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            # Total loss
            loss = policy_loss
            
            # Backward pass
            loss.backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        metrics = {
            'loss': loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_kl': kl_div.mean().item(),
        }
        
        return metrics
    
    def train(self, train_dataloader):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.config.total_epochs):
            self.model.train()
            
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
        """Save model checkpoint"""
        suffix = "final" if final else f"step_{self.global_step}"
        save_path = Path(self.config.checkpoint_dir) / f"checkpoint_{suffix}"
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Saved checkpoint to {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    # Configuration
    config = TrainingConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        precision="bf16",
        total_epochs=3,
        batch_size=4,
    )
    
    # Load data
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = RLDataset("path/to/your/data.jsonl", tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Train
    trainer = PPOTrainer(config)
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()
