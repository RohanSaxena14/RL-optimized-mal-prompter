import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class RankingExample:
    """Single training example from ranking feedback."""
    prompt_id: int
    category: str
    jailbreak_template: str
    strategy_description: str
    rank: int  # 0 = best, higher = worse
    reward: float
    

class RankingDataset(Dataset):
    """Dataset for RL training using ranking feedback."""
    
    def __init__(
        self, 
        examples: List[RankingExample],
        tokenizer,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format as a training prompt
        # The model learns to generate jailbreak templates given category
        prompt = f"Generate a jailbreak prompt for category: {example.category}\nStrategy: {example.strategy_description}\nTemplate: "
        target = example.jailbreak_template
        
        # Tokenize
        encoding = self.tokenizer(
            prompt + target,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels (only compute loss on the template part)
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels = encoding['input_ids'].clone()
        labels[0, :prompt_length] = -100  # Ignore prompt tokens in loss
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'reward': torch.tensor(example.reward, dtype=torch.float32),
            'rank': torch.tensor(example.rank, dtype=torch.long)
        }


class PreferenceDataset(Dataset):
    """Dataset for preference learning (pairwise comparisons)."""
    
    def __init__(
        self,
        preference_pairs: List[Tuple[RankingExample, RankingExample]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            preference_pairs: List of (winner, loser) example tuples
        """
        self.preference_pairs = preference_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx):
        winner, loser = self.preference_pairs[idx]
        
        # Tokenize both examples
        winner_prompt = f"Generate a jailbreak prompt for category: {winner.category}\nStrategy: {winner.strategy_description}\nTemplate: {winner.jailbreak_template}"
        loser_prompt = f"Generate a jailbreak prompt for category: {loser.category}\nStrategy: {loser.strategy_description}\nTemplate: {loser.jailbreak_template}"
        
        winner_encoding = self.tokenizer(
            winner_prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        loser_encoding = self.tokenizer(
            loser_prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'winner_input_ids': winner_encoding['input_ids'].squeeze(),
            'winner_attention_mask': winner_encoding['attention_mask'].squeeze(),
            'loser_input_ids': loser_encoding['input_ids'].squeeze(),
            'loser_attention_mask': loser_encoding['attention_mask'].squeeze()
        }


class RLTrainer:
    """
    Trains the attacker model using ranking feedback.
    Supports multiple RL algorithms.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-5,
        rl_algorithm: str = 'reward_weighted'  # 'reward_weighted', 'dpo', 'ppo_simple'
    ):
        """
        Args:
            model_name_or_path: HuggingFace model name or path
            rl_algorithm: Which RL algorithm to use
                - 'reward_weighted': Supervised learning with reward weighting
                - 'dpo': Direct Preference Optimization
                - 'ppo_simple': Simplified PPO with advantage estimation
        """
        self.device = device
        self.rl_algorithm = rl_algorithm
        
        # Load model and tokenizer
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        
        # For DPO, keep a reference model
        if rl_algorithm == 'dpo':
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        self.learning_rate = learning_rate
        self.optimizer = None
        self.scheduler = None
    
    def train_from_rankings(
        self,
        category_rankings: Dict[str, Dict],
        epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        save_path: str = None
    ):
        """
        Train the model using ranking feedback from all categories.
        
        Args:
            category_rankings: Dict mapping category -> ranking data
                {
                    'category_name': {
                        'ranked_prompts': [list of prompt dicts ordered best to worst],
                        'rewards': {prompt_id: reward_value},
                        'preference_pairs': [(winner_id, loser_id), ...]
                    }
                }
        """
        print(f"\n{'='*60}")
        print(f"Starting RL Training with {self.rl_algorithm} algorithm")
        print(f"{'='*60}\n")
        
        if self.rl_algorithm == 'reward_weighted':
            self._train_reward_weighted(category_rankings, epochs, batch_size, 
                                       gradient_accumulation_steps, warmup_steps, save_path)
        elif self.rl_algorithm == 'dpo':
            self._train_dpo(category_rankings, epochs, batch_size,
                           gradient_accumulation_steps, warmup_steps, save_path)
        elif self.rl_algorithm == 'ppo_simple':
            self._train_ppo_simple(category_rankings, epochs, batch_size,
                                  gradient_accumulation_steps, warmup_steps, save_path)
        else:
            raise ValueError(f"Unknown RL algorithm: {self.rl_algorithm}")
    
    def _train_reward_weighted(
        self, 
        category_rankings: Dict,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        warmup_steps: int,
        save_path: str
    ):
        """
        Reward-weighted supervised learning.
        Higher ranked prompts get higher weight in the loss.
        """
        # Convert rankings to training examples
        examples = []
        for category, ranking_data in category_rankings.items():
            ranked_prompts = ranking_data['ranked_prompts']
            rewards = ranking_data['rewards']
            
            for rank, prompt_dict in enumerate(ranked_prompts):
                example = RankingExample(
                    prompt_id=prompt_dict['prompt_id'],
                    category=category,
                    jailbreak_template=prompt_dict['jailbreak_template'],
                    strategy_description=prompt_dict.get('strategy_description', ''),
                    rank=rank,
                    reward=rewards.get(prompt_dict['prompt_id'], 0.0)
                )
                examples.append(example)
        
        print(f"Created {len(examples)} training examples from rankings")
        
        # Create dataset and dataloader
        dataset = RankingDataset(examples, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            for step, batch in enumerate(dataloader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                rewards = batch['reward'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Compute reward-weighted loss
                loss = outputs.loss
                
                # Weight by rewards (higher reward = more weight)
                reward_weights = rewards / (rewards.sum() + 1e-8)
                weighted_loss = (loss * reward_weights.mean()).mean()
                
                # Scale by gradient accumulation
                weighted_loss = weighted_loss / gradient_accumulation_steps
                weighted_loss.backward()
                
                epoch_loss += weighted_loss.item()
                
                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                if (step + 1) % 100 == 0:
                    print(f"  Step {step + 1}, Loss: {epoch_loss / (step + 1):.4f}")
            
            print(f"Epoch {epoch + 1} completed. Avg Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save model
        if save_path:
            self.save_model(save_path)
            print(f"Model saved to {save_path}")
    
    def _train_dpo(
        self,
        category_rankings: Dict,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        warmup_steps: int,
        save_path: str
    ):
        """
        Direct Preference Optimization (DPO).
        Learns from pairwise preferences without explicit reward model.
        """
        # Convert rankings to preference pairs
        all_preference_pairs = []
        
        for category, ranking_data in category_rankings.items():
            ranked_prompts = ranking_data['ranked_prompts']
            
            # Create pairwise preferences
            for i in range(len(ranked_prompts)):
                for j in range(i + 1, min(i + 5, len(ranked_prompts))):  # Compare with next 5
                    winner_dict = ranked_prompts[i]
                    loser_dict = ranked_prompts[j]
                    
                    winner = RankingExample(
                        prompt_id=winner_dict['prompt_id'],
                        category=category,
                        jailbreak_template=winner_dict['jailbreak_template'],
                        strategy_description=winner_dict.get('strategy_description', ''),
                        rank=i,
                        reward=1.0
                    )
                    
                    loser = RankingExample(
                        prompt_id=loser_dict['prompt_id'],
                        category=category,
                        jailbreak_template=loser_dict['jailbreak_template'],
                        strategy_description=loser_dict.get('strategy_description', ''),
                        rank=j,
                        reward=0.0
                    )
                    
                    all_preference_pairs.append((winner, loser))
        
        print(f"Created {len(all_preference_pairs)} preference pairs for DPO")
        
        # Create dataset
        dataset = PreferenceDataset(all_preference_pairs, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # DPO hyperparameters
        beta = 0.1  # KL penalty coefficient
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            for step, batch in enumerate(dataloader):
                # Move to device
                winner_ids = batch['winner_input_ids'].to(self.device)
                winner_mask = batch['winner_attention_mask'].to(self.device)
                loser_ids = batch['loser_input_ids'].to(self.device)
                loser_mask = batch['loser_attention_mask'].to(self.device)
                
                # Get log probs from policy model
                with torch.cuda.amp.autocast():
                    winner_logits = self.model(winner_ids, attention_mask=winner_mask).logits
                    loser_logits = self.model(loser_ids, attention_mask=loser_mask).logits
                    
                    # Get log probs from reference model
                    with torch.no_grad():
                        ref_winner_logits = self.ref_model(winner_ids, attention_mask=winner_mask).logits
                        ref_loser_logits = self.ref_model(loser_ids, attention_mask=loser_mask).logits
                    
                    # Compute log probabilities
                    winner_logprobs = F.log_softmax(winner_logits, dim=-1)
                    loser_logprobs = F.log_softmax(loser_logits, dim=-1)
                    ref_winner_logprobs = F.log_softmax(ref_winner_logits, dim=-1)
                    ref_loser_logprobs = F.log_softmax(ref_loser_logits, dim=-1)
                    
                    # Get per-token log probs (simplified - just use mean)
                    winner_lp = winner_logprobs.mean(dim=[1, 2])
                    loser_lp = loser_logprobs.mean(dim=[1, 2])
                    ref_winner_lp = ref_winner_logprobs.mean(dim=[1, 2])
                    ref_loser_lp = ref_loser_logprobs.mean(dim=[1, 2])
                    
                    # DPO loss
                    pi_logratios = winner_lp - loser_lp
                    ref_logratios = ref_winner_lp - ref_loser_lp
                    
                    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
                
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                if (step + 1) % 100 == 0:
                    print(f"  Step {step + 1}, DPO Loss: {epoch_loss / (step + 1):.4f}")
            
            print(f"Epoch {epoch + 1} completed. Avg Loss: {epoch_loss / len(dataloader):.4f}")
        
        if save_path:
            self.save_model(save_path)
            print(f"Model saved to {save_path}")
    
    def _train_ppo_simple(
        self,
        category_rankings: Dict,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        warmup_steps: int,
        save_path: str
    ):
        """
        Simplified PPO using advantage estimation from rankings.
        """
        print("Note: This is a simplified PPO implementation using ranking-based advantages")
        
        # Similar to reward_weighted but with advantage normalization
        examples = []
        for category, ranking_data in category_rankings.items():
            ranked_prompts = ranking_data['ranked_prompts']
            rewards = ranking_data['rewards']
            
            # Convert rewards to advantages
            reward_values = [rewards.get(p['prompt_id'], 0.0) for p in ranked_prompts]
            mean_reward = np.mean(reward_values)
            std_reward = np.std(reward_values) + 1e-8
            
            for rank, prompt_dict in enumerate(ranked_prompts):
                reward = rewards.get(prompt_dict['prompt_id'], 0.0)
                advantage = (reward - mean_reward) / std_reward  # Normalized advantage
                
                example = RankingExample(
                    prompt_id=prompt_dict['prompt_id'],
                    category=category,
                    jailbreak_template=prompt_dict['jailbreak_template'],
                    strategy_description=prompt_dict.get('strategy_description', ''),
                    rank=rank,
                    reward=advantage  # Use advantage as "reward"
                )
                examples.append(example)
        
        # Rest is similar to reward_weighted
        self._train_reward_weighted(
            {'ppo_data': {'ranked_prompts': [
                {'prompt_id': e.prompt_id, 'jailbreak_template': e.jailbreak_template, 
                 'strategy_description': e.strategy_description} for e in examples
            ], 'rewards': {e.prompt_id: e.reward for e in examples}}},
            epochs, batch_size, gradient_accumulation_steps, warmup_steps, save_path
        )
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a trained model."""
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")