import json
import os
import pickle
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

from pipelines.unified_category_attacker import UnifiedCategoryAttacker
from pipelines.rank_based_scorer import RankBasedScorer
from pipelines.rl_trainer import RLTrainer


class RLJailbreakPipeline:
    """
    Main pipeline for RL-based jailbreak prompt optimization.
    
    Iteration Flow:
    1. Generate prompts per category (warmup or RL-guided)
    2. Create attack dataset by filling prompts with queries
    3. Test on target model
    4. Rank-order results per category
    5. Train attacker using RL
    6. Reload updated attacker model for next iteration
    """
    
    def __init__(
        self,
        attacker: UnifiedCategoryAttacker,
        scorer: RankBasedScorer,
        target_model,
        rl_trainer: RLTrainer,
        categories: List[str] = None,
        logger = None
    ):
        """
        Args:
            attacker: UnifiedCategoryAttacker instance
            scorer: RankBasedScorer instance
            target_model: Model to jailbreak
            rl_trainer: RLTrainer instance
            categories: List of attack categories
            logger: Optional logger
        """
        self.attacker = attacker
        self.scorer = scorer
        self.target_model = target_model
        self.rl_trainer = rl_trainer
        self.categories = categories or UnifiedCategoryAttacker.CATEGORIES
        self.logger = logger
        
        self.iteration_history = []
    
    def log(self, message: str):
        """Log message."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def run_iteration(
        self,
        iteration_num: int,
        queries_per_category: Dict[str, List[str]],
        prompts_per_category: int = 20,
        queries_to_test: int = 10,
        is_warmup: bool = False,
        previous_best_prompts: Dict[str, List[Dict]] = None,
        reward_scheme: str = "linear",
        save_dir: str = "./rl_jailbreak_results"
    ) -> Dict:
        """
        Run one complete iteration of the RL pipeline.
        
        Args:
            iteration_num: Current iteration number
            queries_per_category: Dict mapping category -> list of harmful queries
            prompts_per_category: How many jailbreak prompts to generate per category
            queries_to_test: How many queries to test each prompt on
            is_warmup: If True, use warmup generation; else use RL-guided
            previous_best_prompts: Best prompts from previous iteration (for RL-guided)
            reward_scheme: How to convert rankings to rewards ('linear', 'exponential', 'winner_takes_all')
            save_dir: Where to save results
        
        Returns:
            Dict with iteration results
        """
        self.log(f"\n{'='*80}")
        self.log(f"ITERATION {iteration_num} {'(WARMUP)' if is_warmup else '(RL-GUIDED)'}")
        self.log(f"{'='*80}\n")
        
        os.makedirs(save_dir, exist_ok=True)
        
        iteration_results = {
            'iteration': iteration_num,
            'is_warmup': is_warmup,
            'timestamp': datetime.now().isoformat(),
            'categories': {}
        }
        
        # Step 1: Generate jailbreak prompts for each category
        self.log("STEP 1: Generating jailbreak prompts...")
        all_prompts = {}
        
        for category in self.categories:
            self.log(f"\n  Category: {category}")
            
            # Get example query and top prompts if available
            example_query = queries_per_category.get(category, [None])[0]
            top_prompts = previous_best_prompts.get(category, []) if previous_best_prompts else []
            
            # Use unified generate_prompts method
            # It automatically handles warmup vs RL-guided based on top_prompts
            prompts = self.attacker.generate_prompts(
                category=category,
                num_prompts=prompts_per_category,
                example_query=example_query,
                top_performing_prompts=top_prompts if not is_warmup else None
            )
            
            all_prompts[category] = prompts
            mode = "warmup" if is_warmup else "RL-guided"
            self.log(f"    Generated {len(prompts)} {mode} prompts for {category}")
        
        # Step 2: Create attack dataset
        self.log("\nSTEP 2: Creating attack dataset...")
        attack_dataset = {}
        
        for category in self.categories:
            category_queries = queries_per_category.get(category, [])[:queries_to_test]
            if not category_queries:
                self.log(f"  WARNING: No queries for category {category}")
                continue
            
            category_attacks = []
            prompts = all_prompts[category]
            
            for prompt_template in prompts:
                # Apply template to all queries in this category
                filled_prompts = self.attacker.apply_template_to_queries(
                    prompt_template, category_queries
                )
                category_attacks.extend(filled_prompts)
            
            attack_dataset[category] = category_attacks
            self.log(f"  {category}: {len(category_attacks)} attack instances " +
                    f"({len(prompts)} prompts × {len(category_queries)} queries)")
        
        # Step 3: Test on target model
        self.log("\nSTEP 3: Testing attacks on target model...")
        target_responses = {}
        
        for category in self.categories:
            if category not in attack_dataset:
                continue
            
            self.log(f"  Processing {category}...")
            category_responses = []
            
            for attack in attack_dataset[category]:
                try:
                    # Send jailbreak prompt to target model
                    response = self.target_model.generate(attack['jailbreak_prompt'])
                    
                    category_responses.append({
                        'prompt_id': attack['prompt_id'],
                        'query': attack['original_query'],
                        'jailbreak_prompt': attack['jailbreak_prompt'],
                        'target_response': response,
                        'strategy_description': attack.get('strategy_description', '')
                    })
                except Exception as e:
                    self.log(f"    Error testing attack: {e}")
                    category_responses.append({
                        'prompt_id': attack['prompt_id'],
                        'query': attack['original_query'],
                        'jailbreak_prompt': attack['jailbreak_prompt'],
                        'target_response': "[ERROR: Could not generate response]",
                        'strategy_description': attack.get('strategy_description', '')
                    })
            
            target_responses[category] = category_responses
            self.log(f"    Completed {len(category_responses)} attacks for {category}")
        
        # Step 4: Rank-order results per category
        self.log("\nSTEP 4: Ranking attack effectiveness...")
        category_rankings = {}
        
        for category in self.categories:
            if category not in target_responses:
                continue
            
            self.log(f"  Ranking {category}...")
            
            # Group responses by prompt_id (same prompt on different queries)
            prompt_groups = {}
            for response in target_responses[category]:
                pid = response['prompt_id']
                if pid not in prompt_groups:
                    prompt_groups[pid] = []
                prompt_groups[pid].append(response)
            
            # For ranking, we'll use one representative example per prompt
            # (or could average scores across queries - this is simplified)
            query_response_pairs = []
            for pid, responses in prompt_groups.items():
                # Use first response as representative (or could do more sophisticated aggregation)
                rep_response = responses[0]
                query_response_pairs.append(rep_response)
            
            # Get rankings from scorer
            ranked_ids, reasoning = self.scorer.rank_category_attempts(
                category=category,
                query_response_pairs=query_response_pairs
            )
            
            # Compute rewards from rankings
            rewards = self.scorer.get_ranking_rewards(ranked_ids, reward_scheme=reward_scheme)
            
            # Create ranked prompt list
            id_to_prompt = {p['prompt_id']: p for p in all_prompts[category]}
            ranked_prompts = [id_to_prompt[pid] for pid in ranked_ids]
            
            # Store category ranking data
            category_rankings[category] = {
                'ranked_prompts': ranked_prompts,
                'rewards': rewards,
                'ranking_reasoning': reasoning,
                'query_response_pairs': query_response_pairs
            }
            
            iteration_results['categories'][category] = {
                'num_prompts': len(ranked_prompts),
                'best_prompt_id': ranked_ids[0] if ranked_ids else None,
                'worst_prompt_id': ranked_ids[-1] if ranked_ids else None,
                'best_reward': rewards.get(ranked_ids[0], 0.0) if ranked_ids else 0.0,
                'ranking_reasoning': reasoning
            }
            
            self.log(f"    Ranked {len(ranked_ids)} prompts")
            self.log(f"    Best prompt ID: {ranked_ids[0] if ranked_ids else 'N/A'}")
        
        # Step 5: Train attacker using RL
        if not is_warmup:  # Only train after warmup
            self.log("\nSTEP 5: Training attacker with RL...")
            
            try:
                model_save_path = os.path.join(save_dir, f"attacker_model_iter_{iteration_num}")
                
                self.rl_trainer.train_from_rankings(
                    category_rankings=category_rankings,
                    epochs=3,
                    batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=50,
                    save_path=model_save_path
                )
                self.log("  RL training completed successfully")
                iteration_results['rl_training_completed'] = True
                
                # CRITICAL: Reload the updated model into the attacker
                self.log(f"\nSTEP 6: Reloading updated attacker model...")
                if not self.attacker.use_api:
                    self.attacker.load_finetuned_model(model_save_path)
                    self.log("  Attacker model updated with new weights!")
                else:
                    self.log("  Skipping model reload (using API model)")
                
            except Exception as e:
                self.log(f"  ERROR in RL training: {e}")
                iteration_results['rl_training_completed'] = False
                iteration_results['rl_training_error'] = str(e)
        else:
            self.log("\nSTEP 5: Skipping RL training (warmup iteration)")
            iteration_results['rl_training_completed'] = False
        
        # Save iteration results
        self.log("\nSaving iteration results...")
        
        # Save full results
        results_file = os.path.join(save_dir, f"iteration_{iteration_num}_results.json")
        with open(results_file, 'w') as f:
            # Convert for JSON serialization
            json_safe_results = {
                'iteration': iteration_results['iteration'],
                'is_warmup': iteration_results['is_warmup'],
                'timestamp': iteration_results['timestamp'],
                'rl_training_completed': iteration_results.get('rl_training_completed', False),
                'categories': iteration_results['categories']
            }
            json.dump(json_safe_results, f, indent=2)
        
        # Save category rankings (for next iteration)
        rankings_file = os.path.join(save_dir, f"iteration_{iteration_num}_rankings.pkl")
        with open(rankings_file, 'wb') as f:
            pickle.dump(category_rankings, f)
        
        self.log(f"Results saved to {save_dir}")
        
        # Store in history
        self.iteration_history.append(iteration_results)
        
        return iteration_results
    
    def run_full_pipeline(
        self,
        queries_per_category: Dict[str, List[str]],
        num_iterations: int = 5,
        prompts_per_category: int = 20,
        queries_to_test: int = 10,
        reward_scheme: str = "linear",
        save_dir: str = "./rl_jailbreak_results"
    ):
        """
        Run the complete RL pipeline for multiple iterations.
        
        Args:
            queries_per_category: Dict mapping category -> list of harmful queries
            num_iterations: Total iterations to run (including warmup)
            prompts_per_category: Prompts to generate per category per iteration
            queries_to_test: Queries to test each prompt on
            reward_scheme: Reward scheme for RL training
            save_dir: Where to save all results
        """
        self.log(f"\n{'#'*80}")
        self.log(f"STARTING RL JAILBREAK PIPELINE")
        self.log(f"Total iterations: {num_iterations}")
        self.log(f"Categories: {', '.join(self.categories)}")
        self.log(f"{'#'*80}\n")
        
        previous_best_prompts = None
        
        for iteration in range(num_iterations):
            is_warmup = (iteration == 0)
            
            results = self.run_iteration(
                iteration_num=iteration,
                queries_per_category=queries_per_category,
                prompts_per_category=prompts_per_category,
                queries_to_test=queries_to_test,
                is_warmup=is_warmup,
                previous_best_prompts=previous_best_prompts,
                reward_scheme=reward_scheme,
                save_dir=save_dir
            )
            
            # Extract top performing prompts for next iteration
            if iteration < num_iterations - 1:  # Not last iteration
                rankings_file = os.path.join(save_dir, f"iteration_{iteration}_rankings.pkl")
                with open(rankings_file, 'rb') as f:
                    category_rankings = pickle.load(f)
                
                previous_best_prompts = {}
                for category, ranking_data in category_rankings.items():
                    # Take top 5 prompts from this iteration
                    top_k = 5
                    previous_best_prompts[category] = ranking_data['ranked_prompts'][:top_k]
        
        self.log(f"\n{'#'*80}")
        self.log(f"PIPELINE COMPLETED")
        self.log(f"All results saved to: {save_dir}")
        self.log(f"{'#'*80}\n")
        
        # Save final summary
        summary_file = os.path.join(save_dir, "pipeline_summary.json")
        with open(summary_file, 'w') as f:
            summary = {
                'num_iterations': num_iterations,
                'categories': self.categories,
                'prompts_per_category': prompts_per_category,
                'queries_to_test': queries_to_test,
                'reward_scheme': reward_scheme,
                'iterations': [
                    {
                        'iteration': r['iteration'],
                        'is_warmup': r['is_warmup'],
                        'timestamp': r['timestamp'],
                        'categories': r['categories']
                    }
                    for r in self.iteration_history
                ]
            }
            json.dump(summary, f, indent=2)
        
        self.log(f"Summary saved to {summary_file}")
        
        return self.iteration_history