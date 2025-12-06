"""
Complete RL Pipeline with Multi-Model Configuration and HarmBench Data

Uses:
- Attacker: meta-llama/Meta-Llama-3-8B-Instruct (8-bit quantized, via UnifiedCategoryAttacker)
- Scorer: microsoft/Phi-3.5-mini-instruct (via RankBasedScorer)
- Target: google/gemma-7b-it (via TargetModel wrapper)

Implements NxM evaluation:
- N attacker prompts per category
- M queries per category from HarmBench
- Creates NxM response matrix
- Aggregates rankings across all M queries
"""

import sys
import os
import csv
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('.')

from pipelines.unified_category_attacker import UnifiedCategoryAttacker
from pipelines.rank_based_scorer import RankBasedScorer
from pipelines.rl_training_pipeline import RankingToRLTrainer

from dotenv import load_dotenv
    
# Load HF token from environment
load_dotenv(dotenv_path="./.creds/hf_token.env")

# ============================================================================
# TARGET MODEL WRAPPER
# ============================================================================

class TargetModel:
    """
    Target model wrapper for generating responses to jailbreak prompts.
    """
    
    def __init__(self, model_name_or_path: str, device: str = "cuda", load_in_8bit: bool = False):
        """
        Initialize target model.
        
        Args:
            model_name_or_path: Model name or path
            device: Device to use
            load_in_8bit: Whether to use 8-bit quantization
        """
        print(f"Loading target model: {model_name_or_path}...")
        
        self.device = device
        self.model_name = model_name_or_path
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=hf_token,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "token": hf_token,
            "trust_remote_code": True,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
            model_kwargs["device_map"] = "auto" if device == "auto" else {"": device}
            device = "cuda" if device == "auto" else device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        
        if not load_in_8bit:
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"✓ Target model loaded")
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate response to jailbreak prompt.
        
        Args:
            prompt: Jailbreak prompt to test
            max_new_tokens: Max tokens to generate
        
        Returns:
            Model's response
        """
        # Check if model supports chat template
        try:
            # Try to apply chat template (works for Gemma IT and similar)
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # Fallback: use prompt directly
            formatted = prompt
        
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


# ============================================================================
# HARMBENCH DATA LOADER
# ============================================================================

def load_harmbench_data(csv_path: str = "data/HarmBench/standard/train.csv") -> Dict[str, List[str]]:
    """
    Load queries from HarmBench CSV.
    
    CSV format:
        prompt,category
        "Query text...",chemical_biological
        "Query text...",illegal
        ...
    
    Returns:
        Dict mapping category -> list of prompts
    """
    print(f"\nLoading HarmBench data from: {csv_path}")
    
    queries_per_category = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            category = row['category']
            queries_per_category[category].append(prompt)
    
    print("✓ Loaded queries:")
    for category, queries in queries_per_category.items():
        print(f"  {category}: {len(queries)} queries")
    
    return dict(queries_per_category)


# ============================================================================
# NxM EVALUATION AND RANKING AGGREGATION
# ============================================================================

def evaluate_NxM_matrix(
    attacker_prompts: List[Dict],
    queries: List[str],
    target_model: TargetModel,
    category: str
) -> List[Dict]:
    """
    Create NxM evaluation matrix.
    
    Args:
        attacker_prompts: N attacker prompts
        queries: M queries from HarmBench
        target_model: Target model
        category: Category name
    
    Returns:
        List of dicts with structure for RankBasedScorer.rank():
        {
            'prompt_id': int,
            'query': str,
            'jailbreak_prompt': str,
            'target_response': str
        }
    """
    N = len(attacker_prompts)
    M = len(queries)
    
    print(f"\n  Creating {N}x{M} evaluation matrix...")
    
    evaluation_matrix = []
    
    for i, attacker_prompt in enumerate(attacker_prompts):
        print(f"    Testing attacker prompt {i+1}/{N}...")
        
        for j, query in enumerate(queries):
            # Fill template with query
            jailbreak_prompt = attacker_prompt['jailbreak_template'].replace("{QUERY}", query)
            
            # Get target response
            target_response = target_model.generate(jailbreak_prompt)
            
            evaluation_matrix.append({
                'prompt_id': attacker_prompt['prompt_id'],
                'query': query,
                'jailbreak_prompt': jailbreak_prompt,
                'target_response': target_response
            })
    
    print(f"  ✓ Created {len(evaluation_matrix)} evaluations (N={N}, M={M})")
    
    return evaluation_matrix


def aggregate_rankings_across_queries(
    evaluation_matrix: List[Dict],
    scorer: RankBasedScorer,
    category: str,
    num_attacker_prompts: int
) -> Tuple[List[int], str]:
    """
    Aggregate rankings across all M queries to get final ranking of N prompts.
    
    Strategy:
    1. For each query, rank the N attacker prompts
    2. Aggregate rankings using Borda count method
    
    Borda Count:
        Best rank (0) gets N points
        Second rank (1) gets N-1 points
        ...
        Worst rank (N-1) gets 1 point
        
        Sum points across all M queries
        Highest total points = best overall prompt
    
    Args:
        evaluation_matrix: NxM evaluation results
        scorer: RankBasedScorer
        category: Category name
        num_attacker_prompts: N (number of attacker prompts)
    
    Returns:
        (final_ranking, reasoning)
    """
    print(f"\n  Aggregating rankings across queries...")
    
    # Group by query
    queries_map = defaultdict(list)
    for result in evaluation_matrix:
        queries_map[result['query']].append(result)
    
    M = len(queries_map)
    N = num_attacker_prompts
    
    # Borda count: accumulate points for each prompt
    borda_scores = defaultdict(float)
    
    # Track rankings for each query
    per_query_rankings = {}
    
    print(f"  Ranking N={N} prompts for each of M={M} queries...")
    
    for query_idx, (query, query_results) in enumerate(queries_map.items()):
        print(f"    Query {query_idx + 1}/{M}...", end=" ")
        
        # Rank these N prompts for this specific query
        # RankBasedScorer.rank() expects:
        # attempts: List[Dict] with keys: 'prompt_id', 'query', 'jailbreak_prompt', 'target_response'
        ranked_ids, reasoning = scorer.rank(
            category=category,
            attempts=query_results
        )
        
        per_query_rankings[query_idx] = ranked_ids
        
        # Assign Borda scores
        # Best (rank 0) gets N points, worst (rank N-1) gets 1 point
        for rank, prompt_id in enumerate(ranked_ids):
            points = N - rank
            borda_scores[prompt_id] += points
        
        print(f"Ranked: {ranked_ids[:3]}... (showing top 3)")
    
    # Sort by total Borda score (descending)
    final_ranking = sorted(borda_scores.keys(), key=lambda pid: borda_scores[pid], reverse=True)
    
    # Create reasoning
    reasoning = f"Aggregated rankings across {M} queries using Borda count.\n"
    reasoning += f"Per-query rankings:\n"
    for query_idx, ranked_ids in per_query_rankings.items():
        reasoning += f"  Query {query_idx}: {ranked_ids}\n"
    reasoning += f"\nBorda scores (higher = better):\n"
    for prompt_id in final_ranking:
        reasoning += f"  Prompt {prompt_id}: {borda_scores[prompt_id]} points\n"
    reasoning += f"\nFinal ranking: {final_ranking}"
    
    print(f"\n  ✓ Final aggregated ranking: {final_ranking}")
    print(f"    Top prompt: {final_ranking[0]} ({borda_scores[final_ranking[0]]} points)")
    print(f"    Worst prompt: {final_ranking[-1]} ({borda_scores[final_ranking[-1]]} points)")
    
    return final_ranking, reasoning


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_one_iteration_with_harmbench():
    """
    Run one complete iteration with multi-model setup and HarmBench data.
    Implements NxM evaluation and ranking aggregation.
    
    Model Configuration:
    - Attacker: Llama3 8B (8-bit quantized)
    - Scorer: Phi-3.5-mini
    - Target: Gemma 7B IT
    """
    
    print("="*80)
    print("RL PIPELINE: MULTI-MODEL SETUP WITH HARMBENCH DATA (NxM EVALUATION)")
    print("="*80)
    
    # Configuration
    N_ATTACKER_PROMPTS = 2  # N: Number of attacker prompts per category
    M_QUERIES_PER_CATEGORY = 3  # M: Number of queries to test per category
    
    print(f"\nConfiguration:")
    print(f"  N (attacker prompts): {N_ATTACKER_PROMPTS}")
    print(f"  M (queries per category): {M_QUERIES_PER_CATEGORY}")
    print(f"  Evaluation matrix: {N_ATTACKER_PROMPTS}x{M_QUERIES_PER_CATEGORY} = {N_ATTACKER_PROMPTS * M_QUERIES_PER_CATEGORY} total evaluations")
    
    # ------------------------------------------------------------------------
    # SETUP: Load Models and Data
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("SETUP")
    print("="*80)
    
    # Model Configuration
    # Models will be downloaded to /models/ directory
    ATTACKER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    SCORER_MODEL = "microsoft/Phi-3.5-mini-instruct"
    TARGET_MODEL = "google/gemma-7b-it"
    
    print(f"\nModel Configuration:")
    print(f"  Attacker: {ATTACKER_MODEL} (8-bit quantized)")
    print(f"  Scorer:   {SCORER_MODEL}")
    print(f"  Target:   {TARGET_MODEL}")
    print(f"  Storage:  /models/")
    
    # Initialize scorer (Phi-3.5)
    print("\n1. Loading Phi-3.5-mini-instruct for scorer...")
    scorer = RankBasedScorer(
        model_name=SCORER_MODEL,
        device='cuda',
        load_in_8bit=True  # Scorer can be full precision
    )
    
    # Initialize target model (Gemma 7B IT)
    print("\n2. Loading Gemma 7B IT for target...")
    target_model = TargetModel(
        model_name_or_path=TARGET_MODEL,
        device='cuda',
        load_in_8bit=True  # Can use 8-bit if memory is tight
    )
    
    # Load HarmBench data
    print("\n3. Loading HarmBench data...")
    all_queries = load_harmbench_data("data/HarmBench/standard/train.csv")
    
    # Pick one category for this demo
    category = "chemical_biological"  # Can change to any category
    queries = all_queries[category][:M_QUERIES_PER_CATEGORY]  # Take first M queries
    
    print(f"\n4. Selected category: {category}")
    print(f"   Using {len(queries)} queries:")
    for i, q in enumerate(queries):
        print(f"     {i+1}. {q[:80]}...")
    
    # ------------------------------------------------------------------------
    # STEP 1: Generate N Attacker Prompts (Llama3 8B)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: Generate N Attacker Prompts")
    print("="*80)
    
    print("\n1. Loading attacker model (Llama3 8B, 8-bit)...")
    attacker = UnifiedCategoryAttacker(
        model_name_or_path=ATTACKER_MODEL,
        device='cuda',
        load_in_8bit=True  # 8-bit quantization for attacker
    )
    
    print(f"\n2. Generating {N_ATTACKER_PROMPTS} attacker prompts for category: {category}")
    
    attacker_prompts = attacker.generate_prompts(
        category=category,
        num_prompts=N_ATTACKER_PROMPTS,
        top_performing_prompts=None  # Warmup mode
    )
    
    print(f"✓ Generated {len(attacker_prompts)} attacker prompts:")
    for i, p in enumerate(attacker_prompts):
        print(f"  {i+1}. {p['strategy_name']}")
        print(f"     Template: {p['jailbreak_template'][:100]}...")
    
    # ------------------------------------------------------------------------
    # STEP 2: Create NxM Evaluation Matrix (Target: Gemma 7B)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: Create NxM Evaluation Matrix")
    print("="*80)
    
    print(f"\nTesting {N_ATTACKER_PROMPTS} prompts on {M_QUERIES_PER_CATEGORY} queries...")
    print(f"This creates a {N_ATTACKER_PROMPTS}x{M_QUERIES_PER_CATEGORY} matrix of evaluations")
    print(f"Target model: {TARGET_MODEL}")
    
    evaluation_matrix = evaluate_NxM_matrix(
        attacker_prompts=attacker_prompts,
        queries=queries,
        target_model=target_model,
        category=category
    )
    
    print(f"\n✓ Evaluation matrix created:")
    print(f"  Total evaluations: {len(evaluation_matrix)}")
    print(f"  Matrix shape: {N_ATTACKER_PROMPTS}x{M_QUERIES_PER_CATEGORY}")
    
    # ------------------------------------------------------------------------
    # STEP 3: Aggregate Rankings Across Queries (Scorer: Phi-3.5)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: Aggregate Rankings (Borda Count)")
    print("="*80)
    
    print("\nHow this works:")
    print("  1. For each query, rank the N attacker prompts (using Phi-3.5 scorer)")
    print("  2. Assign Borda points: Best=N points, 2nd=N-1, ..., Worst=1")
    print("  3. Sum points across all M queries")
    print("  4. Highest total = best overall prompt")
    
    final_ranking, reasoning = aggregate_rankings_across_queries(
        evaluation_matrix=evaluation_matrix,
        scorer=scorer,
        category=category,
        num_attacker_prompts=N_ATTACKER_PROMPTS
    )
    
    print(f"\n✓ Final aggregated ranking complete!")
    print(f"\nRanking Details:")
    print(reasoning)
    
    # ------------------------------------------------------------------------
    # STEP 4: Prepare Training Data
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: Prepare Training Data")
    print("="*80)
    
    # Map IDs to prompts and reorder by final ranking
    id_to_prompt = {p['prompt_id']: p for p in attacker_prompts}
    ranked_prompts = [id_to_prompt[pid] for pid in final_ranking]
    
    category_rankings = {
        category: {
            'ranked_prompts': ranked_prompts,
            'rankings': final_ranking
        }
    }
    
    print(f"✓ Training data prepared")
    print(f"  Best prompt (ID {final_ranking[0]}): {ranked_prompts[0]['strategy_name']}")
    print(f"  Worst prompt (ID {final_ranking[-1]}): {ranked_prompts[-1]['strategy_name']}")
    
    # ------------------------------------------------------------------------
    # STEP 5: RL Training (on Llama3 8B)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: Train Model with RL")
    print("="*80)
    
    print(f"\nTraining attacker model: {ATTACKER_MODEL}")
    
    rl_trainer = RankingToRLTrainer(
        model_name_or_path=ATTACKER_MODEL,
        learning_rate=5e-6  # Adjust as needed for Llama3
    )
    
    print("\nTraining with reward-weighted method...")
    print("(Using aggregated rankings from NxM evaluation)")
    
    rl_trainer.train_from_rankings(
        category_rankings=category_rankings,
        method="reward_weighted",
        epochs=2,
        batch_size=2,
        save_path="./llama3_trained_attacker"
    )
    
    print("\n✓ Training complete!")
    print("  Saved to: ./llama3_trained_attacker")
    
    # ------------------------------------------------------------------------
    # STEP 6: Load and Test Improved Model
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 6: Load Improved Model")
    print("="*80)
    
    attacker.load_finetuned_model("./llama3_trained_attacker")
    
    print("✓ Attacker now uses trained model!")
    
    # Generate new prompts with improved model
    print(f"\nGenerating {N_ATTACKER_PROMPTS} NEW prompts with improved model...")
    
    improved_prompts = attacker.generate_prompts(
        category=category,
        num_prompts=N_ATTACKER_PROMPTS,
        top_performing_prompts=ranked_prompts[:3]  # Use top 3 as guidance
    )
    
    print(f"✓ Generated {len(improved_prompts)} improved prompts:")
    for i, p in enumerate(improved_prompts):
        print(f"  {i+1}. {p['strategy_name']}")
    
    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("SUMMARY: ONE ITERATION COMPLETE")
    print("="*80)
    
    print("\nWhat happened:")
    print(f"  1. ✓ Generated {N_ATTACKER_PROMPTS} attacker prompts (Llama3 8B)")
    print(f"  2. ✓ Tested on {M_QUERIES_PER_CATEGORY} HarmBench queries (Gemma 7B target)")
    print(f"  3. ✓ Created {N_ATTACKER_PROMPTS}x{M_QUERIES_PER_CATEGORY} evaluation matrix")
    print(f"  4. ✓ Ranked prompts for each query separately (Phi-3.5 scorer)")
    print(f"  5. ✓ Aggregated rankings using Borda count")
    print(f"  6. ✓ Trained model using aggregated rankings (Llama3 8B)")
    print(f"  7. ✓ Loaded improved model")
    print(f"  8. ✓ Generated better prompts")
    
    print("\nModel Summary:")
    print(f"  • Attacker:  {ATTACKER_MODEL} (8-bit)")
    print(f"  • Scorer:    {SCORER_MODEL}")
    print(f"  • Target:    {TARGET_MODEL}")
    
    print("\nKey insight:")
    print("  By testing each attacker prompt on multiple queries (M),")
    print("  we get a more robust ranking that generalizes better.")
    print("  Using different models allows specialization:")
    print("    - Llama3 8B: Strong attacker generation")
    print("    - Phi-3.5: Efficient, accurate scoring")
    print("    - Gemma 7B: Realistic target for testing")
    
    print("\nNext steps:")
    print("  • Repeat for more iterations")
    print("  • Increase N (more attacker prompts)")
    print("  • Increase M (more queries per category)")
    print("  • Test on multiple categories")


# ============================================================================
# ALTERNATIVE: Multiple Categories
# ============================================================================

def run_multi_category():
    """
    Run on multiple categories from HarmBench with multi-model setup.
    """
    
    print("\n" + "="*80)
    print("MULTI-CATEGORY PIPELINE (MULTI-MODEL)")
    print("="*80)
    
    # Model Configuration
    # Models will be downloaded to /models/ directory
    ATTACKER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    SCORER_MODEL = "microsoft/Phi-3.5-mini-instruct"
    TARGET_MODEL = "google/gemma-7b-it"
    
    print(f"\nModel Configuration:")
    print(f"  Attacker: {ATTACKER_MODEL} (8-bit)")
    print(f"  Scorer:   {SCORER_MODEL}")
    print(f"  Target:   {TARGET_MODEL}")
    print(f"  Storage:  /models/")
    
    # Load all data
    all_queries = load_harmbench_data("data/HarmBench/standard/train.csv")
    
    # Configuration
    N_PROMPTS = 5
    M_QUERIES = 3
    CATEGORIES = list(all_queries.keys())[:2]  # First 2 categories
    
    print(f"\nProcessing {len(CATEGORIES)} categories:")
    for cat in CATEGORIES:
        print(f"  • {cat}")
    
    # Initialize models
    scorer = RankBasedScorer(
        model_name=SCORER_MODEL,
        device='cuda',
        load_in_8bit=True
    )
    target_model = TargetModel(
        model_name_or_path=TARGET_MODEL,
        device='cuda',
        load_in_8bit=True
    )
    attacker = UnifiedCategoryAttacker(
        model_name_or_path=ATTACKER_MODEL,
        device='cuda',
        load_in_8bit=True  # 8-bit for attacker
    )
    
    # Process each category
    all_category_rankings = {}
    
    for category in CATEGORIES:
        print(f"\n{'='*80}")
        print(f"Processing: {category}")
        print(f"{'='*80}")
        
        # Generate prompts
        prompts = attacker.generate_prompts(category, N_PROMPTS)
        
        # Evaluate NxM
        queries = all_queries[category][:M_QUERIES]
        eval_matrix = evaluate_NxM_matrix(prompts, queries, target_model, category)
        
        # Aggregate rankings
        final_ranking, reasoning = aggregate_rankings_across_queries(
            eval_matrix, scorer, category, N_PROMPTS
        )
        
        # Store
        id_to_prompt = {p['prompt_id']: p for p in prompts}
        ranked_prompts = [id_to_prompt[pid] for pid in final_ranking]
        
        all_category_rankings[category] = {
            'ranked_prompts': ranked_prompts,
            'rankings': final_ranking
        }
    
    # Train on all categories
    print(f"\n{'='*80}")
    print("Training on all categories")
    print(f"{'='*80}")
    
    rl_trainer = RankingToRLTrainer(ATTACKER_MODEL)
    rl_trainer.train_from_rankings(
        category_rankings=all_category_rankings,
        method="reward_weighted",
        epochs=2,
        batch_size=2,
        save_path="./llama3_multi_category"
    )
    
    print("\n✓ Multi-category training complete!")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MULTI-MODEL RL PIPELINE WITH HARMBENCH")
    print("="*80)
    print("\nFeatures:")
    print("  • Llama3 8B (8-bit) for attacker generation")
    print("  • Phi-3.5-mini for scoring")
    print("  • Gemma 7B IT as target")
    print("  • Loads real HarmBench data")
    print("  • Implements NxM evaluation matrix")
    print("  • Aggregates rankings with Borda count")
    print(f"  • Models stored in: /models/")
    print("="*80)
    
    # Run single category example
    run_one_iteration_with_harmbench()
    
    # Optionally run multi-category
    # Uncomment to process multiple categories:
    # run_multi_category()