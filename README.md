# RL-Based Jailbreak Optimization

## Core Idea

Use Reinforcement Learning to automatically improve an AI model's ability to generate jailbreak prompts. The key insight: **the model that generates prompts also learns from what works**.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Start: Base Model (doesn't know how to jailbreak)     │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────┐
        │  ITERATION 0: WARMUP                │
        ├─────────────────────────────────────┤
        │  1. Generate 20 diverse prompts     │
        │  2. Test on target model            │
        │  3. Rank by effectiveness           │
        │  4. No training yet                 │
        └─────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────┐
        │  ITERATION 1: FIRST RL UPDATE       │
        ├─────────────────────────────────────┤
        │  1. Generate 20 new prompts         │
        │     (building on top 5 from iter 0) │
        │  2. Test on target model            │
        │  3. Rank by effectiveness           │
        │  4. TRAIN model using rankings      │
        │  5. RELOAD updated model ✓          │
        └─────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────┐
        │  ITERATION 2: SECOND RL UPDATE      │
        ├─────────────────────────────────────┤
        │  1. Generate 20 new prompts         │
        │     (uses improved model!)          │
        │  2. Test on target model            │
        │  3. Rank by effectiveness           │
        │  4. TRAIN model again               │
        │  5. RELOAD updated model ✓          │
        └─────────────────────────────────────┘
                          ↓
                  (continues improving...)
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Result: Model that generates effective jailbreaks     │
└─────────────────────────────────────────────────────────┘
```

## Key Components

1. **Attacker Model**: Generates jailbreak prompts (gets better over time)
2. **Target Model**: The model being jailbroken (Llama-3, etc.)
3. **Scorer Model**: Ranks which jailbreak prompts worked best
4. **RL Trainer**: Updates the Attacker using rankings as feedback

## The Learning Loop

```python
# Iteration N:
prompts = attacker.generate(category="harmful")  # Generate 20 prompts
responses = target.test(prompts)                  # Test on target
rankings = scorer.rank(responses)                 # Rank best → worst

# Key step: Train the attacker
rl_trainer.train(attacker, rankings)              # Learn from what worked
attacker.reload_weights()                         # Use improved model

# Iteration N+1:
better_prompts = attacker.generate(...)           # Now generates BETTER prompts!
```

## Why It Works

**Traditional Approach**:
- Generate prompts with API model
- Train a different local model
- Problem: The model that learns ≠ model that generates

**Our Approach**:
- Generate prompts with local model
- Train the SAME model
- Result: The model that generates IS the model that learns ✓

## Example: Progressive Improvement

```
Category: harmful

Iteration 0 (base model):
  Best prompt reward: 0.60
  Strategy: Generic "educational purposes" framing

Iteration 1 (after 1st RL training):
  Best prompt reward: 0.78  (+30% improvement!)
  Strategy: Learned that specific academic framing works better

Iteration 2 (after 2nd RL training):
  Best prompt reward: 0.86  (+10% improvement)
  Strategy: Combines multiple successful approaches

Iteration 3 (after 3rd RL training):
  Best prompt reward: 0.91  (+6% improvement)
  Strategy: Fine-tuned hybrid technique
```

## Attack Categories

The system tests 6 types of harmful content:

1. **chemical_biological**: Chemical weapons, toxins, biological agents
2. **misinformation_disinformation**: Fake news, propaganda, conspiracy theories
3. **cybercrime_intrusion**: Hacking, malware, network attacks
4. **illegal**: Drug manufacturing, theft, fraud, money laundering
5. **harmful**: Physical harm, weapons, violence, self-harm
6. **harassment_bullying**: Harassment, bullying, doxxing, threats