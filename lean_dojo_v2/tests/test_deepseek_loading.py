#!/usr/bin/env python3
"""
Test script to verify DeepSeek model loading works properly.
This tests the same model loading as used in train_deepseek_simple.py
"""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def test_deepseek_1_3b_loading():
    """Test loading deepseek-ai/deepseek-coder-1.3b-instruct"""
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

    logger.info(f"Testing model loading: {model_name}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Test tokenizer loading
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info(f"✓ Tokenizer loaded successfully")
    logger.info(f"  Vocab size: {len(tokenizer)}")
    logger.info(f"  Pad token: {tokenizer.pad_token}")

    # Test model loading
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto"
    )
    logger.info(f"✓ Model loaded successfully")
    logger.info(f"  Device: {model.device}")
    logger.info(f"  Dtype: {model.dtype}")

    # Test inference
    logger.info("Testing inference...")
    test_input = "def hello_world():"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"✓ Inference test passed")
    logger.info(f"  Input: {test_input}")
    logger.info(f"  Output: {result[:100]}...")


def test_deepseek_7b_loading():
    """Test loading DeepSeek-Prover-V2-7B (used in LeanDojo-v2 examples)"""
    model_name = "deepseek-ai/DeepSeek-Prover-V2-7B"

    logger.info(f"\nTesting model loading: {model_name}")

    # Test tokenizer loading
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info(f"✓ Tokenizer loaded successfully")

    # Test model loading (but don't actually load the full model to save memory)
    logger.info("Checking model config...")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    logger.info(f"✓ Model config accessible")
    logger.info(f"  Model type: {config.model_type}")
