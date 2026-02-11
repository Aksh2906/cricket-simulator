#!/bin/bash

echo "=== Cricket NLP Pipeline End-to-End Test ==="
echo ""
echo "1. Training models on example data..."
python3 training/train_models.py --data data/example_train.jsonl --out models/ 2>&1 | grep -E "(Speed MAE|line classifier|length classifier|swing classifier|Models saved)"

echo ""
echo "2. Inference Example 1: Death over, explicit speed"
python3 inference/run_inference.py --models models/ --text "Short and wide, batsman scoops, 137 kph" --over 19 2>&1 | grep -v "NotOpenSSLWarning" | grep -v "urllib3" | tail -20

echo ""
echo "3. Inference Example 2: Missing speed, power play"
python3 inference/run_inference.py --models models/ --text "Excellent yorker, digs it out, great delivery" --over 2 2>&1 | grep -v "NotOpenSSLWarning" | grep -v "urllib3" | tail -20

echo ""
echo "=== Pipeline Complete ==="
