#!/bin/bash

# Create a synthetic dataset with instruction-response pairs
cat <<EOF2 > data/instructions.json
[
    {"instruction": "Translate English to French: 'Hello, how are you?'", "response": "Bonjour, comment ça va?"},
    {"instruction": "Translate English to French: 'Good morning!'", "response": "Bonjour!"},
    {"instruction": "Summarize the following text: 'The sun rises in the east and sets in the west.'", "response": "The sun rises in the east and sets in the west."},
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "Write a Python function to reverse a string.", "response": "def reverse_string(s): return s[::-1]"}
]
EOF2
