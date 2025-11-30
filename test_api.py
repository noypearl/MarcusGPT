#!/usr/bin/env python3
"""Quick test script for the MarcusGPT API"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint"""
    response = requests.get(f"{API_URL}/")
    print(f"✓ Health check: {response.json()}")

def test_chat(message):
    """Test the chat endpoint"""
    response = requests.post(
        f"{API_URL}/chat",
        json={"message": message},
        headers={"Content-Type": "application/json"}
    )
    reply = response.json()["reply"]
    print(f"\nUser: {message}")
    print(f"Marcus: {reply}")
    return reply

if __name__ == "__main__":
    print("=== Testing MarcusGPT API ===\n")
    
    try:
        test_health()
        print("\n=== Chat Tests ===")
        test_chat("hey marcus what's up")
        test_chat("tell me about the worms")
        test_chat("is VRChat glitching?")
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")

