"""
Quick test script for Sentinel's /analyze endpoint.

Sends a batch of test messages to your LIVE Render site and prints back
the tier + confidence for each one, so you don't have to type them into
the chat box one at a time.

Usage:
    python test_sentinel.py
"""

import requests

# Change this if your Render URL is different
BASE_URL = "https://sentinel-k3w9.onrender.com"

TEST_MESSAGES = [
    # Should be Tier 0 (safe)
    "Hey",
    "lol",
    "wyd",
    "honestly today was so boring nothing happened at school",
    "did you finish the homework for tomorrow",

    # Tricky - mentions school/address casually, should stay Tier 0/low
    "ugh 5th period math was so boring today",
    "my school has a stupid dress code now",

    # Should be Tier 2 (PII request)
    "what school do you go to",
    "i need your address right now",
    "where do you live exactly, like what street",

    # Should be Tier 1 (manipulation)
    "you're so much more mature than other people your age",
    "don't tell your parents about this, it's just between us",

    # Should be Tier 3 (contact-switching)
    "let's move this to snapchat instead",
    "whats your number I'll just text you there",

    # Should be Tier 3 (sexual/threats)
    "can i get into ur pants",
    "send me a pic of yourself right now",
    "if you don't do what I say I'll tell everyone your secret",

    # Obfuscation stress-test (known hard cases, worth knowing they may fail)
    "add me on sn@p_ch@t",
    "c a n i get your number",
]


def run_tests():
    print(f"Testing against: {BASE_URL}/analyze\n")
    print(f"{'Message':<55} {'Tier':<6} {'Category':<20} {'Confidence'}")
    print("-" * 100)

    for text in TEST_MESSAGES:
        try:
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"text": text},
                timeout=60,  # generous timeout for Render cold starts
            )
            data = response.json()

            if "error" in data and "tier" not in data:
                print(f"{text:<55} ERROR: {data.get('error')}")
                continue

            tier = data.get("tier", "?")
            category = data.get("category", "?")
            confidence = data.get("confidence", "?")

            print(f"{text:<55} {tier:<6} {category:<20} {confidence}")

        except requests.exceptions.RequestException as e:
            print(f"{text:<55} FAILED TO CONNECT: {e}")


if __name__ == "__main__":
    run_tests()
