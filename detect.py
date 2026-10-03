"""Score a message locally: model probability + human-readable reasons."""
import re
import sys

import joblib

RULES = [
    (r"https?://\S+", "Contains a link"),
    (r"bit\.ly|tinyurl|\.(xyz|top|vip|club|live|online|cc)\b", "Link uses a shortener or a suspicious domain"),
    (r"\b(urgent|immediately|final notice|today|within \d+ ?(hrs|hours)|expires?)\b", "Creates time pressure"),
    (r"\b(kyc|pan|aadhaar|account (will be )?(blocked|suspended)|sim)\b.*\b(update|link|verify|blocked|suspended)", "Threatens account or ID action"),
    (r"\botp\b.*\b(share|send|enter)|\b(share|send|enter)\b.*\b(otp|pin)\b", "Asks for an OTP or PIN"),
    (r"\b(processing fee|registration|clearance fee|deposit|pay rs)\b", "Asks for an upfront payment"),
    (r"\b(lottery|won|winner|free iphone|win iphone|congratulations)\b", "Prize or lottery bait"),
    (r"\b(anydesk|teamviewer|install app)\b", "Asks you to install remote-access or unknown apps"),
    (r"\b(arrest|police|cbi|case is registered)\b", "Impersonates authorities"),
]


def analyse(text, model):
    prob = float(model.predict_proba([text])[0][1])
    reasons = [why for pat, why in RULES if re.search(pat, text, re.I)]
    score = round(0.6 * prob + 0.4 * min(1.0, len(reasons) / 3), 2)
    verdict = "HIGH RISK" if score >= 0.6 else "SUSPICIOUS" if score >= 0.35 else "LOOKS SAFE"
    return score, verdict, reasons


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("Paste a message: ")
    score, verdict, reasons = analyse(text, joblib.load("model/model.joblib"))
    print(f"\nRisk score: {score}  ->  {verdict}")
    for r in reasons:
        print(f" - {r}")
