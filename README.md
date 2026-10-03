# ScamShield Circle

Private, on-device scam defence for communities. Built for the **iQOO Hackathon Grand Finale (Community App track)**.

## Problem
Fake KYC messages, UPI collect requests, loan-app harassment and impersonation calls hit students and first-time digital users hardest. Cloud scam filters need private messages uploaded, community warnings arrive late, and victims don't know how to report.

## Solution
A community app with **on-device AI at the core**:
1. **Detect** - a lightweight classifier scores SMS, links and transcripts locally and explains why.
2. **Share** - users post anonymised scam patterns (never raw messages) to local circles.
3. **Recover** - a guided first-hour flow: contact the bank, preserve evidence, draft a cybercrime complaint.

## What is in this repo (prototype status)
- `train.py` - trains a TF-IDF + logistic regression classifier and exports `model/model.json` (vocabulary, IDF, weights) for porting to TFLite / Android / browser.
- `detect.py` - CLI that combines the model probability with transparent rule-based reasons (urgency, suspicious links, OTP requests, upfront fees, authority impersonation).
- `data/sample_messages.csv` - a **small synthetic** sample so the pipeline runs out of the box.

Not built yet: the Android app, community feed and reporting guide (see roadmap).

## Quick start
```bash
pip install -r requirements.txt
python train.py
python detect.py "Your SBI account will be blocked today. Update KYC: http://bit.ly/x"
```

### Training on real data
Download a public dataset such as the UCI SMS Spam Collection (or a phishing-SMS dataset from Kaggle), save it as `data/dataset.csv` with columns `label,text` (the UCI `v1,v2` format also works), then run `python train.py` again. The bundled sample is far too small to measure real accuracy.

## Privacy by design
Detection runs locally. Only anonymised patterns (template text, fake domain, number prefix) are ever meant to be shared.

## Roadmap
- Phase 1: on-device detector (TFLite) and demo app
- Phase 2: community circles, trending feed, moderation
- Phase 3: Kannada/Hindi support, voice input, pilot with student groups

## Author
Mohammed Naheem - B.E. Data Science, Yenepoya Institute of Technology. President, DataNova Students Association.

## License
MIT
