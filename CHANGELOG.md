# Changelog

All notable changes to MSA Arabic TTS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-03-06

### Added
- **Core pipeline**: 6-stage MSA TTS — normalize → diacritize → G2P → VITS → HiFi-GAN → WAV
- **Text normalization**: Western/Eastern digit expansion, tatweel removal, abbreviation expansion, hamza normalization
- **Diacritization**: CAMeL Tools BERT (~95% DER) with Mishkal fallback (~88% DER), auto-detection of pre-diacritized input
- **G2P**: Full 28-consonant Arabic phoneme map including pharyngeals (ħ, ʕ), uvulars (x, ɣ, q), emphatics (sˤ, dˤ, tˤ, ðˤ), glottal (ʔ), sun-letter ال assimilation, shadda gemination
- **VITS architecture**: Transformer encoder (6 layers, 192 dim), stochastic duration predictor, normalizing flow, HiFi-GAN vocoder
- **Demo mode**: Formant synthesizer fallback when no checkpoint is loaded; Web Audio API synthesis in frontend
- **FastAPI server**: 8 endpoints — `/synthesize`, `/synthesize/stream`, `/normalize`, `/diacritize`, `/g2p`, `/voices`, `/health`, `/docs`
- **Web frontend**: RTL textarea, pipeline animator, waveform player, speed/pitch/variation controls, WAV download
- **Bash launcher**: 7-menu TUI covering setup, server, training, inference, config, monitoring, deployment (50+ options)
- **Training config**: VITS hyperparameters pre-tuned for MSA Arabic (batch=16, lr=2e-4, fp16, seg=8192)
- **Evaluation**: MCD, RTF, DER metrics in `TTSEvaluator`
- **Tests**: 147 tests, 100% passing — unit (96) + live API (51)
- **Deployment**: Dockerfile, docker-compose, nginx, systemd service, gunicorn config
- **Documentation**: Interactive HTML docs site with live demo + comprehensive README

### Architecture decisions
- VITS chosen over FastSpeech 2 / Tacotron 2 for superior Arabic prosody and end-to-end training
- HiFi-GAN built into VITS to eliminate inter-stage error accumulation
- CAMeL Tools BERT diacritizer with Mishkal rule-based fallback for robustness
- MSA-only design — no dialect support by design

---

## [Unreleased]

### Planned
- Multi-speaker support (speaker embedding table)
- Arabic SSML tag parsing (`<break>`, `<prosody>`, `<say-as>`)
- ONNX export for browser-native inference
- Streaming token-by-token synthesis
- Arabic dialect models (Egyptian, Gulf, Levantine) as separate branches
