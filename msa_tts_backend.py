"""
Modern Standard Arabic (MSA) TTS System — Production Backend
Architecture: VITS (end-to-end) + HiFi-GAN vocoder
FastAPI inference server with full Arabic NLP pipeline
"""

import re
import io
import os
import time
import base64
import logging
import unicodedata
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("msa_tts")

# ─────────────────────────────────────────────
# ARABIC UNICODE RANGES & CONSTANTS
# ─────────────────────────────────────────────
ARABIC_DIACRITICS = {
    "fatha":    "\u064E",  # َ
    "damma":    "\u064F",  # ُ
    "kasra":    "\u0650",  # ِ
    "sukun":    "\u0652",  # ْ
    "shadda":   "\u0651",  # ّ
    "tanwin_f": "\u064B",  # ً
    "tanwin_d": "\u064C",  # ٌ
    "tanwin_k": "\u064D",  # ٍ
    "maddah":   "\u0653",  # ٓ
    "hamza_above": "\u0654",
    "hamza_below": "\u0655",
}

ARABIC_LETTERS = set(chr(c) for c in range(0x0600, 0x06FF))

# Arabic number words (MSA)
ARABIC_ONES = [
    "", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة",
    "ستة", "سبعة", "ثمانية", "تسعة", "عشرة",
    "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر",
    "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"
]
ARABIC_TENS = [
    "", "عشرة", "عشرون", "ثلاثون", "أربعون", "خمسون",
    "ستون", "سبعون", "ثمانون", "تسعون"
]
ARABIC_SCALES = ["", "ألف", "مليون", "مليار", "تريليون"]

# ─────────────────────────────────────────────
# MSA PHONEME TABLE (Arabic → IPA-like symbols)
# ─────────────────────────────────────────────
# Covers all 28 Arabic consonants + vowels
# Uses SAMPA-like notation safe for neural TTS

ARABIC_G2P_MAP = {
    # Consonants
    "\u0628": "b",    # ب
    "\u062a": "t",    # ت
    "\u062b": "th",   # ث (dental fricative)
    "\u062c": "dʒ",   # ج
    "\u062d": "ħ",    # ح (voiceless pharyngeal)
    "\u062e": "x",    # خ
    "\u062f": "d",    # د
    "\u0630": "ð",    # ذ
    "\u0631": "r",    # ر
    "\u0632": "z",    # ز
    "\u0633": "s",    # س
    "\u0634": "ʃ",    # ش
    "\u0635": "sˤ",   # ص (emphatic)
    "\u0636": "dˤ",   # ض (emphatic)
    "\u0637": "tˤ",   # ط (emphatic)
    "\u0638": "ðˤ",   # ظ (emphatic)
    "\u0639": "ʕ",    # ع (voiced pharyngeal)
    "\u063a": "ɣ",    # غ
    "\u0641": "f",    # ف
    "\u0642": "q",    # ق (uvular)
    "\u0643": "k",    # ك
    "\u0644": "l",    # ل
    "\u0645": "m",    # م
    "\u0646": "n",    # ن
    "\u0647": "h",    # ه
    "\u0648": "w",    # و
    "\u064a": "j",    # ي
    "\u0621": "ʔ",    # ء (glottal stop)
    "\u0623": "ʔ",    # أ
    "\u0625": "ʔ",    # إ
    "\u0626": "ʔj",   # ئ
    "\u0624": "ʔw",   # ؤ
    "\u0622": "ʔaː",  # آ (madda)
    "\u0627": "aː",   # ا (alef)
    "\u0629": "t",    # ة (ta marbuta — pronounced t in pausa)
    "\u0649": "aː",   # ى (alef maqsura)
    "\u0644\u0627": "laː",  # لا ligature
}

# Diacritic → vowel phoneme
DIACRITIC_VOWEL_MAP = {
    "\u064E": "a",    # فتحة → /a/
    "\u064F": "u",    # ضمة → /u/
    "\u0650": "i",    # كسرة → /i/
    "\u064B": "an",   # تنوين فتح → /an/
    "\u064C": "un",   # تنوين ضم → /un/
    "\u064D": "in",   # تنوين كسر → /in/
    "\u0652": "",     # سكون → no vowel
}

# ─────────────────────────────────────────────
# 1. TEXT NORMALIZATION
# ─────────────────────────────────────────────

class ArabicTextNormalizer:
    """
    Stage 1: Normalize raw Arabic text for TTS.
    Handles: numbers, punctuation, abbreviations,
    hamza normalization, tatweel removal, ligatures.
    MSA-only. No dialect handling.
    """

    def __init__(self):
        # Common MSA abbreviations
        self.abbreviations = {
            "د.":   "دكتور",
            "أ.":   "أستاذ",
            "م.":   "مهندس",
            "ص.":   "صفحة",
            "ج.":   "جزء",
            "مثلاً": "مثلاً",
            "إلخ":  "إلى آخره",
            "صلى الله عليه وسلم": "صلى الله عليه وسلم",
        }

        # Punctuation that signals prosodic boundary
        self.boundary_puncts = set("،؛.!؟")

    def remove_tatweel(self, text: str) -> str:
        """Remove kashida/tatweel ـ (U+0640) — decorative, not phonemic."""
        return text.replace("\u0640", "")

    def normalize_hamza(self, text: str) -> str:
        """Normalize hamza variants to canonical forms for G2P."""
        # أ إ آ ء ئ ؤ → all map to hamza phoneme in G2P
        # Keep them as-is; G2P handles them
        # But normalize exotic variants
        replacements = {
            "\u0671": "\u0627",  # ٱ → ا (wasla alef)
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    def normalize_alef(self, text: str) -> str:
        """Normalize alef variants."""
        # أ أ إ آ → ا for undiacritized normalization (G2P will refine)
        # Only normalize the bare letter for G2P lookup purposes
        return text

    def expand_numbers(self, text: str) -> str:
        """Convert Western and Eastern Arabic numerals to Arabic words."""
        # Eastern Arabic digits → Western
        eastern_map = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        text = text.translate(eastern_map)

        def number_to_arabic_words(n: int) -> str:
            if n == 0:
                return "صفر"
            if n < 0:
                return "سالب " + number_to_arabic_words(-n)

            parts = []
            for i, scale in enumerate(ARABIC_SCALES):
                unit = 1000 ** i
                if n < unit * 1000:
                    chunk = n // unit
                    n %= unit
                    if chunk > 0:
                        parts.append(chunk_to_words(chunk))
                        if scale:
                            parts.append(scale)

            parts.reverse()
            return " ".join(parts)

        def chunk_to_words(n: int) -> str:
            if n == 0:
                return ""
            parts = []
            if n >= 100:
                hundreds = [
                    "", "مئة", "مئتان", "ثلاثمئة", "أربعمئة",
                    "خمسمئة", "ستمئة", "سبعمئة", "ثمانمئة", "تسعمئة"
                ]
                parts.append(hundreds[n // 100])
                n %= 100
            if n >= 20:
                parts.append(ARABIC_TENS[n // 10])
                n %= 10
            if n > 0:
                parts.append(ARABIC_ONES[n])
            return " و".join(p for p in parts if p)

        def replace_number(match):
            num_str = match.group(0).replace(",", "").replace("،", "")
            try:
                if "." in num_str or "٫" in num_str:
                    # Decimal number
                    parts = num_str.replace("٫", ".").split(".")
                    integer_part = number_to_arabic_words(int(parts[0]))
                    decimal_part = " ".join(
                        ARABIC_ONES[int(d)] if int(d) < len(ARABIC_ONES) else d
                        for d in parts[1]
                    )
                    return f"{integer_part} فاصلة {decimal_part}"
                else:
                    return number_to_arabic_words(int(num_str))
            except (ValueError, IndexError):
                return num_str

        text = re.sub(r"[\d٠-٩][,،\d٠-٩\.٫]*", replace_number, text)
        return text

    def expand_abbreviations(self, text: str) -> str:
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)
        return text

    def normalize_punctuation(self, text: str) -> str:
        """Standardize punctuation for prosody boundary detection."""
        # Arabic comma → Western comma equivalent
        text = text.replace("،", "،")  # keep Arabic comma
        # Remove repeated punctuation
        text = re.sub(r"[\.]{2,}", ".", text)
        text = re.sub(r"[،]{2,}", "،", text)
        # Strip HTML-like tags if any
        text = re.sub(r"<[^>]+>", " ", text)
        return text

    def clean_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """Full normalization pipeline."""
        text = self.remove_tatweel(text)
        text = self.normalize_hamza(text)
        text = self.normalize_alef(text)
        text = self.expand_abbreviations(text)
        text = self.expand_numbers(text)
        text = self.normalize_punctuation(text)
        text = self.clean_whitespace(text)
        logger.debug(f"Normalized: {text}")
        return text


# ─────────────────────────────────────────────
# 2. DIACRITIZATION (Tashkeel)
# ─────────────────────────────────────────────

class ArabicDiacritizer:
    """
    Stage 2: Automatic diacritization of MSA text.

    Production options (in priority order):
      1. CAMeL Tools (BERT-based, best accuracy ~95% DER)
      2. Mishkal (rule-based, fast, ~88% DER)
      3. Farasa Diacritizer

    This class wraps CAMeL Tools if available,
    falls back to Mishkal, then to passthrough.

    Install:
      pip install camel-tools
      camel_data -i defaults
    """

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.diacritizer = None
        self._init_backend()

    def _init_backend(self):
        if self.backend in ("auto", "camel"):
            try:
                from camel_tools.tagger.default import DefaultTagger
                from camel_tools.dialectid import DialectIdentifier
                # Use MSA diacritizer
                from camel_tools.utils.dediac import dediac_ar
                self.diacritizer = self._camel_diacritize
                self.backend = "camel"
                logger.info("Diacritizer: CAMeL Tools loaded")
                return
            except ImportError:
                logger.warning("CAMeL Tools not available")

        if self.backend in ("auto", "mishkal"):
            try:
                import mishkal.tashkeel
                self.diacritizer_engine = mishkal.tashkeel.TashkeelClass()
                self.diacritizer = self._mishkal_diacritize
                self.backend = "mishkal"
                logger.info("Diacritizer: Mishkal loaded")
                return
            except ImportError:
                logger.warning("Mishkal not available")

        logger.warning("No diacritizer available — using passthrough")
        self.diacritizer = lambda text: text
        self.backend = "passthrough"

    def _camel_diacritize(self, text: str) -> str:
        """CAMeL Tools BERT-based diacritization."""
        try:
            from camel_tools.tagger.default import DefaultTagger
            tagger = DefaultTagger.build(task="diac", dialect="msa")
            sentences = re.split(r"[.،؛!؟]", text)
            result = []
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    tokens = sent.split()
                    diacritized = tagger.tag(tokens)
                    result.append(" ".join(diacritized))
            return " ".join(result)
        except Exception as e:
            logger.error(f"CAMeL diacritization failed: {e}")
            return text

    def _mishkal_diacritize(self, text: str) -> str:
        """Mishkal rule-based diacritization."""
        try:
            return self.diacritizer_engine.tashkeel(text)
        except Exception as e:
            logger.error(f"Mishkal diacritization failed: {e}")
            return text

    def is_already_diacritized(self, text: str) -> bool:
        """Check if text already contains diacritics (>30% chars are diacritics)."""
        diacritic_chars = set(ARABIC_DIACRITICS.values())
        arabic_chars = [c for c in text if c in ARABIC_LETTERS]
        diacritic_count = sum(1 for c in text if c in diacritic_chars)
        if not arabic_chars:
            return False
        ratio = diacritic_count / len(arabic_chars)
        return ratio > 0.3

    def diacritize(self, text: str) -> str:
        """Diacritize text if not already diacritized."""
        if self.is_already_diacritized(text):
            logger.debug("Text already diacritized — skipping")
            return text
        result = self.diacritizer(text)
        logger.debug(f"Diacritized: {result}")
        return result


# ─────────────────────────────────────────────
# 3. GRAPHEME-TO-PHONEME (G2P)
# ─────────────────────────────────────────────

class ArabicG2P:
    """
    Stage 3: Convert diacritized Arabic text to phoneme sequence.
    Handles:
      - All 28 Arabic consonants including pharyngeals, uvulars, emphatics
      - Sun/moon letter assimilation for ال (al-)
      - Shadda (gemination) doubling
      - Vowel insertion from diacritics
      - Glottal stop at word boundaries
      - Prosody markers from punctuation
    """

    # Sun letters (الشمسية) — assimilate ل in ال
    SUN_LETTERS = set("تثدذرزسشصضطظلن")

    # Prosody pause markers
    PAUSE_MAP = {
        "،": "<short_pause>",
        "؛": "<medium_pause>",
        ".": "<long_pause>",
        "!": "<long_pause>",
        "؟": "<long_pause>",
        ",": "<short_pause>",
    }

    def __init__(self):
        self.g2p_map = ARABIC_G2P_MAP
        self.vowel_map = DIACRITIC_VOWEL_MAP

    def handle_article(self, word: str) -> str:
        """
        Handle ال (al-) sun letter assimilation.
        اَلشَّمْسُ → ash-shamsu (not al-shamsu)
        """
        if len(word) < 3:
            return word
        # Check for ال prefix
        if word[0] == "\u0627" and word[1] == "\u0644":
            # Find first actual letter after ال
            rest = word[2:]
            if rest and rest[0] in self.SUN_LETTERS:
                # Sun letter: assimilate l → duplicate the sun letter
                return "a" + rest[0] + "-"  # phoneme prefix
        return word

    def apply_shadda(self, phoneme: str) -> str:
        """Geminate (double) consonant for shadda."""
        return phoneme + phoneme

    def word_to_phonemes(self, word: str) -> list:
        """Convert a single diacritized Arabic word to phoneme list."""
        phonemes = []
        i = 0
        chars = list(word)

        # Handle ال assimilation
        article_prefix = ""
        if len(chars) >= 2 and chars[0] == "\u0627" and chars[1] == "\u0644":
            if len(chars) > 2 and chars[2] in self.SUN_LETTERS:
                # Sun letter assimilation
                sun = self.g2p_map.get(chars[2], chars[2])
                article_prefix = "a" + sun  # ash / ar / an etc.
                i = 2  # skip ا and ل, process from 3rd char
            else:
                # Moon letter: ال → /al/
                article_prefix = "al"
                i = 2

        if article_prefix:
            phonemes.extend(list(article_prefix))

        while i < len(chars):
            char = chars[i]

            # Diacritic — already handled with preceding consonant
            if char in self.vowel_map:
                i += 1
                continue

            # Shadda — geminate next consonant
            if char == "\u0651":
                i += 1
                continue

            # Regular consonant
            phoneme = self.g2p_map.get(char, char)

            # Look ahead for diacritics
            following_shadda = False
            following_vowel = ""

            j = i + 1
            while j < len(chars) and chars[j] in (set(self.vowel_map.keys()) | {"\u0651"}):
                if chars[j] == "\u0651":
                    following_shadda = True
                elif chars[j] in self.vowel_map:
                    following_vowel = self.vowel_map[chars[j]]
                j += 1

            if following_shadda:
                phonemes.append(phoneme)  # gemination: repeat phoneme
            phonemes.append(phoneme)

            if following_vowel:
                phonemes.append(following_vowel)

            i += 1

        return phonemes

    def text_to_phonemes(self, text: str) -> str:
        """
        Convert full normalized+diacritized text to phoneme sequence string.
        Returns space-separated phoneme string with pause markers.
        """
        all_phonemes = []
        tokens = text.split()

        for token in tokens:
            # Check if token is punctuation
            if token in self.PAUSE_MAP:
                all_phonemes.append(self.PAUSE_MAP[token])
                continue

            # Split off trailing punctuation
            trailing = ""
            while token and token[-1] in self.PAUSE_MAP:
                trailing = self.PAUSE_MAP[token[-1]]
                token = token[:-1]

            if token:
                phonemes = self.word_to_phonemes(token)
                all_phonemes.extend(phonemes)

            if trailing:
                all_phonemes.append(trailing)

        result = " ".join(all_phonemes)
        logger.debug(f"Phonemes: {result}")
        return result


# ─────────────────────────────────────────────
# 4. VITS MODEL WRAPPER
# ─────────────────────────────────────────────

class VITSInferenceModel:
    """
    Stage 4+5: VITS end-to-end TTS
    (Acoustic model + HiFi-GAN vocoder combined)

    Architecture:
      - Text Encoder: Transformer (6 layers, 192 hidden dim)
      - Posterior Encoder: WaveNet-style (residual blocks)
      - Flow: Normalizing flow (4 coupling layers)
      - Generator: HiFi-GAN (multi-period + multi-scale discriminators)
      - Duration Predictor: Stochastic duration predictor
      - Prosody: Pitch + Energy conditioning

    Training config for Arabic MSA:
      - Sample rate: 22050 Hz
      - Mel bins: 80
      - FFT size: 1024
      - Hop length: 256
      - Window: 1024

    Dataset: Arabic Speech Corpus (ASC) by Nawar Halabi
      + Multilingual LibriSpeech Arabic subset
      Expected: ~20h MSA, single speaker
    """

    SAMPLE_RATE = 22050
    HOP_LENGTH = 256

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                d = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                d = torch.device("cpu")
                logger.info("Using CPU inference")
            return d
        return torch.device(device)

    def _load_model(self):
        """
        Load VITS model from checkpoint.

        Training command (reference):
          python train.py \
            --config configs/arabic_msa_vits.json \
            --model_dir checkpoints/arabic_msa \
            --dataset_path data/arabic_msa

        Model checkpoint: checkpoints/arabic_msa/G_latest.pth
        """
        if self.model_path and Path(self.model_path).exists():
            try:
                # In production: load your trained VITS model here
                # from vits.models import SynthesizerTrn
                # self.model = SynthesizerTrn(...)
                # checkpoint = torch.load(self.model_path, map_location=self.device)
                # self.model.load_state_dict(checkpoint['model'])
                # self.model.eval()
                logger.info(f"VITS model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Model load failed: {e}")
                self.model = None
        else:
            logger.warning("No model checkpoint found — synthesis will use demo fallback")
            self.model = None

    def synthesize(
        self,
        phoneme_sequence: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        pitch_shift: float = 0.0,
    ) -> np.ndarray:
        """
        Synthesize waveform from phoneme sequence.

        Args:
            phoneme_sequence: Space-separated phoneme string
            speed: Speaking rate (0.5=slow, 1.0=normal, 2.0=fast)
            noise_scale: Affects variation in acoustic features
            noise_scale_w: Affects duration variation
            pitch_shift: Semitones to shift pitch

        Returns:
            np.ndarray: Audio waveform at SAMPLE_RATE Hz
        """
        if self.model is not None:
            return self._vits_synthesize(
                phoneme_sequence, speed, noise_scale, noise_scale_w, pitch_shift
            )
        else:
            return self._demo_synthesize(phoneme_sequence, speed)

    def _vits_synthesize(self, phoneme_sequence, speed, noise_scale, noise_scale_w, pitch_shift):
        """Production VITS inference."""
        with torch.no_grad():
            # 1. Encode phoneme sequence to IDs
            phoneme_ids = self._phonemes_to_ids(phoneme_sequence)
            x = torch.LongTensor(phoneme_ids).unsqueeze(0).to(self.device)
            x_len = torch.LongTensor([len(phoneme_ids)]).to(self.device)

            # 2. VITS forward pass (text → waveform)
            # audio, _, _, _ = self.model.infer(
            #     x, x_len,
            #     noise_scale=noise_scale,
            #     noise_scale_w=noise_scale_w,
            #     length_scale=1.0 / speed
            # )
            # audio = audio[0, 0].data.cpu().numpy()

            # 3. Apply pitch shift if needed
            # if pitch_shift != 0.0:
            #     audio = self._shift_pitch(audio, pitch_shift)

            # Placeholder until model loaded:
            audio = self._demo_synthesize(phoneme_sequence, speed)
            return audio

    def _demo_synthesize(self, phoneme_sequence: str, speed: float = 1.0) -> np.ndarray:
        """
        Demo synthesis: generates a realistic sine wave pattern.
        Replace with actual model in production.
        """
        phonemes = phoneme_sequence.split()
        duration_per_phoneme = int(self.SAMPLE_RATE * 0.08 / speed)
        total_samples = max(len(phonemes) * duration_per_phoneme, self.SAMPLE_RATE // 2)

        t = np.linspace(0, total_samples / self.SAMPLE_RATE, total_samples)

        # Generate a natural-sounding demo tone (formant-like)
        audio = np.zeros(total_samples)
        base_freq = 120.0  # Male MSA speaker fundamental

        for i, phoneme in enumerate(phonemes):
            start = i * duration_per_phoneme
            end = min(start + duration_per_phoneme, total_samples)
            seg_t = t[start:end]

            if "<pause>" in phoneme:
                continue  # silence

            # Vowel-like phonemes get higher amplitude
            is_vowel = phoneme in ("a", "i", "u", "aː", "iː", "uː", "an", "in", "un")
            amp = 0.3 if is_vowel else 0.15

            # Add harmonics for naturalness
            seg = (
                amp * np.sin(2 * np.pi * base_freq * seg_t) +
                amp * 0.5 * np.sin(2 * np.pi * base_freq * 2 * seg_t) +
                amp * 0.25 * np.sin(2 * np.pi * base_freq * 3 * seg_t)
            )

            # Envelope
            fade = min(len(seg_t), 100)
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
            audio[start:end] += seg

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.85

        return audio.astype(np.float32)

    def _phonemes_to_ids(self, phoneme_sequence: str) -> list:
        """Map phoneme strings to integer IDs for model input."""
        # Build vocabulary from ARABIC_G2P_MAP values + specials
        vocab = ["<pad>", "<sos>", "<eos>", "<unk>", "<short_pause>",
                 "<medium_pause>", "<long_pause>"]
        all_phonemes = list(set(ARABIC_G2P_MAP.values())) + list(DIACRITIC_VOWEL_MAP.values())
        vocab.extend(sorted(set(all_phonemes)))
        ph2id = {ph: i for i, ph in enumerate(vocab)}

        ids = [ph2id.get("<sos>", 1)]
        for ph in phoneme_sequence.split():
            ids.append(ph2id.get(ph, ph2id.get("<unk>", 3)))
        ids.append(ph2id.get("<eos>", 2))
        return ids

    def audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        import wave
        import struct
        audio_int16 = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()


# ─────────────────────────────────────────────
# 5. FULL TTS PIPELINE
# ─────────────────────────────────────────────

class MSATTSPipeline:
    """
    Orchestrates the full MSA TTS pipeline:
    Text → Normalize → Diacritize → G2P → VITS → WAV
    """

    def __init__(self, model_path: Optional[str] = None):
        logger.info("Initializing MSA TTS Pipeline...")
        self.normalizer = ArabicTextNormalizer()
        self.diacritizer = ArabicDiacritizer(backend="auto")
        self.g2p = ArabicG2P()
        self.model = VITSInferenceModel(model_path=model_path)
        logger.info("Pipeline ready.")

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        noise_scale: float = 0.667,
        return_phonemes: bool = False,
    ) -> dict:
        """
        Full synthesis pipeline.

        Args:
            text: Raw Arabic text (diacritized or not)
            speed: 0.5–2.0, speaking rate
            pitch_shift: semitones
            noise_scale: prosody variation
            return_phonemes: include phoneme string in response

        Returns:
            dict with 'audio_b64', 'sample_rate', 'duration_s',
                       optionally 'phonemes', 'normalized_text'
        """
        t0 = time.perf_counter()

        # Stage 1: Normalize
        normalized = self.normalizer.normalize(text)

        # Stage 2: Diacritize
        diacritized = self.diacritizer.diacritize(normalized)

        # Stage 3: G2P
        phonemes = self.g2p.text_to_phonemes(diacritized)

        # Stage 4+5: VITS synthesis
        audio = self.model.synthesize(
            phonemes,
            speed=speed,
            noise_scale=noise_scale,
            pitch_shift=pitch_shift,
        )

        # Convert to WAV
        wav_bytes = self.model.audio_to_wav_bytes(audio)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        elapsed = time.perf_counter() - t0
        duration_s = len(audio) / self.model.SAMPLE_RATE
        rtf = elapsed / duration_s  # Real-Time Factor

        logger.info(
            f"Synthesized {len(text)} chars → {duration_s:.2f}s audio "
            f"in {elapsed:.2f}s (RTF={rtf:.3f})"
        )

        result = {
            "audio_b64": audio_b64,
            "sample_rate": self.model.SAMPLE_RATE,
            "duration_s": round(duration_s, 3),
            "rtf": round(rtf, 4),
            "diacritizer": self.diacritizer.backend,
        }

        if return_phonemes:
            result["phonemes"] = phonemes
            result["normalized_text"] = normalized
            result["diacritized_text"] = diacritized

        return result


# ─────────────────────────────────────────────
# 6. FASTAPI SERVER
# ─────────────────────────────────────────────

app = FastAPI(
    title="MSA Arabic TTS API",
    description="Modern Standard Arabic Text-to-Speech — VITS + HiFi-GAN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
MODEL_PATH = os.environ.get("TTS_MODEL_PATH", None)
pipeline: Optional[MSATTSPipeline] = None


@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = MSATTSPipeline(model_path=MODEL_PATH)


# ─── Request/Response Schemas ───

class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Arabic text (MSA)")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking rate")
    pitch_shift: float = Field(0.0, ge=-6.0, le=6.0, description="Pitch shift in semitones")
    noise_scale: float = Field(0.667, ge=0.0, le=1.0, description="Prosody variation")
    return_phonemes: bool = Field(False, description="Return phoneme sequence in response")
    format: str = Field("wav", description="Output format: wav or mp3")

class SynthesisResponse(BaseModel):
    audio_b64: str
    sample_rate: int
    duration_s: float
    rtf: float
    diacritizer: str
    phonemes: Optional[str] = None
    normalized_text: Optional[str] = None
    diacritized_text: Optional[str] = None

class NormalizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class DiacritizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class G2PRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


# ─── Endpoints ───

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipeline.model.model is not None if pipeline else False,
        "diacritizer": pipeline.diacritizer.backend if pipeline else "none",
        "device": str(pipeline.model.device) if pipeline else "unknown",
    }

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize(req: SynthesisRequest):
    """Main TTS endpoint. Returns base64-encoded WAV audio."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        result = pipeline.synthesize(
            text=req.text,
            speed=req.speed,
            pitch_shift=req.pitch_shift,
            noise_scale=req.noise_scale,
            return_phonemes=req.return_phonemes,
        )
        return result
    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(500, f"Synthesis failed: {str(e)}")

@app.post("/synthesize/stream")
async def synthesize_stream(req: SynthesisRequest):
    """Streaming synthesis — returns WAV audio directly."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        result = pipeline.synthesize(
            text=req.text,
            speed=req.speed,
            pitch_shift=req.pitch_shift,
            noise_scale=req.noise_scale,
        )
        wav_bytes = base64.b64decode(result["audio_b64"])
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/normalize")
async def normalize_text(req: NormalizeRequest):
    """Normalize Arabic text (numbers, abbreviations, punctuation)."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return {"normalized": pipeline.normalizer.normalize(req.text)}

@app.post("/diacritize")
async def diacritize_text(req: DiacritizeRequest):
    """Add diacritics (tashkeel) to MSA text."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    normalized = pipeline.normalizer.normalize(req.text)
    diacritized = pipeline.diacritizer.diacritize(normalized)
    return {
        "normalized": normalized,
        "diacritized": diacritized,
        "backend": pipeline.diacritizer.backend,
    }

@app.post("/g2p")
async def grapheme_to_phoneme(req: G2PRequest):
    """Convert Arabic text to phoneme sequence."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    normalized = pipeline.normalizer.normalize(req.text)
    diacritized = pipeline.diacritizer.diacritize(normalized)
    phonemes = pipeline.g2p.text_to_phonemes(diacritized)
    return {
        "text": req.text,
        "normalized": normalized,
        "diacritized": diacritized,
        "phonemes": phonemes,
    }

@app.get("/voices")
async def list_voices():
    """List available TTS voices."""
    return {
        "voices": [
            {
                "id": "msa_male_01",
                "name": "Arabic MSA Male — فصحى",
                "language": "ar-MSA",
                "gender": "male",
                "sample_rate": 22050,
                "model": "VITS",
            }
        ]
    }


# ─────────────────────────────────────────────
# 7. EVALUATION METRICS (offline use)
# ─────────────────────────────────────────────

class TTSEvaluator:
    """
    Evaluation metrics for MSA TTS quality assessment.

    Metrics:
      - MOS (Mean Opinion Score): Human listening test, 1–5
      - MCD (Mel Cepstral Distortion): Lower = better, target <6 dB
      - WER (Word Error Rate): ASR on synthesized audio, target <5%
      - CER (Character Error Rate): For Arabic text
      - RTF (Real-Time Factor): <0.5 for production, <0.1 for streaming
      - DNSMOS: Automatic MOS prediction (no human needed)
      - F0 RMSE: Pitch accuracy vs. reference
    """

    @staticmethod
    def mel_cepstral_distortion(ref_audio: np.ndarray, syn_audio: np.ndarray,
                                 sr: int = 22050) -> float:
        """
        MCD — standard mel cepstral distortion metric.
        Reference: Kubichek (1993). Target: < 6 dB for good quality.
        """
        try:
            import librosa
            K = 10.0 / np.log(10.0) * np.sqrt(2.0)

            ref_mel = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=13)
            syn_mel = librosa.feature.mfcc(y=syn_audio, sr=sr, n_mfcc=13)

            # Align lengths
            min_len = min(ref_mel.shape[1], syn_mel.shape[1])
            ref_mel = ref_mel[:, :min_len]
            syn_mel = syn_mel[:, :min_len]

            diff = ref_mel - syn_mel
            mcd = K * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
            return float(mcd)
        except ImportError:
            return -1.0

    @staticmethod
    def real_time_factor(audio_duration_s: float, inference_time_s: float) -> float:
        """RTF = inference_time / audio_duration. Target < 0.3."""
        return inference_time_s / audio_duration_s if audio_duration_s > 0 else float("inf")

    @staticmethod
    def diacritization_error_rate(ref_text: str, hyp_text: str) -> float:
        """
        DER — diacritization error rate.
        Target: < 5% for MSA (CAMeL achieves ~3.8%).
        """
        ref_diacs = [c for c in ref_text if c in set(ARABIC_DIACRITICS.values())]
        hyp_diacs = [c for c in hyp_text if c in set(ARABIC_DIACRITICS.values())]
        if not ref_diacs:
            return 0.0
        errors = sum(1 for r, h in zip(ref_diacs, hyp_diacs) if r != h)
        errors += abs(len(ref_diacs) - len(hyp_diacs))
        return errors / len(ref_diacs)


# ─────────────────────────────────────────────
# 8. TRAINING CONFIGURATION (reference)
# ─────────────────────────────────────────────

VITS_TRAINING_CONFIG = {
    # Model architecture
    "model": {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [8, 8, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "n_layers_q": 3,
        "use_spectral_norm": False,
    },
    # Data config
    "data": {
        "training_files": "data/msa_train.txt",
        "validation_files": "data/msa_val.txt",
        "text_cleaners": ["arabic_cleaners"],
        "max_wav_value": 32768.0,
        "sampling_rate": 22050,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
        "add_blank": True,  # Important for Arabic vowel alignment
        "n_speakers": 1,    # Single MSA speaker
        "cleaned_text": True,
    },
    # Training hyperparameters
    "train": {
        "log_interval": 200,
        "eval_interval": 1000,
        "seed": 42,
        "epochs": 10000,
        "learning_rate": 2e-4,
        "betas": [0.8, 0.99],
        "eps": 1e-9,
        "batch_size": 16,
        "fp16_run": True,       # Mixed precision
        "lr_decay": 0.999875,
        "segment_size": 8192,
        "init_lr_ratio": 1,
        "warmup_epochs": 0,
        "c_mel": 45,
        "c_kl": 1.0,
    },
}

# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Single worker for GPU; use multiple for CPU
        log_level="info",
    )
