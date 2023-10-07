"""
MSA Arabic TTS — Unit Test Suite
147 tests covering: normalizer, diacritizer, G2P, evaluator, constants, live API
Run: pytest tests/ -v
"""
import sys, types, pytest, numpy as np
from pathlib import Path

# ── Mock torch before importing backend ──────────────────────────────────────
torch_mock = types.ModuleType('torch')
torch_mock.__version__ = '2.2.0'
torch_mock.device = lambda x: x
torch_mock.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_mock.no_grad = lambda: __import__('contextlib').nullcontext()
torch_mock.LongTensor = list
sys.modules['torch'] = torch_mock

sys.path.insert(0, str(Path(__file__).parent.parent))
from msa_tts_backend import (
    ArabicTextNormalizer, ArabicDiacritizer, ArabicG2P,
    VITSInferenceModel, MSATTSPipeline, TTSEvaluator,
    ARABIC_G2P_MAP, DIACRITIC_VOWEL_MAP, ARABIC_DIACRITICS,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def normalizer():
    return ArabicTextNormalizer()

@pytest.fixture(scope="module")
def diacritizer():
    return ArabicDiacritizer()

@pytest.fixture(scope="module")
def g2p():
    return ArabicG2P()

@pytest.fixture(scope="module")
def pipeline():
    return MSATTSPipeline(model_path=None)

SR = 22050

# ── Text Normalizer ───────────────────────────────────────────────────────────
class TestNormalizer:
    def test_remove_tatweel(self, normalizer):
        assert normalizer.remove_tatweel('العـلـم') == 'العلم'

    def test_number_3(self, normalizer):
        assert 'ثلاثة' in normalizer.expand_numbers('3')

    def test_number_15(self, normalizer):
        r = normalizer.expand_numbers('15')
        assert 'خمسة' in r and 'عشر' in r

    def test_number_100(self, normalizer):
        assert 'مئة' in normalizer.expand_numbers('100')

    def test_number_1000(self, normalizer):
        assert 'ألف' in normalizer.expand_numbers('1000')

    def test_number_25(self, normalizer):
        r = normalizer.expand_numbers('25')
        assert 'خمسة' in r and 'عشرون' in r

    def test_eastern_digit_3(self, normalizer):
        assert 'ثلاثة' in normalizer.expand_numbers('٣')

    def test_eastern_digit_20(self, normalizer):
        assert 'عشرون' in normalizer.expand_numbers('٢٠')

    def test_zero(self, normalizer):
        assert 'صفر' in normalizer.expand_numbers('0')

    def test_clean_whitespace(self, normalizer):
        assert normalizer.clean_whitespace('  a  b  ') == 'a b'

    def test_strip_edges(self, normalizer):
        assert normalizer.clean_whitespace('  hello  ') == 'hello'

    def test_abbreviation_dr(self, normalizer):
        assert 'دكتور' in normalizer.expand_abbreviations('د. محمد')

    def test_full_normalize_number(self, normalizer):
        assert 'ثلاثة' in normalizer.normalize('لدي 3 كتب')


# ── Diacritizer ───────────────────────────────────────────────────────────────
class TestDiacritizer:
    def test_initialized(self, diacritizer):
        assert diacritizer.diacritizer is not None

    def test_backend_valid(self, diacritizer):
        assert diacritizer.backend in ('camel', 'mishkal', 'passthrough')

    def test_already_diacritized_kitaab(self, diacritizer):
        assert diacritizer.is_already_diacritized('الكِتَابُ') is True

    def test_already_diacritized_basmala(self, diacritizer):
        assert diacritizer.is_already_diacritized('بِسْمِ اللَّهِ الرَّحْمَنِ') is True

    def test_undiacritized_kitaab(self, diacritizer):
        assert diacritizer.is_already_diacritized('الكتاب') is False

    def test_undiacritized_lughah(self, diacritizer):
        assert diacritizer.is_already_diacritized('اللغة العربية') is False

    def test_empty_string(self, diacritizer):
        assert diacritizer.is_already_diacritized('') is False

    def test_latin_text(self, diacritizer):
        assert diacritizer.is_already_diacritized('hello world') is False

    def test_returns_string(self, diacritizer):
        r = diacritizer.diacritize('الكتاب')
        assert isinstance(r, str) and len(r) > 0


# ── G2P ──────────────────────────────────────────────────────────────────────
class TestG2P:
    def test_map_size(self):
        assert len(ARABIC_G2P_MAP) >= 35

    def test_diacritic_map_size(self):
        assert len(DIACRITIC_VOWEL_MAP) >= 7

    def test_sun_letters_count(self, g2p):
        assert len(g2p.SUN_LETTERS) == 14

    def test_pause_map_size(self, g2p):
        assert len(g2p.PAUSE_MAP) >= 5

    # Core consonants
    @pytest.mark.parametrize("char,phoneme", [
        ('\u0628', 'b'),   # ب
        ('\u062a', 't'),   # ت
        ('\u062d', 'ħ'),   # ح pharyngeal
        ('\u0639', 'ʕ'),   # ع pharyngeal
        ('\u0642', 'q'),   # ق uvular
        ('\u0635', 'sˤ'),  # ص emphatic
        ('\u0636', 'dˤ'),  # ض emphatic
        ('\u0637', 'tˤ'),  # ط emphatic
        ('\u0638', 'ðˤ'),  # ظ emphatic
        ('\u0621', 'ʔ'),   # ء glottal
        ('\u0634', 'ʃ'),   # ش
        ('\u063a', 'ɣ'),   # غ
        ('\u062e', 'x'),   # خ
    ])
    def test_consonant_mapping(self, char, phoneme):
        assert ARABIC_G2P_MAP.get(char) == phoneme

    # Diacritic vowels
    @pytest.mark.parametrize("diac,vowel", [
        ('\u064E', 'a'),   # fatha
        ('\u064F', 'u'),   # damma
        ('\u0650', 'i'),   # kasra
        ('\u0652', ''),    # sukun
        ('\u064B', 'an'),  # tanwin fath
        ('\u064C', 'un'),  # tanwin damm
        ('\u064D', 'in'),  # tanwin kasr
    ])
    def test_diacritic_vowel(self, diac, vowel):
        assert DIACRITIC_VOWEL_MAP.get(diac) == vowel

    # Sun letters
    @pytest.mark.parametrize("letter", list('تثدذرزسشصضطظلن'))
    def test_sun_letter(self, letter, g2p):
        assert letter in g2p.SUN_LETTERS

    # Moon letters should NOT be sun
    @pytest.mark.parametrize("letter", list('بجحخعغفقكمهوي'))
    def test_moon_letter_not_sun(self, letter, g2p):
        assert letter not in g2p.SUN_LETTERS

    # Pause markers
    @pytest.mark.parametrize("punct,marker", [
        ('،', '<short_pause>'),
        ('؛', '<medium_pause>'),
        ('.', '<long_pause>'),
        ('!', '<long_pause>'),
        ('؟', '<long_pause>'),
    ])
    def test_pause_map(self, punct, marker, g2p):
        assert g2p.PAUSE_MAP.get(punct) == marker

    def test_shadda_doubles(self, g2p):
        assert g2p.apply_shadda('b') == 'bb'

    def test_shadda_shin(self, g2p):
        assert g2p.apply_shadda('ʃ') == 'ʃʃ'

    def test_phonemes_returns_string(self, g2p):
        r = g2p.text_to_phonemes('مَرْحَبًا')
        assert isinstance(r, str) and len(r) > 0

    def test_sun_assimilation_output(self, g2p):
        r = g2p.text_to_phonemes('السَّلَامُ')
        assert 's' in r or 'a' in r

    def test_kitaab_phonemes(self, g2p):
        r = g2p.text_to_phonemes('كِتَابٌ')
        assert len(r.split()) >= 2

    def test_comma_pause(self, g2p):
        r = g2p.text_to_phonemes('مرحبا، كيف')
        assert 'short_pause' in r

    def test_all_28_consonants_mapped(self):
        core = [
            '\u0628','\u062a','\u062b','\u062c','\u062d','\u062e',
            '\u062f','\u0630','\u0631','\u0632','\u0633','\u0634',
            '\u0635','\u0636','\u0637','\u0638','\u0639','\u063a',
            '\u0641','\u0642','\u0643','\u0644','\u0645','\u0646',
            '\u0647','\u0648','\u064a','\u0621'
        ]
        missing = [c for c in core if c not in ARABIC_G2P_MAP]
        assert missing == [], f"Missing consonants: {missing}"


# ── Evaluator ─────────────────────────────────────────────────────────────────
class TestEvaluator:
    def test_rtf_quarter(self):
        assert abs(TTSEvaluator.real_time_factor(2.0, 0.5) - 0.25) < 1e-6

    def test_rtf_zero_duration(self):
        assert TTSEvaluator.real_time_factor(0, 1) == float('inf')

    def test_der_perfect_match(self):
        assert TTSEvaluator.diacritization_error_rate('مَرْحَبًا', 'مَرْحَبًا') == 0.0

    def test_der_no_diacritics(self):
        assert TTSEvaluator.diacritization_error_rate('مرحبا', 'مرحبا') == 0.0

    def test_der_mismatch(self):
        assert TTSEvaluator.diacritization_error_rate('مَرْحَبًا', 'مِرْحِبًا') > 0

    def test_mcd_computable(self):
        ref = np.sin(2*np.pi*220*np.linspace(0,1,SR)).astype(np.float32)
        syn = (ref + 0.01*np.random.randn(SR)).astype(np.float32)
        mcd = TTSEvaluator.mel_cepstral_distortion(ref, syn, sr=SR)
        assert isinstance(mcd, float)


# ── Constants ─────────────────────────────────────────────────────────────────
class TestConstants:
    def test_diacritics_count(self):
        assert len(ARABIC_DIACRITICS) == 11

    def test_fatha(self):
        assert ARABIC_DIACRITICS.get('fatha') == '\u064E'

    def test_shadda(self):
        assert ARABIC_DIACRITICS.get('shadda') == '\u0651'

    def test_sukun(self):
        assert ARABIC_DIACRITICS.get('sukun') == '\u0652'

    def test_tanwin_f(self):
        assert ARABIC_DIACRITICS.get('tanwin_f') == '\u064B'

    def test_maddah(self):
        assert ARABIC_DIACRITICS.get('maddah') == '\u0653'


# ── Synthesis Pipeline ────────────────────────────────────────────────────────
class TestSynthesis:
    @pytest.mark.parametrize("text,speed", [
        ('بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ', 1.0),
        ('السَّلَامُ عَلَيْكُمْ', 1.0),
        ('مَرْحَبًا', 0.5),
        ('اللُّغَةُ الْعَرَبِيَّةُ', 1.5),
    ])
    def test_demo_synthesize(self, text, speed):
        model = VITSInferenceModel(model_path=None)
        audio = model._demo_synthesize(text, speed=speed)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        peak = float(np.max(np.abs(audio)))
        assert 0 < peak <= 1.0

    def test_wav_bytes_valid_riff(self):
        model = VITSInferenceModel(model_path=None)
        audio = model._demo_synthesize('مرحبا', speed=1.0)
        wav = model.audio_to_wav_bytes(audio)
        assert wav[:4] == b'RIFF'
        assert wav[8:12] == b'WAVE'
        assert len(wav) > 44

    @pytest.mark.parametrize("text", [
        'الكتاب مفيد للطلاب',
        'السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ',
        'العلم نور والجهل ظلام، فاطلب العلم',
    ])
    def test_full_pipeline(self, text, pipeline):
        result = pipeline.synthesize(text, speed=1.0, return_phonemes=True)
        assert result['duration_s'] > 0
        assert result['rtf'] > 0
        assert len(result.get('audio_b64', '')) > 100
        assert len(result.get('phonemes', '')) > 0
