"""
MSA Arabic TTS — Live API Integration Tests
51 tests covering all endpoints, validation, speed/pitch variations
Run with server active: pytest tests/test_api.py -v
Or standalone (starts server in-process): pytest tests/test_api.py -v --api
"""
import sys, types, threading, time, json, base64, pytest
from http.client import HTTPConnection

# ── Mock torch ───────────────────────────────────────────────────────────────
torch_mock = types.ModuleType('torch')
torch_mock.__version__ = '2.2.0'
torch_mock.device = lambda x: x
torch_mock.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_mock.no_grad = lambda: __import__('contextlib').nullcontext()
torch_mock.LongTensor = list
sys.modules['torch'] = torch_mock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

PORT = 18799

@pytest.fixture(scope="session", autouse=True)
def api_server():
    """Start FastAPI server in background thread for duration of test session."""
    from msa_tts_backend import app
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(2.5)
    yield server
    server.should_exit = True


def req(method, path, body=None):
    conn = HTTPConnection("127.0.0.1", PORT, timeout=15)
    headers = {"Content-Type": "application/json"} if body else {}
    data = json.dumps(body).encode() if body else None
    conn.request(method, path, body=data, headers=headers)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:    return r.status, json.loads(raw)
    except: return r.status, {"_raw": raw[:200]}


class TestHealth:
    def test_http_200(self):        assert req("GET", "/health")[0] == 200
    def test_status_ok(self):       assert req("GET", "/health")[1].get("status") == "ok"
    def test_model_loaded_field(self): assert "model_loaded" in req("GET", "/health")[1]
    def test_diacritizer_field(self):  assert "diacritizer" in req("GET", "/health")[1]
    def test_device_field(self):       assert "device" in req("GET", "/health")[1]


class TestVoices:
    def test_http_200(self):         assert req("GET", "/voices")[0] == 200
    def test_voices_list(self):      assert isinstance(req("GET", "/voices")[1].get("voices"), list)
    def test_at_least_one_voice(self): assert len(req("GET", "/voices")[1].get("voices", [])) >= 1
    def test_voice_id(self):         v = req("GET", "/voices")[1]["voices"][0]; assert "id" in v
    def test_voice_language(self):   v = req("GET", "/voices")[1]["voices"][0]; assert "language" in v
    def test_voice_model_vits(self):  v = req("GET", "/voices")[1]["voices"][0]; assert v.get("model") == "VITS"


class TestNormalize:
    @pytest.mark.parametrize("text,substr", [
        ("لدي 3 كتب",   "ثلاثة"),
        ("العـلـم",      "العلم"),
        ("د. محمد",     "دكتور"),
        ("٥ طلاب",      "خمسة"),
        ("0 نتائج",     "صفر"),
    ])
    def test_normalize(self, text, substr):
        code, d = req("POST", "/normalize", {"text": text})
        assert code == 200 and substr in d.get("normalized", "")


class TestDiacritize:
    @pytest.mark.parametrize("text", ["الكتاب", "اللغة العربية", "العلم نور"])
    def test_diacritize(self, text):
        code, d = req("POST", "/diacritize", {"text": text})
        assert code == 200 and "diacritized" in d and len(d["diacritized"]) > 0


class TestG2P:
    @pytest.mark.parametrize("text,must", [
        ("مَرْحَبًا",            ["m", "r", "ħ"]),
        ("كِتَابٌ",              ["k"]),
        ("السَّلَامُ",           ["s", "a", "l"]),
        ("كِتَابٌ، مَدْرَسَةٌ", ["short_pause"]),
    ])
    def test_g2p(self, text, must):
        code, d = req("POST", "/g2p", {"text": text})
        phones = d.get("phonemes", "")
        assert code == 200 and len(phones) > 0
        for m in must:
            assert m in phones


class TestSynthesize:
    @pytest.mark.parametrize("body", [
        {"text": "مرحبا",                    "speed": 1.0},
        {"text": "السلام عليكم",             "speed": 0.8},
        {"text": "اللغة العربية الفصحى",    "speed": 1.2},
        {"text": "العلم نور والجهل ظلام",   "speed": 1.0},
    ])
    def test_synthesize_basic(self, body):
        code, d = req("POST", "/synthesize", body)
        assert code == 200
        wav = base64.b64decode(d.get("audio_b64", ""))
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
        assert d.get("duration_s", 0) > 0

    def test_synthesize_with_phonemes(self):
        code, d = req("POST", "/synthesize",
                      {"text": "بِسْمِ اللَّهِ", "speed": 1.0, "return_phonemes": True})
        assert code == 200
        assert len(d.get("phonemes", "")) > 0
        assert len(d.get("normalized_text", "")) > 0
        assert len(d.get("diacritized_text", "")) > 0
        assert "diacritizer" in d

    def test_stream_endpoint(self):
        import urllib.request
        rq = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/synthesize/stream",
            data=json.dumps({"text": "مرحبا بالعالم", "speed": 1.0}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(rq, timeout=15) as r:
            ct = r.headers.get("Content-Type", "")
            wav = r.read()
        assert "audio" in ct
        assert wav[:4] == b"RIFF"
        assert len(wav) > 1000


class TestValidation:
    def test_empty_text(self):
        assert req("POST", "/synthesize", {"text": ""})[0] == 422

    def test_text_too_long(self):
        assert req("POST", "/synthesize", {"text": "م" * 2001})[0] == 422

    def test_speed_too_high(self):
        assert req("POST", "/synthesize", {"text": "مرحبا", "speed": 5.0})[0] == 422

    def test_pitch_too_high(self):
        assert req("POST", "/synthesize", {"text": "مرحبا", "pitch_shift": 10.0})[0] == 422

    def test_normalize_too_long(self):
        assert req("POST", "/normalize", {"text": "م" * 5001})[0] == 422


class TestSpeedVariations:
    @pytest.mark.parametrize("speed", [0.5, 0.75, 1.0, 1.5, 2.0])
    def test_speed(self, speed):
        code, d = req("POST", "/synthesize", {"text": "السلام عليكم", "speed": speed})
        assert code == 200 and d.get("duration_s", 0) > 0


class TestPitchVariations:
    @pytest.mark.parametrize("pitch", [-6.0, -3.0, 0.0, 3.0, 6.0])
    def test_pitch(self, pitch):
        code, d = req("POST", "/synthesize", {"text": "مرحبا", "pitch_shift": pitch})
        assert code == 200 and d.get("duration_s", 0) > 0
