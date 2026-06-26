# UAT — End-to-End Manual Test Script

> **Goal**: Verify captions transcribes real audio from a virtual meeting
> in < 2 seconds with no duplicate phrases.

## Prerequisites

- Zorin OS 17 (or any Ubuntu 22.04+ with PulseAudio or PipeWire)
- NVIDIA GPU with CUDA installed (recommended) OR accept CPU mode (`tiny.en`)
- A virtual meeting platform: Google Meet, Zoom, or any browser tab playing audio
- ~10 minutes

## Setup

```bash
# 1. Install system deps
sudo apt install -y python3.11 python3-pip python3-tk pulseaudio-utils

# 2. Clone + install
cd ~
git clone <repo-url> captions
cd captions
./run.sh   # downloads model on first run (~460MB)
```

## Test Cases

### TC1 — App starts and lists sources

```bash
./run.sh --list-sources
```

**Expected**: List of sinks + sources. At least one entry ending in `.monitor`.

**Pass criteria**: Output shows `alsa_output.*.monitor` or similar.

### TC2 — App starts with overlay

```bash
./run.sh
```

**Expected**:
- Log line: `captions starting | source=... model=small.en device=cuda`
- Tk window appears, semi-transparent black, bottom-center of screen
- Window text says "captions" (title bar) and is empty initially

**Pass criteria**: Window visible and not behind other windows.

### TC3 — Transcribes browser audio

1. Open Chrome → YouTube → play a 1-minute English podcast/talk
2. Watch the overlay
3. Note the latency between speech in video and text appearing

**Expected**: Text appears within 1-2 seconds of speech. English accuracy >90%.

**Pass criteria**:
- Latency ≤ 2s
- Recognizable phrases
- No spurious duplicates from overlap

### TC4 — Transcribes Google Meet

1. Join a Meet meeting (or open it in a separate window)
2. Have someone talk (or use Meet's "test speakers" feature)
3. Verify overlay shows their words

**Pass criteria**: Same as TC3.

### TC5 — Hotkey toggle (Ctrl+Shift+T)

1. With overlay visible, press **Ctrl+Shift+T**
2. Overlay should hide (withdraw but stays in tray)
3. Press again → overlay reappears at same position

**Pass criteria**: Toggle works, no crashes.

### TC6 — Hotkey reset (Ctrl+Shift+R)

1. Let captions accumulate for ~30s
2. Press **Ctrl+Shift+R**
3. Overlay should clear its history; next utterance is treated as "new"

**Pass criteria**: No error in stderr.

### TC7 — Hotkey quit (Ctrl+Shift+Q)

1. Press **Ctrl+Shift+Q**
2. App should exit cleanly

**Pass criteria**: Process terminates, log says "captions stopped".

### TC8 — SIGINT shutdown

1. Run `./run.sh --no-overlay` in a terminal
2. Press **Ctrl+C**
3. App should exit

**Pass criteria**: No stack trace, log says "interrupted by user".

### TC9 — CPU fallback

```bash
./run.sh --model tiny.en --device cpu
```

**Pass criteria**: Runs on CPU, warns "CUDA not available", still transcribes (slower).

### TC10 — Console mode

```bash
./run.sh --no-overlay
```

Play audio in another tab. Expected: English text printed to stdout, one line per chunk.

**Pass criteria**: No GUI window opens. Captions printed as text.

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Latency (CUDA small.en) | < 1.5s | 2s chunks + 400ms inference |
| Latency (CPU tiny.en) | < 4s | 2s chunks + 2s inference |
| CPU usage (idle) | < 5% | VAD skips silence |
| RAM usage | < 1GB | Excluding model |
| First-run setup | < 5min | Including model download |

## Known Limitations (v1)

- Single language at a time (no mid-stream switching)
- English-optimized model (small.en). Multilingual needs `--model small` (slower)
- No microphone capture (only system audio loopback)
- No file export
- Linux only (X11 session required for overlay)

## Sign-off

After completing all TCs, file a UAT report at `.specs/features/captions/UAT-RESULT.md`
with: tester name, date, results per TC, any defects found.

```
Tester: _______________
Date:   _______________
GPU:    _______________
Model:  _______________

[ ] TC1  [ ] TC2  [ ] TC3  [ ] TC4  [ ] TC5
[ ] TC6  [ ] TC7  [ ] TC8  [ ] TC9  [ ] TC10

Notes:
_____________________________________________
_____________________________________________
```
