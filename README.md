# Captions

> Real-time audio transcriber overlay for Linux (Zorin OS / Ubuntu).

Captures system audio (Meet, Zoom, browser, anything playing) and shows live captions in a
floating, always-on-top window. **100% offline** — uses local Whisper model.

## Features

- 🎧 Captures system audio (PulseAudio loopback) — no microphone needed
- ⚡ Real-time transcription with `faster-whisper` (CUDA-accelerated)
- 🪟 Floating overlay always on top of other windows
- 🔁 Smart dedup — no repeated phrases
- ⌨️ Global hotkeys (toggle / quit / reset)
- 🌍 Multi-language ready (English now, PT translation in v2)

## Quick Start (Zorin 17 / Ubuntu)

### 1. System dependencies

```bash
sudo apt update
sudo apt install -y python3.11 python3-pip python3-venv \
                    pulseaudio-utils libpulse-dev
```

### 2. NVIDIA CUDA (recommended, optional)

For 3-5× faster transcription:

```bash
# Follow https://docs.nvidia.com/cuda/cuda-installation-guide-linux-ubuntu/
sudo apt install -y nvidia-cuda-toolkit
# Install PyTorch with CUDA: https://pytorch.org/get-started/locally/
```

### 3. Run

```bash
./run.sh
```

The first run downloads the `small.en` Whisper model (~460MB, cached afterwards).

## Usage

```bash
./run.sh                                    # default settings
./run.sh --model tiny.en                    # CPU-friendly model
./run.sh --device cpu                       # force CPU
./run.sh --no-overlay                       # CLI-only mode (logs to stdout)
./run.sh --list-sources                     # debug PulseAudio sources
./run.sh --source alsa_output.pci-0000_...  # specific source
```

## Hotkeys

| Combo          | Action               |
|----------------|----------------------|
| `Ctrl+Shift+T` | Toggle overlay       |
| `Ctrl+Shift+R` | Reset history        |
| `Ctrl+Shift+Q` | Quit                 |

## Architecture

See [`.specs/features/captions/design.md`](.specs/features/captions/design.md).

```
PulseAudio .monitor → AudioCapture → VAD → Whisper (CUDA) → Dedup → Overlay
```

## Troubleshooting

### `parec: command not found`

You need `pulseaudio-utils`:

```bash
sudo apt install -y pulseaudio-utils libpulse-dev
```

Verify: `which parec` should return `/usr/bin/parec`.

### `No PulseAudio source found` / `Found N sinks but none has a .monitor source`

List sources and look for one ending in `.monitor`:

```bash
./run.sh --list-sources
```

You should see entries like `alsa_output.pci-0000_00.monitor`. If only `.monitor`
entries are listed and none have audio, make sure something is actually playing
(audio playing in another tab/window → sink state becomes "running").

Force a specific source:

```bash
./run.sh --source alsa_output.pci-0000_00.monitor
```

If you're on **PipeWire** (Zorin 17 default), install `pulseaudio-utils` for
`parec` compatibility. PipeWire emulates the PulseAudio API.

### `CUDA not available — falling back to CPU`

The app detected no NVIDIA GPU. Either:

1. Install CUDA + PyTorch with CUDA: https://pytorch.org/get-started/locally/
2. Accept CPU mode (use `--model tiny.en` for ~3x realtime)

### Overlay invisible / behind windows / shows "display not found"

The overlay uses Tkinter + X11 `wm_attributes('-topmost')`. On **Wayland**
sessions (some Zorin configs), `topmost` is unreliable.

Fix: log out → gear icon at login → "Zorin on Xorg".

Verify Tk is installed:

```bash
python3 -c "import tkinter; print('ok')"
```

If `ModuleNotFoundError: No module named 'tkinter'`:

```bash
sudo apt install -y python3-tk
```

Workaround: `--no-overlay` mode prints captions to the terminal.

### `faster-whisper is not installed`

The first run downloads the model (~460MB). If the install failed:

```bash
source .venv/bin/activate
pip install -e ".[stt]"
```

Verify: `python -c "import faster_whisper; print('ok')"`.

### `AudioCaptureError: parec exited unexpectedly. stderr: Connection refused`

PulseAudio daemon crashed or isn't running for your user:

```bash
systemctl --user status pulseaudio
# or for PipeWire:
systemctl --user status pipewire pipewire-pulse
```

Restart: `systemctl --user restart pulseaudio`.

### `Tkinter is required for the GUI overlay but is not installed`

See "Overlay invisible" above, OR run in console mode:

```bash
./run.sh --no-overlay
```

### Model download is slow / fails

Models cache at `~/.cache/huggingface/`. Delete and retry:

```bash
rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-*
./run.sh  # re-downloads on first inference
```

## Hotkeys

| Combo | Action |
|-------|--------|
| `Ctrl+Shift+T` | Show / hide overlay |
| `Ctrl+Shift+R` | Reset caption history |
| `Ctrl+Shift+Q` | Quit captions |

## Development

```bash
./run.sh                                   # install dev deps
source .venv/bin/activate
pytest -q                                  # run tests
ruff check src tests                       # lint
mypy src                                   # type check
```

## License

MIT
