# TUI Freeze Issue on Windows - Technical Documentation

> **Status: RESOLVED** ✅  
> **Solution: Subprocess-based inference isolation**

## Problem Summary

The Parishad TUI froze on Windows during LLM inference when using `llama-cpp-python`. The UI would become completely unresponsive even though model inference completed successfully in the background.

### Symptoms
- TUI freezes immediately after submitting a query
- Model inference completes (GPU usage spikes then drops)
- Output files are generated correctly
- TUI remains frozen until force-quit
- **Works fine on macOS, only affects Windows**

---

## Root Cause

### Python's Global Interpreter Lock (GIL)

The `llama-cpp-python` library holds Python's GIL during C-level inference operations. This blocks **all Python threads**, including Textual's event loop.

Even when inference runs in a separate thread:
1. Main thread (Textual TUI) waits for timer callbacks
2. Worker thread (llama-cpp) acquires GIL for C extension
3. C extension doesn't release GIL during computation
4. Main thread cannot execute any Python code
5. **TUI freezes completely**

### Proof

Debug logging showed a 31-second gap where the TUI's timer stopped firing entirely:

```
[16:04:33.134] POLL: Timer fired...    ← LAST TIMER CALLBACK
                                        
     ⚠️ 31 SECONDS - NO TIMER CALLBACKS ⚠️
                                        
[16:05:04.329] === INFERENCE COMPLETE ===
```

This proves the GIL was held continuously during inference.

---

## Solution: Process Isolation

Since threads share the GIL, the only solution is **true process isolation** using `subprocess.Popen`.

### Architecture

```
┌─────────────────────────┐      FILE IPC      ┌─────────────────────────┐
│    MAIN PROCESS         │  ←─────────────→  │    SUBPROCESS           │
│    (TUI - Textual)      │                    │    (Inference)          │
│                         │                    │                         │
│  • Timers work ✓        │  query.txt →       │  • Loads Parishad       │
│  • UI responsive ✓      │  ← result.json     │  • Runs council.run()   │
│  • Polls every 500ms    │                    │  • Saves JSON result    │
│                         │                    │                         │
│  OWN GIL (free)         │                    │  OWN GIL (busy)         │
└─────────────────────────┘                    └─────────────────────────┘
```

### Implementation

1. Save query to `~/.parishad/temp_query.txt`
2. Generate Python script that imports Parishad and runs inference
3. Launch subprocess with `subprocess.Popen()` (hidden window on Windows)
4. Poll for `temp_result.json` every 500ms
5. Display result when file appears

### Key Code

```python
# Launch subprocess - SEPARATE PROCESS = SEPARATE GIL
self._subprocess = subprocess.Popen(
    [python_exe, str(script_file)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    startupinfo=startupinfo,  # Hidden window on Windows
    cwd=str(self.cwd),
)

# Poll for result (TUI stays responsive!)
def poll_subprocess_result():
    if result_file.exists():
        result = json.loads(result_file.read_text())
        display_result(result)
    else:
        self.set_timer(0.5, poll_subprocess_result)
```

---

## Tradeoffs

| Aspect | Before (Threading) | After (Subprocess) |
|--------|-------------------|-------------------|
| TUI responsiveness | **FROZEN** | **RESPONSIVE** |
| GIL blocking | Yes | No |
| Model reload per query | No | Yes (~10-15s) |
| Memory usage | Shared | Doubled |
| IPC complexity | Low | Medium (file-based) |

The model reload overhead is acceptable because the TUI now remains usable during inference.

---

## Files Modified

- `src/parishad/cli/code.py` - Replaced threading with subprocess-based inference

---

## Testing

1. Run `parishad code`
2. Submit any query
3. Verify TUI remains responsive (can scroll, etc.)
4. Result appears after inference completes

---

## Related Issues

This is a known pattern in LLM TUI applications:

- `anomalyco/opencode` - Issues #9269, #8229
- `ggerganov/llama.cpp` - Issues #3135, #1793
- `Textualize/textual` - Issues #2167, #4552

All report similar symptoms: TUI freezes during LLM inference on Windows due to GIL contention.

---

## Future Improvements

1. **Persistent subprocess** - Keep model loaded between queries
2. **Named pipes** - Faster IPC than file polling
3. **Model caching** - Reduce reload time
