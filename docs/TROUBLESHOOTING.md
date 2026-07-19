# Troubleshooting

- If `py` is unavailable, use the full path to a Python 3.11 executable.
- If training reports missing dependencies, install `".[ml]"`.
- If no images are found, verify the exact folders `train`, `valid`, and `test`.
- If the test command refuses to run, this is the locked-test safeguard.
- A blank page intentionally returns `placement: unavailable`.
- The psychologist source importer remains disabled until the PDF is readable;
  never hand-create inferred clinical rules as a workaround.
