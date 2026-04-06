# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly by
emailing **asafgallea@gmail.com**. Do not open a public issue.

You can expect:

- An acknowledgment within **48 hours**
- A status update within **7 days**
- Coordinated disclosure once a fix is available

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest on `master` | Yes |
| Older releases | No |

## Scope

This policy covers the `embedding-clusters` application code, including:

- The Python backend (FastAPI server, indexer, plot computation)
- The React frontend
- Configuration and build tooling

## Security Considerations

- **File uploads** — CSV uploads are saved to a sandboxed `./uploads/`
  directory. The server validates file paths to prevent directory traversal.
- **AI credentials** — LLM API keys are configured per-session in the
  browser and sent per-request. They are not stored server-side.
- **ChromaDB** — runs embedded (no network exposure). Data is stored
  locally in `./chromadb/`.
- **No authentication** — the application is designed for local or trusted
  network use. Do not expose it to the public internet without adding an
  authentication layer.
