# Project Security Guide

Comprehensive security reference for all projects. Covers secrets management, personal data protection, dependency security, secure coding, infrastructure, and incident response.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Secrets & Credentials](#secrets--credentials)
3. [API Keys & Tokens](#api-keys--tokens)
4. [Personal Data Protection](#personal-data-protection)
5. [Secure Coding Practices](#secure-coding-practices)
6. [Dependency Security](#dependency-security)
7. [Database & Storage Security](#database--storage-security)
8. [Network & Infrastructure](#network--infrastructure)
9. [Authentication & Authorization](#authentication--authorization)
10. [File Rules by Type](#file-rules-by-type)
11. [Gitignore Checklist](#gitignore-checklist)
12. [Template File Convention](#template-file-convention)
13. [Pre-Commit Checklist](#pre-commit-checklist)
14. [Incident Response](#incident-response)
15. [AI Session Security](#ai-session-security)

---

## Core Principles

1. **Never commit secrets to git.** No API keys, passwords, tokens, certificates, or credentials.
2. **Never commit personal data to git.** No names, emails, usernames, file paths, employer names, or document content.
3. **Least privilege.** Every component gets the minimum access it needs.
4. **Defense in depth.** Combine gitignore + env vars + runtime validation + access controls.
5. **Fail secure.** When something goes wrong, deny access rather than grant it.
6. **Local-first.** Keep sensitive data on-device. Only send to external services when explicitly required.

---

## Secrets & Credentials

### What Counts as a Secret

| Type | Examples | Common File Locations |
|------|----------|-----------------------|
| API keys | `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID` | .env, config.py, settings.json |
| Passwords | Database passwords, service account passwords | .env, docker-compose.yml |
| Tokens | OAuth tokens, JWT signing keys, refresh tokens | .env, auth configs |
| Certificates | TLS/SSL certs, private keys, signing certs | *.pem, *.key, *.crt |
| Connection strings | Database URLs with embedded credentials | .env, config files |
| Webhook URLs | Slack webhooks, Discord webhooks | .env, config files |
| Encryption keys | AES keys, Fernet keys, master keys | .env, keyfiles |
| Service accounts | GCP service account JSON, AWS credentials | credentials.json |

### How to Handle Secrets

**Do:**
- Store in environment variables loaded from `.env` (gitignored)
- Use a secrets manager for production (AWS Secrets Manager, HashiCorp Vault, 1Password CLI)
- Use `.example` files with placeholder values committed to the repo
- Load secrets at runtime, never at import time
- Rotate secrets on a schedule and after any suspected leak

**Don't:**
- Hardcode secrets anywhere in source code
- Pass secrets as command-line arguments (visible in process lists)
- Log secrets at any level
- Store secrets in comments, TODOs, or documentation
- Copy secrets between projects
- Commit secrets "temporarily"

### Environment Variable Pattern

```bash
# .env (GITIGNORED)
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
OPENAI_API_KEY=sk-proj-abc123...

# .env.example (COMMITTED)
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
OPENAI_API_KEY=sk-proj-your-key-here
```

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set. Copy .env.example to .env and configure.")
```

---

## API Keys & Tokens

### Key Management Rules

1. **One key per service per environment.** Don't reuse between dev/staging/prod.
2. **Scope keys minimally.** Use the narrowest permissions possible.
3. **Set expiry dates.** Rotate long-lived keys quarterly.
4. **Monitor usage.** Set billing alerts and usage caps.
5. **Revoke immediately** if a key is committed, logged, or exposed.

### If a Key Is Leaked

1. **Revoke the key immediately** from the provider's dashboard
2. Generate a new key
3. Update `.env` and deployed environments
4. If committed to git: purge from history with `git filter-repo`
5. Check service logs for unauthorized usage
6. Enable billing alerts if not already set

---

## Personal Data Protection

### What Counts as Personal Data

| Category | Examples | Where It Hides |
|----------|----------|----------------|
| Identity | Name, email, username | pyproject.toml, README, config |
| File paths | `/Users/yourname/...` | Scripts, config, docs |
| Org structure | Employer names, folder categories | Classification configs |
| Document content | Text from ingested files | Staging manifests, logs, fixtures |
| Session logs | Processing history, queries | Progress logs, debug output |

### Rules

- Never commit real personal data — use synthetic/placeholder data
- Use role labels instead of names: "(Owner)", "(Admin)", "User"
- Use generic paths: `$HOME/...`, `/path/to/project/`, `/Users/yourname/`
- Keep session-specific logs in gitignored directories

---

## Secure Coding Practices

### Input Validation

Always validate at system boundaries (user input, API requests, file uploads).

```python
def search(query: str, k: int = 5) -> list:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
    query = query.strip()[:1000]
```

### Injection Prevention

| Attack | Prevention |
|--------|------------|
| SQL injection | Parameterized queries, never string concatenation |
| Command injection | `subprocess.run([...])` with list args, never `shell=True` with user input |
| Path traversal | Validate with `os.path.realpath()`, reject `..` components |
| XSS | Escape HTML output, use auto-escaping template engines |
| SSRF | Validate URLs against allowlist, reject internal IPs |

### Error Handling

- Never expose stack traces or internal paths in API responses
- Log full errors server-side, return generic messages to clients
- Don't include secrets or PII in error messages

### Logging

- Never log secrets, passwords, tokens, or API keys
- Never log full request/response bodies that may contain PII
- Use structured logging with levels (DEBUG for dev, INFO+ for production)

---

## Dependency Security

1. **Pin versions** for reproducible builds
2. **Audit regularly** with `pip-audit`, `safety check`, `npm audit`
3. **Update promptly** for security advisories
4. **Minimize dependencies** — fewer deps = smaller attack surface
5. **Review new deps** — check maintenance status, known vulnerabilities

```bash
pip-audit                          # Check for known vulnerabilities
pip list --outdated                # See what needs updating
safety check -r requirements.txt  # Alternative scanner
```

---

## Database & Storage Security

### Local Databases

- Store database files outside the project directory (e.g., `~/.appname/`)
- Gitignore all database files (`*.db`, `*.sqlite`, `*.lance`)
- Set restrictive file permissions: `chmod 600`
- Back up regularly with checksums

### Connection Strings

- Always use environment variables
- Never embed credentials in URLs committed to git
- Use SSL/TLS for remote connections

---

## Network & Infrastructure

### Localhost Binding

```python
app.run(host="127.0.0.1", port=8000)  # Safe — localhost only
app.run(host="0.0.0.0", port=8000)    # DANGEROUS — exposes to network
```

### HTTPS/TLS

- Always use HTTPS for external API calls
- Never set `verify=False` in production
- Use TLS 1.2+

### CORS

- Restrict origins to known domains
- Never use `allow_origins=["*"]` with credentials

---

## Authentication & Authorization

### Local-First Applications

- Bind servers to `127.0.0.1` only
- Validate requests come from localhost
- Don't expose destructive endpoints without confirmation

---

## File Rules by Type

| File Type | Rules |
|-----------|-------|
| Scripts (.sh, .py) | Use `SCRIPT_DIR` / env vars, never hardcoded paths |
| Config (.yaml, .json, .env) | Commit only `.example` templates with placeholders |
| Documentation (.md) | Generic paths, no real names, no real API keys in examples |
| Tests (.py) | Synthetic data only, `tempfile.TemporaryDirectory()`, mock external services |
| Docker / CI | Never embed secrets; use platform secret stores |

---

## Gitignore Checklist

```gitignore
# Secrets & Credentials
.env
.env.*
!.env.example
*.pem
*.key
*.crt
credentials.json
service-account*.json
.secrets/

# Database & Runtime
*.db
*.sqlite
*.sqlite3
*.log
logs/

# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/
.venv/
venv/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE & OS
.idea/
.vscode/
.DS_Store
```

---

## Template File Convention

For any file containing user-specific data or secrets:

1. Create the real file locally (gitignored): `config.yaml`
2. Create a committed template: `config.example.yaml`
3. Add comment at top: `# Copy to config.yaml and customize`
4. Document the setup step in README

**Naming**: `<filename>.example.<ext>`

---

## Pre-Commit Checklist

- [ ] No API keys, tokens, or passwords in code
- [ ] No `.env` file staged
- [ ] No real names, emails, or usernames
- [ ] No hardcoded `/Users/yourname/` paths
- [ ] No `verify=False` on HTTPS requests
- [ ] No `shell=True` with user input
- [ ] No sensitive data in log statements
- [ ] Error responses don't expose internals

```bash
# Quick scan
git diff --cached | grep -iE "(api_key|secret|password|token|sk-|/Users/[a-z])"
```

---

## Incident Response

### If a Secret Was Committed

1. **Revoke immediately** at the source
2. Remove from tracking: `git rm --cached <file>` + add to `.gitignore`
3. Purge from history: `git filter-repo --path <file> --invert-paths`
4. Force push (coordinate with team)
5. Audit service logs for unauthorized usage
6. Post-mortem: identify cause, add safeguards

### Prevention Tools

- **pre-commit hooks**: `detect-secrets`, `gitleaks`
- **CI scanning**: GitHub secret scanning, `bandit`
- **Project scanner**: `./scripts/security_scan.sh --staged`

---

## AI Session Security

When working with AI assistants (Claude Code, Copilot, ChatGPT):

- **Session logs** contain personal context — keep gitignored
- **AI-generated code** may include placeholder secrets that look real — verify
- **AI-generated docs** may include your name or details — review before committing
- **Don't paste secrets** into AI conversations — use placeholder values
- **Review AI output** for hardcoded paths, names, or config values

---

*Reusable across projects. Copy to any new repo as a starting point.*
