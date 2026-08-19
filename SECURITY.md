# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this repository, please **do not open a public
issue**. Report it privately to the repository owner via GitHub's private
vulnerability reporting, or email the contact listed on the author's profile.

Please include:

- The affected file(s) and repository
- A description of the issue and its potential impact
- A minimal reproduction, if possible

## Secrets

- **Never commit secrets.** Broker access tokens, client IDs, API keys and
  passwords must be supplied via environment variables or a secrets manager.
- This project reads credentials from `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN`
  environment variables. See `.env.example` for the supported variables.
- If a secret was ever committed, treat it as compromised: rotate/revoke it
  immediately, purge it from Git history, and coordinate a force-push with any
  clones.

## Live Trading

Live trading is **disabled by default**. It requires:

1. `trading.mode: live` in configuration, **and**
2. the environment variable `ALLOW_LIVE_TRADING=YES_I_UNDERSTAND`.

The system refuses to place live orders unless both conditions hold. All other
paths are paper-trading only.