# Security Policy

## Supported Versions

Security fixes are currently targeted at the active development branch.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| historical commits, tags, and long-lived forks | No |

## Reporting a Vulnerability

Please do **not** open public GitHub issues, pull requests, or discussions for
security problems.

Use one of these private channels instead:

1. GitHub's private vulnerability reporting flow for this repository.
2. The contact options listed on the repository owner's GitHub profile:
   <https://github.com/francislabountyjr>

If you use a general contact channel, include `[security]` in the subject or
opening line so the report can be triaged correctly.

## What to Include

A useful report should include:

- a clear description of the issue and why it is security-relevant
- affected code paths, scripts, configs, or workflows
- reproduction steps or a proof of concept when possible
- impact assessment, including whether the issue can expose data, execute code,
  corrupt outputs, or load untrusted artifacts
- any suggested remediation or mitigations already identified

Please avoid including large private datasets, copyrighted media, secrets, or
production credentials in the report.

## Response Expectations

- The maintainer will aim to acknowledge valid reports within 7 business days.
- Fixes will generally land on `main` first.
- Backports to older commits or tags are not guaranteed.
- Public disclosure should wait until a fix or mitigation is available, unless
  coordinated otherwise with the maintainer.

## Scope

This policy covers vulnerabilities in the repository contents, including:

- `stereo2spatial/` package code
- CLI entrypoints such as `train.py` and `infer.py`
- repository automation under `.github/`
- dataset, export, and Atmos helper scripts under `scripts/`
- vendored code in `stereo2spatial/vendor/` as shipped in this repository

If the root cause is clearly in an upstream dependency or upstream EAR-VAE
project rather than this repository's integration layer, coordinated disclosure
with the upstream project is encouraged.
