# Security

bruhOS drives a physical robot, so we treat security bugs as safety bugs.

## Reporting

Please report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/bruhosdotai/bruhOS/security/advisories/new)
or by email to admin@bruhos.ai. Public issues are fine for low-risk hardening.

Include what an attacker controls (brain output, network, config, a web page), what they gain, and a
minimal reproduction if you have one. We aim to reply within 3 days and to ship a fix with a regression
test.

## Scope

In scope: everything in this repository — the runtime, skill whitelist, brain adapters, bodies, trace
tools, the local dashboard and the GitHub workflows.

Out of scope: third-party model servers, the Hiwonder SDK and robot image, and the bruhOS mobile app.

## Design rules

- A brain only chooses skills from the whitelist; raw joint actions are rejected before the body.
- Every skill is bounded by a step budget, and `stop` preempts everything locally.
- Credentials are only sent over TLS or to loopback, and HTTP redirects are never followed.
- Local tools (the dashboard) bind to `127.0.0.1` and only answer loopback host names.
