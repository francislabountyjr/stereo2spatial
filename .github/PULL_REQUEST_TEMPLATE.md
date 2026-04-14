## Summary

Describe the behavior change and the reason for it.

## Related Issues

Link the issue(s) this PR resolves or references.

## Testing

List the validation you ran, for example:

```bash
ruff check .
mypy stereo2spatial tests
pytest -q
```

If you skipped a check, explain why.

## Reviewer Notes

Call out anything that needs special attention, such as config compatibility,
checkpoint or bundle format changes, vendor updates, or dataset assumptions.

## Checklist

- [ ] The change is scoped to one coherent problem.
- [ ] I added or updated tests when behavior changed.
- [ ] I updated docs, config examples, or CLI help text when user-facing behavior changed.
- [ ] I ran the relevant local checks, or I explained why I could not.
- [ ] I did not commit local datasets, checkpoints, run artifacts, generated audio, or exported bundles.
- [ ] I described any changes under `stereo2spatial/vendor/` or other upstream-derived code.
- [ ] I called out breaking changes to config schema, checkpoint loading, or inference/export compatibility.
- [ ] I did not use this template to report a security issue publicly.
