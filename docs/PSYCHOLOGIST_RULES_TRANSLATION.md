# Psychologist-supplied Arabic rules: translation and evidence audit

The supplied Arabic notes were normalized and translated into English in
`rules_registry.json`. They are recorded as clinician-supplied hypotheses, not
scientifically validated rules.

The search found indirect research on intentionally expressed emotion through
facial cues, drawing size, and relative placement. It did not find empirical
support for the notes' fixed meanings for foxes, squirrels, lions, circles,
stars, vehicles, hearts, geometric shapes, or absolute left/right/top
personality interpretations.

The registry therefore:

- caps confidence between 0.05 and 0.25;
- requires visible, verified evidence;
- leaves unavailable object detections as `not_evaluated`;
- never converts a single symbolic rule into a Suggested Concern Profile;
- includes safe follow-up wording rather than categorical personality claims;
- cites located research in every relevant evaluation;
- requires clinician review for every imported rule.

The final paragraph of the source asserts that every line has psychological
meaning and that studies prove this. The literature located does not justify
that blanket statement. It is retained as source context but not implemented as
an executable rule.
