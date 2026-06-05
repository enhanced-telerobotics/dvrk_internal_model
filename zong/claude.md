Use the following path when testing scripts
/opt/miniconda3/envs/coppeliasim/bin/python

Guidelines:
-Insert detailed line-by-line comments.
- YAGNI (You Aren't Gonna Need It): Implement capabilities when actually needed, not when foreseen. Most foreseen needs never arrive, and speculative code carries real cost: complexity, bugs, and rework when the guess is wrong. Carve-out: if a change adds no complexity, YAGNI doesn't apply — don't invoke it to block trivial future-proofing.
- Resist clever code and architectures that solve hypothetical future problems at the cost of present complexity. "Stupid" means readable by anyone on the team without context. Pick the boring implementation when it works.
- ⁠DRY (Don't Repeat Yourself): Every piece of knowledge must have a single, unambiguous, authoritative representation. DRY governs knowledge, not literal line duplication — two functions that look alike but encode different rules are not a DRY violation, and merging them creates a worse coupling than the duplication. Apply the 𝗥𝘂𝗹𝗲 𝗼𝗳 𝗧𝗵𝗿𝗲𝗲: wait until you see the same pattern three times before extracting it; two occurrences are often a coincidence, three is a pattern. Search existing code before adding a new function.