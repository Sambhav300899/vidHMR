# vidHMR

Human mesh recovery (HMR) from video, with a video diffusion model's DiT used directly as the
encoder. The idea: a video model's features carry temporal context, so the HMR head can recover
body parts that are occluded in a single frame.

## Rules

1. **Data and artifacts** go in `generated/` (gitignored). Use `vidHMR.GENERATED`, not hardcoded paths.
2. **All main code** lives in the `vidHMR` package under `src/vidHMR/`.
3. **External repos** are git submodules (`git submodule add <url> external/<name>`), never copied in.
4. **Reusable code**: shared functionality goes in one module/function that every caller routes through. No one-off duplicates, also within a package. Keep it simple: no abstraction that only one caller uses.
5. **Style**: PEP 8, enforced by ruff (`pyproject.toml`, line length 99): run `ruff format` and `ruff check` before finishing. Separate logical blocks inside functions with blank lines (setup, each processing step, output), the way a person would group them. Use a `def` rather than assigning a lambda.
6. **Communication**: write all messages to the user in ASD-STE100 Simplified Technical English. Use short sentences (20 words maximum for instructions, 25 for descriptions), active voice, one instruction per sentence, and one meaning for each word.
