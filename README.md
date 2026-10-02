# Pause

For people who hand errands to an AI assistant, Pause removes the fear that the agent will confidently do the wrong thing.

A LangChain agent that manages a simulated workspace (calendar, tasks, contacts and notes) from text, voice notes and photos. Before any action that changes something, an uncertainty gate samples a few proposals, measures how much they agree, and then acts, asks one question, or hands the choice to the user.

Work in progress. The backend and the gate middleware are built and tested. The evals and the web app come next.

- The gate as a reusable package: [packages/uncertainty-gate](packages/uncertainty-gate)
- Course walkthrough: [notebooks/pause_course.ipynb](notebooks/pause_course.ipynb)

```bash
cp .env.example .env            # then add GOOGLE_API_KEY and TAVILY_API_KEY
uv sync --all-packages && uv run pytest
uv run python -m pause.cli "Move the dentist to next Friday at 3pm" --thread t1
```
