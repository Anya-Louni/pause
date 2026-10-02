# Versions

Confirmed on 2026-09-29 against the installed packages.

| Package | Version |
|---|---|
| langchain | 1.4.3 |
| langchain-core | 1.6.5 |
| langgraph | 1.2.12 |
| langgraph-checkpoint | 4.2.0 |
| langgraph-checkpoint-sqlite | 3.1.1 |
| langchain-google-genai | 4.4.0 |
| langchain-tavily | 0.2.18 |
| langchain-tests | 1.1.9 |
| Python | 3.12 |

## Confirmed APIs

- `langchain.agents.create_agent(model, tools, *, system_prompt, middleware, response_format, state_schema, context_schema, checkpointer, store, ...)`
- `langchain.chat_models.init_chat_model(model, *, model_provider, configurable_fields, ...)`
- `langchain_core.tools.tool` decorator
- `AgentMiddleware` hooks: `before_agent`, `before_model`, `after_model`, `after_agent`, `wrap_model_call`, `wrap_tool_call`, plus async `a*` versions.
- Built-in middleware in `langchain.agents.middleware`: `ModelCallLimitMiddleware(thread_limit, run_limit, exit_behavior)`, `ToolCallLimitMiddleware(tool_name, thread_limit, run_limit, exit_behavior)`, `ModelRetryMiddleware`, `ToolRetryMiddleware`, `ModelFallbackMiddleware(first_model, *additional_models)`, `SummarizationMiddleware(model, trigger, keep)`, `HumanInTheLoopMiddleware(interrupt_on)`.
- Structured output strategies in `langchain.agents.structured_output`: `ToolStrategy`, `ProviderStrategy`, `AutoStrategy`.
- Checkpointers: `langgraph.checkpoint.memory.InMemorySaver`, `langgraph.checkpoint.sqlite.SqliteSaver`. Store: `langgraph.store.memory.InMemoryStore`.
- Stream modes: `values`, `updates`, `checkpoints`, `tasks`, `debug`, `messages`, `custom`.
- Content blocks in `langchain_core.messages.content`: `TextContentBlock`, `ImageContentBlock`, `AudioContentBlock`, `FileContentBlock` and others.
- Model profiles: `BaseChatModel.profile`. `ChatGoogleGenerativeAI(model="gemini-2.5-flash").profile` reports `tool_calling`, `structured_output`, `image_inputs`, `audio_inputs`, `max_input_tokens` and more. Source data ships in `langchain_google_genai/data/_profiles.py` and `profile_augmentations.toml`.
- `langchain-tests` ships suites for chat models, tools, embeddings, stores, caches, retrievers, vector stores and sandboxes. It has no middleware suite.
- `AgentMiddleware.wrap_model_call(request, handler)` may call `handler` more than once. The uncertainty gate uses this to sample extra proposals. `request.override(model_settings={...})` passes settings such as `temperature` to `bind_tools`.
- `ExtendedModelResponse(model_response, command)` lets `wrap_model_call` return a state update. The gate uses it to append to `gate_log`.
- `dynamic_prompt` builds the system prompt per model call from `request.runtime.context`. Used to add the run's clock.
- `create_agent(context_schema=...)` and `ToolRuntime[Context]` pass per-run facts (now, timezone, workspace file) to tools.
- `ToolErrorMiddleware`, `InMemoryRateLimiter`: as in Claimcheck.

## Gemini API facts checked on 2026-10-02

- `gemini-3.1-flash-lite` accepts `temperature`. `gemini-3.5-flash-lite` uses fixed sampling and ignores it with a warning.
- Flash models allow 20 requests a day on the free tier, so the agent runs on Flash-Lite.
