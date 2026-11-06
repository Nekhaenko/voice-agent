# Spec Delta

## Purpose

Turns a recognized user utterance into an agent answer. The agent converses over a persistent state with the orcarouter LLM and optionally enriches its answer with a fresh DuckDuckGo search, chosen by the LLM itself.

## ADDED Requirements

### Requirement: LLM answers the user
The system SHALL pass the recognized transcript to an LLM and return its answer as the agent's response. The LLM SHALL be reached through the OpenAI-compatible API at `https://api.orcarouter.ai/v1` using the `orcarouter/free` model, with the API key taken from the `ORCAROUTER_API_KEY` environment variable.

#### Scenario: Question that needs no search
- **WHEN** the user asks a question the model can answer without new external information, and the model returns an answer
- **THEN** the answer is shown as the agent's response with no web search performed

#### Scenario: Missing API key
- **WHEN** `ORCAROUTER_API_KEY` is not set and a user exchange completes transcription
- **THEN** the interface reports a configuration error and no answer is produced

### Requirement: Conditional DuckDuckGo search
The LLM SHALL decide whether the question needs information the model cannot supply; when it does, the system SHALL run a DuckDuckGo web search over the internet and the LLM SHALL incorporate the retrieved results into its final response. When the LLM does not request a search, the system SHALL answer directly without contacting DuckDuckGo.

#### Scenario: Search is requested and used
- **WHEN** the LLM decides a search is needed for the question
- **THEN** the system performs a DuckDuckGo search and the final response incorporates the retrieved information

#### Scenario: Search is skipped
- **WHEN** the LLM decides no search is needed
- **THEN** no web search runs and the final response answers directly

#### Scenario: Search fails
- **WHEN** the LLM requests a search but the search cannot be completed
- **THEN** the agent still produces a response, stating it could not retrieve fresh information

### Requirement: Conversation memory persists across turns
The conversation state, including prior exchanges, SHALL be persisted by a checkpointer across turns, so later questions resolve against the dialogue history.

#### Scenario: Follow-up question references history
- **WHEN** the user asks a follow-up that only makes sense with the earlier exchange, and no search is requested
- **THEN** the response reflects the earlier exchange

#### Scenario: New session starts fresh
- **WHEN** the interface starts a new conversation
- **THEN** earlier sessions' turns are not part of the agent's context
