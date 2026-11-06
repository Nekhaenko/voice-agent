# Spec Delta

## Purpose

Lets a user speak into a browser and see, as text, what they said and the agent's answer. Covers the push-to-talk toggle, offline VOSK transcription of the Russian utterance, and the display loop in the Gradio interface.

## ADDED Requirements

### Requirement: Push-to-talk toggle recording
The user interface SHALL use a single toggle button: the first press starts recording, and the next press stops recording. Recording SHALL capture audio from the user's microphone in the browser session running the interface.

#### Scenario: Start recording
- **WHEN** the user presses the toggle button while not recording
- **THEN** the system starts capturing microphone audio and signals to the user that recording has started

#### Scenario: Stop recording
- **WHEN** the user presses the toggle button again while recording
- **THEN** the system stops capturing audio and hands the recorded utterance to transcription

### Requirement: Offline batch transcription
When recording stops, the system SHALL transcribe the recorded utterance in one pass using a locally installed VOSK model, without any network speech service. Transcription SHALL be in Russian.

#### Scenario: Speaker is transcribed
- **WHEN** recording stops after speech captured in Russian
- **THEN** the system displays the recognized text

#### Scenario: Empty recording
- **WHEN** recording stops with no recognizable speech
- **THEN** the system reports that nothing was recognized and does not query the agent

### Requirement: Conversation is displayed as text
The interface SHALL display both the user's transcript and the agent's response as text. The system SHALL NOT produce spoken audio for the response.

#### Scenario: Agent answer appears
- **WHEN** the agent produces a response for a recognized transcript
- **THEN** the transcript and the response are both visible in the interface

#### Scenario: Multi-turn history is visible
- **WHEN** the user completes more than one exchange with the agent
- **THEN** the interface retains and shows the prior exchanges

### Requirement: Microphone access failure is reported
The system SHALL report a clear error when the browser does not grant microphone access or the interface cannot start.

#### Scenario: Microphone denied
- **WHEN** the user has not granted microphone access and presses the toggle button
- **THEN** the interface shows an error naming the microphone problem, and no recording is produced
