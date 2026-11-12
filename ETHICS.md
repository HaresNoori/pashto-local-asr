# Ethics and responsible use

This tool exists because people working with Pashto and Dari audio often handle sensitive recordings and have few safe options. It is built to keep that audio on the user's own machine. These principles are part of how the tool is meant to be used.

## Consent
- Transcribe only audio you recorded, or that you have clear permission to process.
- Where people are identifiable, they should know their speech is being transcribed, and why. Respect any limits they set on use or retention.
- Interviews with people at risk need extra care: consider whether even a local transcript file could expose them if a device is seized, and store or delete it accordingly.

## Local-only processing
- The tool decodes and transcribes on the local CPU. It does not upload audio or transcripts anywhere.
- The only network use is the one-off download of the model files from the Hugging Face Hub. After that, set `HF_HUB_OFFLINE=1` to run fully offline.
- Do not wrap this tool in a service that sends audio or text to third parties without the speaker's informed consent.

## No surveillance, no covert recording
- Do not use this tool to monitor, profile or identify people, or to process audio that was recorded covertly or unlawfully.
- Do not use it to build watch-lists or to support targeting of individuals or communities.

## Outputs are drafts
- Every output is marked `unverified_draft`. Machine transcripts of Pashto and Dari contain errors, and on spontaneous Pashto the default model gets about 70% of words wrong (see the benchmark in the README).
- A passed quality screen is not evidence of accuracy. Have a fluent speaker review any transcript before it is quoted, published, relied on in a legal or humanitarian setting, or used to make a decision about a person.
- Keep the recorded SHA-256 with any transcript so it can always be traced to the exact audio and model that produced it.

## Reporting problems
If you find that the tool has been used in a way that conflicts with this document, or that it produces systematically harmful errors for a dialect or group of speakers, please open an issue.
