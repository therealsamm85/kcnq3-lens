# Medical Disclaimer

**KCNQ3-Lens is NOT a medical device. It does NOT provide medical diagnosis, treatment, or advice.**

## What this tool is

KCNQ3-Lens is a research-grade quantitative analysis tool for EEG recordings. It surfaces patterns in EEG data (spike topography, sleep spindle density, background slowing, sustained rhythmic bursts, spike morphology) that may be useful for families and clinicians to discuss together.

The tool is intended to:

- Help families understand what their child's EEG shows quantitatively
- Generate structured questions that families can bring to their treating neurologist
- Enable consistent measurement of EEG features across time and across recordings, so changes over months and years can be tracked

## What this tool is NOT

- **Not a diagnostic tool.** No output of this tool establishes, confirms, or rules out any medical condition.
- **Not a substitute for clinical EEG interpretation.** Only a qualified neurologist or epileptologist can interpret EEG data in the context of a patient's full clinical picture.
- **Not a basis for medication decisions.** No treatment should be started, stopped, or modified based on this tool's output. All therapeutic decisions belong to the treating physician.
- **Not validated against clinical gold standards.** The algorithms used are research-grade and have not been validated as medical-grade software.

## Privacy

- EEG files are processed **entirely on your local machine**. They are never uploaded to any server.
- If you choose to use the optional AI interpretation feature, only **derived numerical metrics** (e.g., "spindle density: 1.3 per minute") are sent to the Anthropic API, never raw EEG data.
- You are responsible for the security of EEG files on your own device.

## Limitations

- The algorithms make assumptions (sampling rate, channel naming, recording quality) that may not hold for all recordings.
- Age-normative reference ranges built into the tool come from published literature and may not represent all populations.
- The tool was developed with KCNQ3-spectrum patients in mind. Use for other conditions is at the user's discretion.
- The Nihon Kohden EEG-1200A reader is reverse-engineered from a single recording family. Other recordings in this format may not parse correctly.

## Use

By using this software, you acknowledge that:

1. You understand it is not a medical device.
2. You will not make medical decisions based solely on its output.
3. You will share any findings of interest with a qualified physician before acting on them.
4. The authors and contributors assume no liability for any use of this software.

If you are uncertain about anything this tool reports — bring the report to your child's doctor and ask. That is what the tool is built for.
