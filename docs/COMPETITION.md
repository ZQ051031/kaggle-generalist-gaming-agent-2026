# Competition fact check — 2026-11-24

Sources read logged out: Kaggle [Overview](https://www.kaggle.com/competitions/generalist-gaming-agent/overview) (Description, Evaluation, Timeline, Prizes and Code Requirements), [Data](https://www.kaggle.com/competitions/generalist-gaming-agent/data) and [Rules](https://www.kaggle.com/competitions/generalist-gaming-agent/rules). The Data description is public; the actual files require accepting the rules. No account was used and no competition files were downloaded.

| Fact | Status | Evidence / clarification |
| --- | --- | --- |
| GARC (Generalist Agent Research Collective), Research Code Competition, $50,000 | Confirmed | Overview header and Prizes & Awards |
| Final submission: March 1, 2027 23:59 UTC = March 2, 07:59 China Standard Time (UTC+8) | Confirmed | Overview → Timeline. Do not interpret “CST” as US Central Standard Time. |
| Entry and team merger: February 22, 2027 23:59 UTC = February 23, 07:59 China Standard Time | Confirmed | Overview → Timeline |
| Prizes: $16k / $12k / $9k / $7k / $6k | Confirmed | Overview → Prizes |
| Solo entry, team max 5, max 5 submissions daily | Confirmed | Rules B.1–2 and C.5 |
| China not excluded; minimum age 18 | Confirmed with caveats | Rules C.1 excludes Crimea, DNR/LNR, Cuba, Iran and North Korea, and applies US sanctions/export controls and local-law restrictions. Minimum is the greater of 18 and the local age of majority, subject to the stated consent exception. Residence alone does not determine eligibility. |
| One offline, pixel-based agent plays 32 games across 10 genres, 10 minutes each, back-to-back | Confirmed | Overview → Description; Data → Games and Your Agent. Reset per-game state when `obs.elapsed_s` rolls back near zero. |
| JPEG, current viewport coordinates, frames since last decision point and up to 8 earlier frames | Confirmed | Data → Observation. Earlier frames are sampled since the previous observation, evenly spread when there are more. |
| Brief, remaining time, dimensions, prior action result and schemas | Differs in field placement | Data → Observation and Your Agent: `obs.info` supplies brief (`description`, `controls`), `remaining_s`, `width`, `height`; `obs.last_action` is separate, and `tools: list[ToolSpec]` is the second argument to `step`. |
| Action has 11 tools | Confirmed | Data → Actions: click, move, drag, scroll, press_key, hold_keys, key_down, key_up, wait, sequence (up to 200 timed steps), skip. Only skip preserves pending inputs; refused actions do not change inputs. |
| 20 fps, decisions every 4 frames, 90 seconds per decision, 15 minutes startup | Confirmed | Data → Timing and Time Limits. The game does not pause. A reply lands at the next decision boundary; a timed-out game retains its best score so far. An exception or unusable reply plays skip. |
| Notebook runtime ≤9h, no internet, RTX Pro 6000 (96 GB) | Confirmed with caveat | Overview → Code Requirements. The GPU is available, not required: CPU notebooks are also allowed under the same cap. |
| Four sample games shipped; other 28 never distributed | Confirmed | Data → Games. Game ID, title and genre reach the agent only for sample games; this policy doesn't use them. |
| Normalized 0–1 vs human baseline, unweighted average, public 16 incl. samples, private 16 decide rank | Confirmed | Overview → Evaluation and Data → Scoring/Leaderboard. Normalize by clipping `(value - floor) / (human - floor)`, where doing nothing sets the floor. Best private score among up to two selected submissions counts. |
| Freely/publicly available data and pretrained models allowed, no private test data for development | Confirmed | Overview → Code Requirements and Rules B.6 / C.4. Rules qualify external resources with accessibility and reasonableness. No private test data were accessed or used. |
| Complete field/type definitions and tool schema details | Not found in the public descriptions | Data points to the gated `gga/protocol.py`. The repository entry point imports the actual types at runtime and filters actions against the supplied tool schemas. Exact field internals and real evaluator behavior remain untested. |

Data → Submission specifies `/kaggle/working/agent.py` and the final cell
`GGAInferenceServer().run()`. A normal notebook run writes a placeholder
`submission.parquet`, not a competition score. Real scoring occurs during
Kaggle's rerun. Data → Testing Locally describes the sample browser runner;
it does not report scores.
