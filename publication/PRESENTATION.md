# Finished experiment: presentation artifacts

The [X post](twitter-post.txt) is a **draft only**. No post was published to X.
The intended attachment is `direct-weight-chat-v17a-compact.mp4`, a 105-second,
1920 x 1080, 30 fps H.264 video with no audio. It is rendered by
`python -m publication.render_video --output build/showcase-compact` using Pillow
and FFmpeg. The output is local; videos are not tracked in source control.

The [captured transcript](demo-transcript-compact.json) contains actual compact
model outputs and token IDs from 13 turns across four conversations. Context
continues within each conversation and resets between chapters. The UI film
replays those exact responses with edited pacing for readability; it is **not a
live speed recording**. Both known failures are retained. The final card says
useful general chat was not achieved.

The video labels the exact packed weight size: **1,528,592 bytes**. Compression changes storage, not model capability. The
[render record](video-render-compact.json) binds the video to its transcript and
weight hash. [Media verification](video-verification-compact.json) records
FFprobe properties, a full decode and review frames.

The [local UI](demo_ui/index.html) offers separately labelled captured replay
and actual offline inference via `serve_demo.py` and `model_worker.py`. It
does not synthesize, repair, choose or fall back to canned model responses.
The transport layer forwards text to the frozen transformer runtime.

![Live compact model using a supplied fact and a later correction](compact-ui.jpg)

## Additional requested capability probes

[Math probes](math-showcase-probes.json) retain the actual boundary arithmetic
and unsupported multiplication, division, square-root and equation requests.
[Conditional/extra-operand probes](conditional-math-probes.json) include normal
two-operand arithmetic controls, three/four operands, fact-based and numeric
conditions, and conditional arithmetic after a derived rule premise.

These targeted demonstrations do not establish broad mathematical ability.
The two-operand controls work; combining arithmetic with `if` or adding more
operands does not become supported merely because those circuits exist
separately. A false-condition abstention is not evidence of working branching
when the true-condition request also fails.
