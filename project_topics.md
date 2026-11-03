# Computer Vision — Final Project Topics

**Weight:** 50% of the course grade · **Teams:** ~6 members · **Released:** Week 3 · **Presented:** Week 15

*17 topics.*

Each topic below is scoped for a single consumer GPU or free Colab: fine-tuning pretrained models, small
or subsampled datasets, and inference-time methods. None require training a large model from scratch.

Every topic lists:

- **Core task** — the minimum for a complete, passing project
- **Extensions** — the work that separates a good project from an excellent one
- **Publication angle** — the direction to push if your team wants a workshop paper (talk to me early)
- **Split for 6** — how the work divides so nobody is idle

---

### 16. From photos to 3D: NeRF and Gaussian splatting in practice

- **Core task:** capture your own object or scene with a phone; reconstruct it with both a NeRF variant and Gaussian splatting; compare quality, training time, and rendering speed.
- **Extensions:** how few input views can you get away with? Sensitivity to capture pattern, motion blur, and lighting change; novel-view quality metrics.
- **Publication angle:** a practical capture guide backed by measurements — how casual can a phone capture be before reconstruction fails?
- **Split for 6:** 2 capture/preprocessing · 2 NeRF · 2 Gaussian splatting, with evaluation and writing shared.

## Choosing a topic

**For a solid grade**, pick from Group A. The path to a working system is clear and the risk is low.

**For a publication attempt**, Groups B and C suit you better — analysis and benchmark papers need far less compute
than new architectures, and workshop venues actively want careful negative results. Come talk to me in Week 4 or 5,
not Week 13.

## Deliverables

| Milestone                       | Week | Content                                                           |
| ------------------------------- | ---- | ----------------------------------------------------------------- |
| Team formation and topic choice | 4    | Team list, ranked topic preferences                               |
| Proposal                        | 5    | 2 pages: problem, data, method, evaluation plan, division of work |
| Milestone report                | 10   | 3 pages: baseline results, what broke, revised plan               |
| Final presentation              | 15   | 15 minutes plus questions, all members present                    |
| Code and report                 | 15   | Reproducible repository, 6–8 page report in conference format    |
