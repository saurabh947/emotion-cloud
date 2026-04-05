# emotion-cloud

Cloud inference service for the [emotion-detection-action](https://github.com/saurabh947/emotion-detection-action) SDK.
Serves the Two-Tower Multimodal Transformer (ViT video + emotion2vec audio) as a gRPC streaming API on Google Cloud.

Deployed as a Docker container on a Compute Engine VM (`n1-standard-4` + NVIDIA T4, `us-east1-c`).
Managed via `make vm-*` targets — no Kubernetes.

## Project layout

```
api/          gRPC server + schemas
models/       TorchServe handler, GCS weight loader
config/       Pydantic settings (env-driven)
deploy/
  docker/     Dockerfile + docker-compose (local dev)
  vm/         GCE VM startup script
scripts/      Weight download, model packaging
main.py       App entrypoint
```

## Skills

| Skill | When to use |
|---|---|
| `/browse` | All web browsing — never use mcp chrome tools directly |
| `/investigate` | Debugging inference issues, latency regressions |
| `/review` | Pre-merge code review |
| `/ship` | Ship a feature end-to-end |
| `/qa` | Test the API end-to-end and fix issues |
| `/qa-only` | Test without making changes |
| `/design-review` | Review API design or UI |
| `/benchmark` | Measure inference latency / throughput regressions |
| `/canary` | Post-deploy health monitoring |
| `/careful` | Before any destructive infra changes |
| `/cso` | Security audit (secrets, supply chain, LLM trust boundaries) |
| `/retro` | Weekly engineering retrospective |
| `/autoplan` | Full plan review pipeline (CEO + eng + design) |
| `/plan-eng-review` | Engineering plan review |
| `/document-release` | Update docs after shipping |

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
