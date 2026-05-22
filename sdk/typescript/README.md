# @rtmdk/client

TypeScript SDK for RTMDK (Resonance-Topological Memory).

## Installation

```bash
npm install @rtmdk/client
```

## Usage

```typescript
import { RTMDKClient } from "@rtmdk/client";

const client = new RTMDKClient({
  baseUrl: "http://localhost:8080",
  apiKey: "your-api-key", // optional
});

// Add memory
await client.addNode({ text: "The sky is blue" });

// Query
const results = await client.retrieveNodes("What color is the sky?", 3);
console.log(results);
```

## API

### `RTMDKClient`

- `health()` — Check server health
- `addNode(req)` — Add a memory node
- `query(req)` — Query with pipeline or legacy mode
- `retrieveNodes(query, topK, sessionId?)` — Simple retrieval
- `deleteNode(id)` — Delete a node
- `stats()` — Get memory statistics

## License

AGPL-3.0
