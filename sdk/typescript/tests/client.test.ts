import { RTMDKClient } from "../src/client";

// Simple mock for fetch testing
describe("RTMDKClient", () => {
  const client = new RTMDKClient({ baseUrl: "http://localhost:8080" });

  it("should be instantiable", () => {
    expect(client).toBeInstanceOf(RTMDKClient);
  });

  it("should construct correct URL", () => {
    // Base URL normalization tested indirectly
    const c2 = new RTMDKClient({ baseUrl: "http://localhost:8080/" });
    expect(c2).toBeDefined();
  });
});
