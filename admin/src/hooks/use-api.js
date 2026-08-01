import { useCallback } from "react"
import { useServer } from "@/context/server-context"

const API_BASE = "/api/rtmdk"

export function useApi() {
  const { config } = useServer()
  const apiKey = config?.env?.RTMDK_API_KEY || ""

  const authFetch = useCallback(async (url, options = {}) => {
    const headers = {
      ...(options.headers || {}),
    }
    if (apiKey) {
      headers["X-API-Key"] = apiKey
    }
    return fetch(url, { ...options, headers })
  }, [apiKey])

  return { apiBase: API_BASE, authFetch, apiKey }
}
