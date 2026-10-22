export type Label = { id: string; name: string; description: string }
export type ModelInfo = {
  name: string
  display: string
  family: string
  class_weighted: boolean
  n_params: number
  strict_macro_f1: number | null
}
export type Meta = { default_model: string | null; models: ModelInfo[]; labels: Label[]; device: string }
export type Analysis = {
  model: string
  label: string
  confidence: number
  top: { label: string; prob: number }[]
  tokens: { text: string; weight: number }[]
  truncated?: boolean
}
export type Experiment = {
  exp: string
  display: string
  family: string
  class_weighted: boolean
  n_params: number
  seeds: number[]
  train_seconds: { mean: number; std: number }
  metrics: Record<string, Record<string, number>>
  metrics_std: Record<string, Record<string, number>>
  per_class_f1_strict: number[]
}
export type Summary = {
  labels: string[]
  train_counts: number[]
  selected: string
  protocol: { final_rows: Record<string, number>; test_rows_seen_in_train_or_dev: number }
  experiments: Experiment[]
  significance: {
    against: string
    display: string
    delta_macro_f1_seed42: number
    bootstrap_95ci: [number, number]
    p_value: number
  }[]
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new Error(detail?.detail ? String(typeof detail.detail === "string" ? detail.detail : "Invalid input") : `Request failed (${res.status})`)
  }
  return res.json()
}

export const api = {
  meta: () => request<Meta>("/api/meta"),
  results: () => request<Summary>("/api/results"),
  examples: () => request<{ examples: { text: string; label: string }[] }>("/api/examples"),
  analyze: (text: string, model: string) => request<Analysis>("/api/analyze", { text, model }),
  compare: (text: string) =>
    request<{ results: { model: string; label: string; confidence: number }[] }>("/api/compare", { text }),
}

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`
export const f3 = (x: number | null | undefined) => (x == null ? "—" : x.toFixed(3))
export const params = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1e3)}K`)
