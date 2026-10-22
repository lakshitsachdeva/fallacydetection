import { useEffect, useState } from "react"
import { Moon, Sun, TriangleAlert } from "lucide-react"

import { AnalyzePanel } from "@/components/analyze-panel"
import { MethodPanel } from "@/components/method-panel"
import { ResultsPanel } from "@/components/results-panel"
import { Alert, AlertDescription, AlertTitle } from "@/components/reui/alert"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type Meta, type Summary } from "@/lib/api"

function useTheme() {
  const [dark, setDark] = useState(() => {
    try {
      const saved = localStorage.getItem("theme")
      return saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches
    } catch {
      return false
    }
  })
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    try {
      localStorage.setItem("theme", dark ? "dark" : "light")
    } catch {
      /* storage unavailable */
    }
  }, [dark])
  return [dark, setDark] as const
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [summary, setSummary] = useState<Summary | null>(null)
  const [model, setModel] = useState<string>("")
  const [error, setError] = useState<string | null>(null)
  const [dark, setDark] = useTheme()

  useEffect(() => {
    api
      .meta()
      .then((m) => {
        setMeta(m)
        setModel(m.default_model ?? m.models[0]?.name ?? "")
      })
      .catch(() => setError("Cannot reach the FallacyLens API. Start it with: python -m uvicorn app.server:app --port 8000"))
    api.results().then(setSummary).catch(() => setSummary(null))
  }, [])

  const items = meta?.models.map((m) => ({ value: m.name, label: m.display })) ?? []

  return (
    <div className="bg-background text-foreground min-h-screen">
      <header className="bg-background/95 sticky top-0 z-10 border-b backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="min-w-0">
            <h1 className="text-base font-semibold tracking-tight">FallacyLens</h1>
            <p className="text-muted-foreground truncate text-xs">Logical fallacy classification on LOGIC · 13 classes</p>
          </div>
          <div className="flex items-center gap-2">
            {meta && meta.models.length > 0 && (
              <Select items={items} value={model} onValueChange={(v) => v && setModel(String(v))}>
                <SelectTrigger className="w-52" aria-label="Model">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {items.map((it) => (
                    <SelectItem key={it.value} value={it.value}>
                      {it.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            <Button variant="outline" size="icon" onClick={() => setDark(!dark)} aria-label="Toggle dark mode">
              {dark ? <Sun /> : <Moon />}
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-8">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <TriangleAlert />
            <AlertTitle>API unavailable</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        <Tabs defaultValue="analyze" className="gap-6">
          <TabsList>
            <TabsTrigger value="analyze">Analyze</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="method">Method</TabsTrigger>
          </TabsList>
          <TabsContent value="analyze">
            {meta ? <AnalyzePanel meta={meta} model={model} /> : !error && <Skeleton className="h-96 w-full" />}
          </TabsContent>
          <TabsContent value="results">
            {summary ? (
              <ResultsPanel summary={summary} />
            ) : (
              <p className="text-muted-foreground text-sm">Results appear here once `python -m fallacylens.analyze` has run.</p>
            )}
          </TabsContent>
          <TabsContent value="method">
            <MethodPanel />
          </TabsContent>
        </Tabs>
      </main>

      <footer className="text-muted-foreground mx-auto max-w-6xl px-4 pb-8 text-xs sm:px-6">
        Data: LOGIC and LOGICCLIMATE (Jin et al., Findings of EMNLP 2022). Predictions are model hypotheses, not judgements
        of truth.
      </footer>
    </div>
  )
}
