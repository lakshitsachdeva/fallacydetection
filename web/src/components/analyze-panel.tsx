import { useEffect, useMemo, useState } from "react"
import { Info, Loader2, Shuffle, TriangleAlert } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/reui/alert"
import { Badge } from "@/components/reui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { api, pct, type Analysis, type Meta } from "@/lib/api"

type Example = { text: string; label: string }
type Compared = { model: string; label: string; confidence: number }[]

export function AnalyzePanel({ meta, model }: { meta: Meta; model: string }) {
  const [text, setText] = useState("")
  const [examples, setExamples] = useState<Example[]>([])
  const [exampleLabel, setExampleLabel] = useState<string | null>(null)
  const [result, setResult] = useState<Analysis | null>(null)
  const [compared, setCompared] = useState<Compared | null>(null)
  const [busy, setBusy] = useState<"analyze" | "compare" | null>(null)
  const [error, setError] = useState<string | null>(null)

  const labelInfo = useMemo(() => new Map(meta.labels.map((l) => [l.id, l])), [meta.labels])
  const modelName = (name: string) => meta.models.find((m) => m.name === name)?.display ?? name

  useEffect(() => {
    api.examples().then((r) => setExamples(r.examples)).catch(() => setExamples([]))
  }, [])

  // A new model choice invalidates the shown analysis.
  useEffect(() => setResult(null), [model])

  const pickExample = () => {
    if (!examples.length) return
    const e = examples[Math.floor(Math.random() * examples.length)]
    setText(e.text)
    setExampleLabel(e.label)
    setResult(null)
    setCompared(null)
  }

  const run = async (kind: "analyze" | "compare") => {
    setBusy(kind)
    setError(null)
    try {
      if (kind === "analyze") setResult(await api.analyze(text, model))
      else setCompared((await api.compare(text)).results)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong")
    } finally {
      setBusy(null)
    }
  }

  const tooShort = text.trim().length < 3

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Card className="min-w-0">
        <CardHeader>
          <CardTitle>Argument</CardTitle>
          <CardDescription>
            Paste a short argument, or load a held-out example from the strict LOGIC test split.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <Textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value)
              setExampleLabel(null)
              setResult(null)
              setCompared(null)
            }}
            maxLength={2000}
            rows={7}
            placeholder="e.g. Everyone I know uses this phone, so it must be the best one on the market."
            className="min-h-40 resize-y text-[15px] leading-relaxed"
            aria-label="Argument text"
          />
          {exampleLabel && (
            <p className="text-muted-foreground text-sm">
              Dataset label for this example:{" "}
              <span className="text-foreground font-medium">{labelInfo.get(exampleLabel)?.name ?? exampleLabel}</span>
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => run("analyze")} disabled={tooShort || busy !== null}>
              {busy === "analyze" && <Loader2 className="animate-spin" />}
              Classify
            </Button>
            <Button variant="outline" onClick={() => run("compare")} disabled={tooShort || busy !== null}>
              {busy === "compare" && <Loader2 className="animate-spin" />}
              Compare all models
            </Button>
            <Button variant="ghost" onClick={pickExample} disabled={!examples.length}>
              <Shuffle /> Load example
            </Button>
          </div>
          {error && (
            <Alert variant="destructive">
              <TriangleAlert />
              <AlertTitle>Request failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          <Alert variant="info">
            <Info />
            <AlertTitle>How to read the output</AlertTitle>
            <AlertDescription>
              <p>
                The model always picks one of 13 LOGIC fallacy types. It has no "sound argument" class and does not
                check facts, so a label is a hypothesis about the argument's pattern, not a verdict.
              </p>
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      <div className="flex min-w-0 flex-col gap-6">
        {result ? (
          <ResultCard result={result} labelInfo={labelInfo} modelName={modelName(result.model)} />
        ) : (
          <Card className="min-w-0">
            <CardHeader>
              <CardTitle>Prediction</CardTitle>
              <CardDescription>Run Classify to see the predicted fallacy, the probability of each candidate, and which words drove the decision.</CardDescription>
            </CardHeader>
          </Card>
        )}
        {compared && <CompareCard rows={compared} labelInfo={labelInfo} modelName={modelName} />}
      </div>
    </div>
  )
}

function ResultCard({
  result,
  labelInfo,
  modelName,
}: {
  result: Analysis
  labelInfo: Map<string, Meta["labels"][number]>
  modelName: string
}) {
  const info = labelInfo.get(result.label)
  const maxAbs = Math.max(1e-6, ...result.tokens.map((t) => Math.abs(t.weight)))
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardDescription>Predicted by {modelName}</CardDescription>
        <CardTitle className="flex flex-wrap items-center gap-2 text-xl">
          <span className="break-words capitalize">{info?.name ?? result.label}</span>
          <Badge variant="primary-light" size="lg">
            {pct(result.confidence)} confidence
          </Badge>
        </CardTitle>
        <p className="text-muted-foreground text-xs">
          Dataset label: <span className="text-foreground">{result.label}</span>
        </p>
        {info?.description && (
          <p className="text-muted-foreground text-sm leading-relaxed first-letter:uppercase">{info.description}</p>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <section className="flex flex-col gap-2.5">
          <h3 className="text-sm font-medium">Top candidates</h3>
          {result.top.map((t) => (
            <div key={t.label} className="grid grid-cols-[minmax(0,13rem)_minmax(0,1fr)_3.5rem] items-center gap-3 text-sm">
              <span className="leading-snug break-words capitalize" title={t.label}>
                {labelInfo.get(t.label)?.name ?? t.label}
              </span>
              <div className="bg-muted h-2 overflow-hidden rounded-full">
                <div className="bg-primary h-full rounded-full" style={{ width: `${Math.max(1, t.prob * 100)}%` }} />
              </div>
              <span className="text-muted-foreground text-right tabular-nums">{pct(t.prob)}</span>
            </div>
          ))}
        </section>
        <Separator />
        <section className="flex flex-col gap-2.5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-sm font-medium">Word influence</h3>
            <div className="text-muted-foreground flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1.5">
                <span className="inline-block size-2.5 rounded-sm bg-sky-500/70" /> supports
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block size-2.5 rounded-sm bg-rose-500/70" /> works against
              </span>
            </div>
          </div>
          <p className="text-[15px] leading-8 break-words">
            {result.tokens.map((t, i) => {
              const a = Math.min(1, Math.abs(t.weight) / maxAbs)
              const color = t.weight >= 0 ? `rgba(14,165,233,${(a * 0.45).toFixed(3)})` : `rgba(244,63,94,${(a * 0.45).toFixed(3)})`
              return (
                <span key={i}>
                  <span
                    className="rounded px-0.5 py-0.5"
                    style={{ backgroundColor: color }}
                    title={`${t.weight >= 0 ? "+" : ""}${(t.weight * 100).toFixed(1)} pts when removed`}
                  >
                    {t.text}
                  </span>{" "}
                </span>
              )
            })}
          </p>
          {result.truncated && (
            <p className="text-warning-foreground text-xs">
              The text is longer than this model's input window, so only the words shown were used.
            </p>
          )}
          <p className="text-muted-foreground text-xs leading-relaxed">
            Occlusion: each word is deleted in turn and the drop in the predicted class's probability is measured. This shows
            what the model is sensitive to, not a proof of its reasoning.
          </p>
        </section>
      </CardContent>
    </Card>
  )
}

function CompareCard({
  rows,
  labelInfo,
  modelName,
}: {
  rows: Compared
  labelInfo: Map<string, Meta["labels"][number]>
  modelName: (n: string) => string
}) {
  const votes = new Map<string, number>()
  rows.forEach((r) => votes.set(r.label, (votes.get(r.label) ?? 0) + 1))
  const [top, n] = [...votes.entries()].sort((a, b) => b[1] - a[1])[0] ?? ["", 0]
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>All models</CardTitle>
        <CardDescription>
          {n} of {rows.length} models agree on{" "}
          <span className="text-foreground font-medium capitalize">{labelInfo.get(top)?.name ?? top}</span>.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Prediction</TableHead>
              <TableHead className="text-right">Confidence</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={r.model}>
                <TableCell className="font-medium">{modelName(r.model)}</TableCell>
                <TableCell className="capitalize">{labelInfo.get(r.label)?.name ?? r.label}</TableCell>
                <TableCell className="text-right tabular-nums">{pct(r.confidence)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
