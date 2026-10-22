import { Badge } from "@/components/reui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { f3, params, type Summary } from "@/lib/api"

export function ResultsPanel({ summary }: { summary: Summary }) {
  const exps = summary.experiments
  const selected = exps.find((e) => e.exp === summary.selected)
  const best = Math.max(...exps.map((e) => e.metrics.test_strict.macro_f1))
  const rows = summary.protocol.final_rows

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        <Stat label="Training rows" value={rows.train.toLocaleString()} note="after conflict and overlap cleaning" />
        <Stat label="Strict test rows" value={String(rows.test_strict)} note={`${summary.protocol.test_rows_seen_in_train_or_dev} seen rows excluded`} />
        <Stat
          label="Selected model"
          value={selected?.display ?? "—"}
          note={`chosen on dev macro-F1 (${f3(selected?.metrics.dev.macro_f1)})`}
        />
        <Stat label="Best strict macro-F1" value={f3(best)} note={`mean of ${selected?.seeds.length ?? 0} seeds`} />
      </div>

      <Card className="min-w-0">
        <CardHeader>
          <CardTitle>Model comparison</CardTitle>
          <CardDescription>
            Mean ± standard deviation over seeds. Strict test excludes any text seen in train or dev. LOGICCLIMATE is
            direct transfer with no climate training. "cw" marks class-weighted cross-entropy.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">Params</TableHead>
                <TableHead className="text-right">Dev macro-F1</TableHead>
                <TableHead className="text-right">Strict accuracy</TableHead>
                <TableHead className="text-right">Strict macro-F1</TableHead>
                <TableHead className="text-right">Climate macro-F1</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {exps.map((e) => (
                <TableRow key={e.exp}>
                  <TableCell className="font-medium whitespace-nowrap">
                    <span className="flex items-center gap-2">
                      {e.display}
                      {e.exp === summary.selected && <Badge variant="success-light">selected</Badge>}
                    </span>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-right tabular-nums">{params(e.n_params)}</TableCell>
                  <MetricCell e={e} split="dev" />
                  <MetricCell e={e} split="test_strict" metric="accuracy" />
                  <MetricCell e={e} split="test_strict" strong={e.metrics.test_strict.macro_f1 === best} />
                  <MetricCell e={e} split="climate_test" />
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="min-w-0">
          <CardHeader>
            <CardTitle>Per-class F1 — {selected?.display}</CardTitle>
            <CardDescription>Strict test, mean over seeds. Classes ordered by number of training examples (middle column).</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-2.5">
            {summary.labels
              .map((l, i) => ({ l, n: summary.train_counts[i], f: selected?.per_class_f1_strict[i] ?? 0 }))
              .sort((a, b) => b.n - a.n)
              .map(({ l, n, f }) => (
                <div key={l} className="grid grid-cols-[minmax(0,10.5rem)_2.25rem_minmax(0,1fr)_2.5rem] items-center gap-3 text-sm">
                  <span className="leading-snug break-words capitalize">{l}</span>
                  <span className="text-muted-foreground text-right text-xs tabular-nums">{n}</span>
                  <div className="bg-muted h-2 overflow-hidden rounded-full">
                    <div className="h-full rounded-full bg-sky-600" style={{ width: `${Math.max(1, f * 100)}%` }} />
                  </div>
                  <span className="text-muted-foreground text-right tabular-nums">{f.toFixed(2)}</span>
                </div>
              ))}
          </CardContent>
        </Card>

        <Card className="min-w-0">
          <CardHeader>
            <CardTitle>Is the selected model really better?</CardTitle>
            <CardDescription>
              Paired bootstrap (2,000 resamples) of the macro-F1 difference on the strict test, seed 42, plus an exact
              McNemar test. An interval that contains 0 means no reliable difference.
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>vs.</TableHead>
                  <TableHead className="text-right">Δ macro-F1</TableHead>
                  <TableHead className="text-right">95% CI</TableHead>
                  <TableHead className="text-right">p</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {summary.significance.map((s) => {
                  const sig = s.bootstrap_95ci[0] > 0 || s.bootstrap_95ci[1] < 0
                  return (
                    <TableRow key={s.against}>
                      <TableCell className="font-medium whitespace-nowrap">{s.display}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {s.delta_macro_f1_seed42 >= 0 ? "+" : ""}
                        {s.delta_macro_f1_seed42.toFixed(3)}
                      </TableCell>
                      <TableCell className="text-right whitespace-nowrap tabular-nums">
                        [{s.bootstrap_95ci[0].toFixed(3)}, {s.bootstrap_95ci[1].toFixed(3)}]
                        {sig && (
                          <Badge variant="info-light" className="ml-2">
                            sig.
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-muted-foreground text-right tabular-nums">
                        {s.p_value < 0.001 ? "<0.001" : s.p_value.toFixed(3)}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="truncate text-2xl tabular-nums" title={value}>
          {value}
        </CardTitle>
        <p className="text-muted-foreground text-xs">{note}</p>
      </CardHeader>
    </Card>
  )
}

function MetricCell({
  e,
  split,
  metric = "macro_f1",
  strong,
}: {
  e: Summary["experiments"][number]
  split: string
  metric?: string
  strong?: boolean
}) {
  return (
    <TableCell className={`text-right whitespace-nowrap tabular-nums ${strong ? "font-semibold" : ""}`}>
      {f3(e.metrics[split][metric])}
      <span className="text-muted-foreground text-xs"> ± {f3(e.metrics_std[split][metric])}</span>
    </TableCell>
  )
}
