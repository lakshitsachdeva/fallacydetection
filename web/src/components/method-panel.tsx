import { Badge } from "@/components/reui/badge"
import {
  Stepper,
  StepperDescription,
  StepperIndicator,
  StepperItem,
  StepperNav,
  StepperSeparator,
  StepperTitle,
  StepperTrigger,
} from "@/components/reui/stepper"
import {
  Timeline,
  TimelineContent,
  TimelineDate,
  TimelineHeader,
  TimelineIndicator,
  TimelineItem,
  TimelineSeparator,
  TimelineTitle,
} from "@/components/reui/timeline"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

const PIPELINE = [
  { title: "Clean", text: "Pinned LOGIC release, conflicting labels dropped, train/dev overlap removed." },
  { title: "Tokenise", text: "Words and punctuation kept. Connectives such as \"therefore\" carry signal." },
  { title: "Encode", text: "CNN, BiLSTM or DistilBERT turns tokens into one sentence vector." },
  { title: "Classify", text: "Softmax over the 13 fallacy types; best epoch chosen on dev macro-F1." },
  { title: "Explain", text: "Word occlusion shows which tokens moved the prediction." },
]

const MODELS = [
  {
    name: "CNN",
    tag: "local patterns",
    body: "Filters of width 3, 4 and 5 slide over word embeddings and keep the strongest match for each filter. Good at short cue phrases such as \"everyone knows\".",
  },
  {
    name: "BiLSTM + attention",
    tag: "word order",
    body: "Reads the sentence left-to-right and right-to-left, then an attention layer weights the time steps that matter most for the label.",
  },
  {
    name: "DistilBERT",
    tag: "pretrained context",
    body: "A 66M-parameter Transformer pretrained on general English and fine-tuned here. Every word attends to every other word, so meaning depends on context.",
  },
]

const PHASES = [
  { date: "Phase 1", title: "Literature review and CNN baseline", text: "Ten verified papers, dataset audit, LOGIC chosen, first CNN." },
  { date: "Phase 2", title: "Controlled architecture comparison", text: "TF-IDF, CNN, BiLSTM and DistilBERT under one leakage-controlled protocol, three seeds each." },
  { date: "Phase 3", title: "Robustness and explanation", text: "Class weighting, climate-domain transfer, significance tests, word-level explanations, this app." },
]

export function MethodPanel() {
  return (
    <div className="flex flex-col gap-6">
      <Card className="min-w-0">
        <CardHeader>
          <CardTitle>Pipeline</CardTitle>
          <CardDescription>The same five steps run for every model, from raw text to an explained prediction.</CardDescription>
        </CardHeader>
        <CardContent>
          <Stepper defaultValue={5} orientation="vertical" className="md:hidden">
            <StepperNav>
              {PIPELINE.map((s, i) => (
                <StepperItem key={s.title} step={i + 1} className="items-start">
                  <StepperTrigger className="items-start gap-3 text-left">
                    <StepperIndicator>{i + 1}</StepperIndicator>
                    <div className="flex flex-col gap-1 pb-4">
                      <StepperTitle>{s.title}</StepperTitle>
                      <StepperDescription>{s.text}</StepperDescription>
                    </div>
                  </StepperTrigger>
                </StepperItem>
              ))}
            </StepperNav>
          </Stepper>
          <Stepper defaultValue={5} className="hidden md:block">
            <StepperNav>
              {PIPELINE.map((s, i) => (
                <StepperItem key={s.title} step={i + 1} className="relative flex-1 flex-col items-start">
                  <div className="flex w-full items-center">
                    <StepperTrigger>
                      <StepperIndicator>{i + 1}</StepperIndicator>
                    </StepperTrigger>
                    {i < PIPELINE.length - 1 && <StepperSeparator className="bg-border mx-2 h-0.5 flex-1" />}
                  </div>
                  <div className="mt-3 flex flex-col gap-1 pr-4">
                    <StepperTitle>{s.title}</StepperTitle>
                    <StepperDescription className="leading-snug">{s.text}</StepperDescription>
                  </div>
                </StepperItem>
              ))}
            </StepperNav>
          </Stepper>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-3">
        {MODELS.map((m) => (
          <Card key={m.name} className="min-w-0">
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center gap-2">
                {m.name} <Badge variant="secondary">{m.tag}</Badge>
              </CardTitle>
              <CardDescription className="leading-relaxed">{m.body}</CardDescription>
            </CardHeader>
          </Card>
        ))}
      </div>

      <Card className="min-w-0">
        <CardHeader>
          <CardTitle>Project phases</CardTitle>
        </CardHeader>
        <CardContent>
          <Timeline defaultValue={3}>
            {PHASES.map((p, i) => (
              <TimelineItem key={p.date} step={i + 1}>
                <TimelineHeader>
                  <TimelineSeparator />
                  <TimelineDate>{p.date}</TimelineDate>
                  <TimelineTitle>{p.title}</TimelineTitle>
                  <TimelineIndicator />
                </TimelineHeader>
                <TimelineContent>{p.text}</TimelineContent>
              </TimelineItem>
            ))}
          </Timeline>
        </CardContent>
      </Card>
    </div>
  )
}
