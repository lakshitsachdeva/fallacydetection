import model from '../lib/model-data.js'
import summary from '../lib/summary-data.js'
import descriptions from '../lib/descriptions-data.js'
import { MODEL_NAME } from '../lib/baseline.js'

export default {
  fetch() {
    const experiment = summary.experiments.find((item) => item.exp === MODEL_NAME)
    return Response.json({
      default_model: MODEL_NAME,
      models: [{
        name: MODEL_NAME,
        display: 'TF-IDF + LR (live)',
        family: 'tfidf',
        class_weighted: false,
        n_params: model.n_params,
        strict_macro_f1: experiment.metrics.test_strict.macro_f1,
      }],
      device: 'Vercel Node.js',
      labels: model.labels.map((id) => ({ id, ...descriptions[id] })),
      demo_note: 'Live inference uses the measured TF-IDF baseline. The other model results are saved experiments; their checkpoints are not deployed.',
    })
  },
}
