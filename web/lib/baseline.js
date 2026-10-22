import model from './model-data.js'

export const MODEL_NAME = 'tfidf_lr'
export const LABELS = model.labels

function features(text) {
  // Mirrors scikit-learn's default word tokenization and (1, 2)-gram TF-IDF.
  const words = text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) ?? []
  const counts = new Map()
  for (let i = 0; i < words.length; i++) {
    counts.set(words[i], (counts.get(words[i]) ?? 0) + 1)
    if (i + 1 < words.length) {
      const pair = `${words[i]} ${words[i + 1]}`
      counts.set(pair, (counts.get(pair) ?? 0) + 1)
    }
  }
  const sparse = []
  let squared = 0
  for (const [term, count] of counts) {
    if (!Object.hasOwn(model.vocabulary, term)) continue
    const index = model.vocabulary[term]
    const value = (1 + Math.log(count)) * model.idf[index]
    sparse.push([index, value])
    squared += value * value
  }
  const norm = Math.sqrt(squared)
  return norm ? sparse.map(([i, value]) => [i, value / norm]) : []
}

export function predict(text) {
  const sparse = features(text)
  const scores = model.intercepts.map((intercept, c) => {
    let score = intercept
    const weights = model.coefficients[c]
    for (const [i, value] of sparse) score += weights[i] * value
    return score
  })
  const maximum = Math.max(...scores)
  const exps = scores.map((score) => Math.exp(score - maximum))
  const total = exps.reduce((a, b) => a + b, 0)
  return exps.map((value) => value / total)
}

export function classify(text) {
  const probs = predict(text)
  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a])
  return { probs, order, index: order[0] }
}

export function explain(text) {
  const words = text.trim().split(/\s+/).filter(Boolean)
  const base = classify(text)
  const predicted = base.index
  const tokens = words.map((word, i) => ({
    text: word,
    weight: base.probs[predicted] - predict(words.filter((_, j) => j !== i).join(' '))[predicted],
  }))
  return {
    label: LABELS[predicted],
    confidence: base.probs[predicted],
    top: base.order.slice(0, 5).map((i) => ({ label: LABELS[i], prob: base.probs[i] })),
    tokens,
    truncated: false,
  }
}

export function validateBody(body) {
  if (!body || typeof body.text !== 'string' || body.text.length < 3 || body.text.length > 2000) {
    return { detail: 'text must contain 3 to 2000 characters' }
  }
  if (body.model != null && body.model !== MODEL_NAME) {
    return { detail: `unknown model '${String(body.model)}'` }
  }
  return null
}
