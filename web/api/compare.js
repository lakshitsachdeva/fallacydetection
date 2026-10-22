import { classify, LABELS, MODEL_NAME, validateBody } from '../lib/baseline.js'

export default {
  async fetch(request) {
    if (request.method !== 'POST') return Response.json({ detail: 'Method not allowed' }, { status: 405 })
    const body = await request.json().catch(() => null)
    const error = validateBody(body)
    if (error) return Response.json(error, { status: 422 })
    const result = classify(body.text)
    return Response.json({ results: [{ model: MODEL_NAME, label: LABELS[result.index], confidence: result.probs[result.index] }] })
  },
}
