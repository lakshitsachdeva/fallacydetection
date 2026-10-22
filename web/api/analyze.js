import { explain, MODEL_NAME, validateBody } from '../lib/baseline.js'

export default {
  async fetch(request) {
    if (request.method !== 'POST') return Response.json({ detail: 'Method not allowed' }, { status: 405 })
    const body = await request.json().catch(() => null)
    const error = validateBody(body)
    if (error) return Response.json(error, { status: 422 })
    return Response.json({ model: MODEL_NAME, ...explain(body.text) })
  },
}
