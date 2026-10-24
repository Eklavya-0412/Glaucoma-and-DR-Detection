// Typed client for the FastAPI backend (proxied at /api and /files by Vite).

export type Json = Record<string, unknown>

export interface DatasetInfo {
  id: string
  name: string
  description: string
  source_url: string
  access: string
  available: boolean
  status: string
  image_count: number
  annotation_counts: Record<string, number>
}

export interface ImageItem {
  image_id: string
  file: string
  annotations: string[]
  labels: Record<string, unknown>
  analyzed: boolean
}

export interface Stage {
  step: number
  module: string
  key: string
  title: string
  technique: string
  explanation: string
  file: string
  url: string
  params: Record<string, unknown>
  stats: Record<string, unknown>
  kind: 'image' | 'mask' | 'overlay' | 'comparison' | 'plot' | 'reference'
  group: string
}

export interface AnalysisStatus {
  analysis_id: string
  dataset: string
  image_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  stage?: string | null
  error?: string
  elapsed_s?: number
  module_status?: Record<string, string>
}

export interface Feature {
  key: string
  feature: string
  group: string
  value: number | null
  unit: string
  status: 'available' | 'unavailable' | 'failed'
  note: string
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* not JSON */ }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => req<Json>('/api/health'),
  datasets: () => req<DatasetInfo[]>('/api/datasets'),
  images: (ds: string) => req<ImageItem[]>(`/api/datasets/${ds}/images`),
  metadata: (ds: string, id: string) => req<Json>(`/api/images/${ds}/${id}/metadata`),
  previewUrl: (ds: string, id: string) => `/api/images/${ds}/${id}/preview`,
  analyze: (dataset: string, image_id: string) =>
    req<AnalysisStatus>('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, image_id }),
    }),
  status: (aid: string) => req<AnalysisStatus>(`/api/analysis/${aid}`),
  artifacts: (aid: string) => req<{ stages: Stage[] }>(`/api/analysis/${aid}/artifacts`),
  results: (aid: string) => req<Json>(`/api/analysis/${aid}/results`),
  datasetEvaluation: () => req<Json>('/api/evaluation/summary'),
  upload: async (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<{ dataset: string; image_id: string; original_filename: string }>('/api/upload', {
      method: 'POST',
      body: fd,
    })
  },
}

export const analysisId = (ds: string, id: string) => `${ds}--${id}`

export function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return v.toLocaleString()
    return Math.abs(v) >= 1000 ? v.toFixed(1) : Number(v.toPrecision(digits)).toString()
  }
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (Array.isArray(v)) return v.map((x) => fmt(x, digits)).join(', ')
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}
