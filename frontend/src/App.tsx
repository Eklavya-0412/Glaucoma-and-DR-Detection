import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Eye, Loader2, Play, Upload, AlertTriangle, CheckCircle2, Search } from 'lucide-react'
import { api, analysisId, fmt, type AnalysisStatus, type DatasetInfo, type ImageItem, type Json, type Stage } from './api'
import { Badge, Card, Empty, KV, LightboxProvider, ZoomImage, statusTone } from './components/ui'
import { PipelineView } from './components/PipelineView'
import { CompareView } from './components/CompareView'
import { FeaturesView } from './components/FeaturesView'
import { EvaluationView } from './components/EvaluationView'
import { ReportView } from './components/ReportView'
import { DatasetEvalView } from './components/DatasetEvalView'

const TABS = [
  { id: 'pipeline', label: 'Processing pipeline' },
  { id: 'compare', label: 'Comparison' },
  { id: 'features', label: 'Features' },
  { id: 'evaluation', label: 'Evaluation' },
  { id: 'report', label: 'Report' },
  { id: 'dataset', label: 'Dataset evaluation' },
] as const
type Tab = (typeof TABS)[number]['id']

const ANN_LABEL: Record<string, string> = {
  vessel_ah: 'vessel mask', vessel_vk: 'vessel mask (2nd obs.)', vessel_hoover_result: 'published baseline',
  optic_nerve_center: 'nerve centre', manifestation_codes: 'lesion grades', disc_cup_mask: 'disc/cup mask',
  mask_ma: 'MA mask', mask_he: 'HE mask', mask_ex: 'EX mask', mask_se: 'SE mask', mask_od: 'OD mask',
}

export default function App() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [dataset, setDataset] = useState('stare')
  const [images, setImages] = useState<ImageItem[]>([])
  const [imageId, setImageId] = useState<string>('')
  const [filter, setFilter] = useState('')
  const [onlyAnnotated, setOnlyAnnotated] = useState(true)
  const [meta, setMeta] = useState<Json | null>(null)
  const [job, setJob] = useState<AnalysisStatus | null>(null)
  const [stages, setStages] = useState<Stage[]>([])
  const [results, setResults] = useState<Json | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('pipeline')
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api.health().then(() => setBackendOk(true)).catch(() => setBackendOk(false))
    api.datasets().then(setDatasets).catch((e) => setError(String(e.message)))
  }, [])

  const loadImages = useCallback((ds: string, select?: string) => {
    api.images(ds).then((list) => {
      setImages(list)
      setImageId(select ?? (list.find((i) => i.image_id === 'im0077')?.image_id ?? list[0]?.image_id ?? ''))
    }).catch((e) => setError(String(e.message)))
  }, [])
  useEffect(() => { loadImages(dataset) }, [dataset, loadImages])

  const loadResults = useCallback(async (aid: string) => {
    const [a, r] = await Promise.all([api.artifacts(aid), api.results(aid)])
    setStages(a.stages)
    setResults(r)
  }, [])

  // when the image changes, show existing results (if analysed before)
  useEffect(() => {
    setStages([]); setResults(null); setJob(null); setMeta(null); setError(null)
    if (!imageId) return
    api.metadata(dataset, imageId).then(setMeta).catch(() => {})
    const aid = analysisId(dataset, imageId)
    // only ask for results the backend says exist (avoids a 404 for never-analysed images)
    if (images.find((i) => i.image_id === imageId)?.analyzed) {
      api.status(aid).then((s) => { if (s.status === 'completed') { setJob(s); loadResults(aid) } }).catch(() => {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, imageId, loadResults])

  // poll a running job
  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return
    const t = setInterval(async () => {
      try {
        const s = await api.status(job.analysis_id)
        setJob(s)
        if (s.status === 'completed') {
          await loadResults(s.analysis_id)
          setImages((xs) => xs.map((x) => x.image_id === s.image_id ? { ...x, analyzed: true } : x))
        }
        if (s.status === 'failed') setError(s.error ?? 'Analysis failed')
      } catch (e) { setError(String((e as Error).message)) }
    }, 900)
    return () => clearInterval(t)
  }, [job, loadResults])

  const run = async () => {
    setError(null); setStages([]); setResults(null)
    try { setJob(await api.analyze(dataset, imageId)) } catch (e) { setError(String((e as Error).message)) }
  }

  const onUpload = async (f: File | undefined) => {
    if (!f) return
    setError(null)
    try {
      const u = await api.upload(f)
      setDataset('upload')
      loadImages('upload', u.image_id)
      setDatasets(await api.datasets())
    } catch (e) { setError(`Upload rejected — ${(e as Error).message}`) }
    if (fileRef.current) fileRef.current.value = ''
  }

  const visible = useMemo(() => images.filter((i) =>
    i.image_id.toLowerCase().includes(filter.toLowerCase()) &&
    (!onlyAnnotated || dataset !== 'stare' || i.annotations.includes('vessel_ah'))), [images, filter, onlyAnnotated, dataset])
  const current = images.find((i) => i.image_id === imageId)
  const running = job?.status === 'queued' || job?.status === 'running'
  const aid = analysisId(dataset, imageId)
  const q = (results as any)?.quality?.roi

  return (
    <LightboxProvider>
      <div className="min-h-full">
        <header className="border-b border-line bg-surface">
          <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-6 py-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-accent-soft p-2 text-accent"><Eye size={20} /></div>
              <div>
                <h1 className="text-[17px] font-semibold tracking-tight">Fundus Image Analysis &amp; Retinal Screening</h1>
                <p className="text-[12.5px] text-ink-soft">Classical image processing for glaucoma (disc/cup, CDR) and diabetic-retinopathy (vessels, lesions) analysis · IVP project</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[12px]">
              {backendOk === null ? <Badge>checking backend…</Badge> : backendOk
                ? <Badge tone="good">backend connected</Badge> : <Badge tone="bad">backend offline — start uvicorn on :8000</Badge>}
            </div>
          </div>
        </header>

        <div className="mx-auto grid max-w-[1600px] gap-6 px-6 py-6 lg:grid-cols-[300px_1fr]">
          {/* ------------------------------------------------ sidebar */}
          <aside className="space-y-4 lg:sticky lg:top-4 lg:self-start">
            <Card title="Input">
              <label className="block text-[12px] text-ink-faint">Dataset
                <select value={dataset} onChange={(e) => setDataset(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-[13px] text-ink">
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id} disabled={!d.available && d.id !== 'upload'}>
                      {d.name} {d.available ? `(${d.image_count})` : '— not downloaded'}
                    </option>
                  ))}
                </select>
              </label>
              {datasets.filter((d) => !d.available && d.id !== 'upload').map((d) => (
                <p key={d.id} className="mt-1.5 text-[11.5px] text-ink-faint">{d.name}: {d.status}</p>
              ))}

              <div className="mt-3 flex items-center gap-2">
                <div className="relative flex-1">
                  <Search size={14} className="absolute left-2.5 top-2 text-ink-faint" />
                  <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter images"
                    className="w-full rounded-lg border border-line py-1.5 pl-8 pr-2 text-[13px]" />
                </div>
              </div>
              {dataset === 'stare' && (
                <label className="mt-2 flex items-center gap-2 text-[12px] text-ink-soft">
                  <input type="checkbox" checked={onlyAnnotated} onChange={(e) => setOnlyAnnotated(e.target.checked)} />
                  Only images with vessel ground truth (20)
                </label>
              )}
              <div className="mt-2 max-h-[260px] overflow-y-auto rounded-lg border border-line">
                {visible.length === 0 && <div className="p-3 text-[12px] text-ink-faint">No images.</div>}
                {visible.map((i) => (
                  <button key={i.image_id} onClick={() => setImageId(i.image_id)}
                    className={`flex w-full items-center justify-between gap-2 border-b border-line/70 px-2.5 py-1.5 text-left text-[12.5px] last:border-0 ${i.image_id === imageId ? 'bg-ink text-white' : 'hover:bg-canvas'}`}>
                    <span className="font-mono">{i.image_id}</span>
                    <span className={`truncate text-[11px] ${i.image_id === imageId ? 'text-white/70' : 'text-ink-faint'}`}>
                      {i.analyzed ? '● ' : ''}{String(i.labels.diagnosis_text ?? i.labels.original_filename ?? '').slice(0, 26)}
                    </span>
                  </button>
                ))}
              </div>

              <button onClick={() => fileRef.current?.click()}
                className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-line px-3 py-2 text-[13px] text-ink-soft hover:border-accent hover:text-accent">
                <Upload size={15} /> Upload fundus image
              </button>
              <input ref={fileRef} type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.ppm" className="hidden"
                onChange={(e) => onUpload(e.target.files?.[0])} />
              <p className="mt-1 text-[11px] text-ink-faint">PNG, JPG, TIFF, BMP, PPM · max 25 MB · uploads have no ground truth.</p>

              <button onClick={run} disabled={!imageId || running || backendOk === false}
                className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2.5 text-[14px] font-medium text-white hover:bg-accent/90 disabled:opacity-50">
                {running ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                {running ? `Running… ${job?.stage ?? ''}` : results ? 'Re-run analysis' : 'Run analysis'}
              </button>
              {job?.status === 'completed' && (
                <p className="mt-2 flex items-center gap-1.5 text-[12px] text-good"><CheckCircle2 size={14} /> Completed in {fmt(job.elapsed_s)} s</p>
              )}
              {error && <p className="mt-2 flex items-start gap-1.5 rounded-lg bg-bad-soft px-2.5 py-2 text-[12px] text-bad"><AlertTriangle size={14} className="mt-0.5 shrink-0" />{error}</p>}
            </Card>

            {imageId && (
              <Card title="Original image" subtitle={current ? `${current.file}${meta ? ` · ${meta.width}×${meta.height} ${String(meta.format).toUpperCase()}` : ''}` : undefined}>
                <ZoomImage src={api.previewUrl(dataset, imageId)} title={`Original — ${imageId}`} />
                <div className="mt-3 flex flex-wrap gap-1">
                  {(current?.annotations ?? []).map((a) => <Badge key={a} tone="accent">{ANN_LABEL[a] ?? a}</Badge>)}
                  {current && current.annotations.length === 0 && <Badge>no annotations</Badge>}
                </div>
                {meta && (meta as any).labels?.diagnosis_text && (
                  <p className="mt-2 text-[12px] text-ink-soft"><span className="text-ink-faint">Dataset diagnosis: </span>{String((meta as any).labels.diagnosis_text)}</p>
                )}
              </Card>
            )}

            {q && (
              <Card title="Image quality" subtitle="Inside the retinal field">
                <KV data={{ mean_intensity: q.mean_intensity, std_intensity: q.std_intensity, contrast_rms: q.rms_contrast,
                  entropy_bits: q.entropy_bits, edge_density: q.edge_density, sharpness: q.sharpness_laplacian_var, noise_sigma: q.noise_sigma }} />
              </Card>
            )}
          </aside>

          {/* ------------------------------------------------ main */}
          <main className="min-w-0">
            <div className="mb-5 flex flex-wrap gap-1 border-b border-line">
              {TABS.map((t) => (
                <button key={t.id} onClick={() => setTab(t.id)}
                  className={`-mb-px border-b-2 px-3.5 py-2 text-[13.5px] ${tab === t.id ? 'border-accent font-medium text-ink' : 'border-transparent text-ink-soft hover:text-ink'}`}>
                  {t.label}
                </button>
              ))}
            </div>

            {tab === 'dataset' ? <DatasetEvalView /> : !results ? (
              <Empty>
                {running ? <span className="inline-flex items-center gap-2"><Loader2 size={15} className="animate-spin" /> Processing {imageId} — stage: {job?.stage ?? 'queued'}</span>
                  : <>Select an image and press <b>Run analysis</b>. Every stage shown afterwards is generated by the backend for that image.</>}
              </Empty>
            ) : (
              <>
                {Object.entries((results as any).module_status ?? {}).some(([, v]) => v !== 'ok') && (
                  <div className="mb-4 flex flex-wrap gap-1.5">
                    {Object.entries((results as any).module_status).map(([k, v]) => (
                      <span key={k} title={String(v)}><Badge tone={statusTone(String(v))}>{k}: {String(v).length > 60 ? String(v).slice(0, 60) + '…' : String(v)}</Badge></span>
                    ))}
                  </div>
                )}
                {tab === 'pipeline' && <PipelineView stages={stages} moduleStatus={(results as any).module_status ?? {}} />}
                {tab === 'compare' && <CompareView stages={stages} />}
                {tab === 'features' && <FeaturesView results={results} />}
                {tab === 'evaluation' && <EvaluationView results={results} />}
                {tab === 'report' && <ReportView results={results} analysisId={aid} />}
              </>
            )}
          </main>
        </div>
      </div>
    </LightboxProvider>
  )
}
