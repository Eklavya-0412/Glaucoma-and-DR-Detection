import { useMemo, useState } from 'react'
import { ArrowRight, ChevronRight } from 'lucide-react'
import type { Stage } from '../api'
import { Badge, KV, ZoomImage } from './ui'

export const MODULES: { id: string; label: string; blurb: string }[] = [
  { id: 'original', label: 'Original', blurb: 'The unmodified input image. Every later step works on copies.' },
  { id: 'roi', label: 'Retinal ROI', blurb: 'Separates the circular retinal field from the dark camera surround. All metrics and evaluation are restricted to this field.' },
  { id: 'quality', label: 'Image quality', blurb: 'Brightness, contrast, entropy, sharpness and noise measured inside the retinal field.' },
  { id: 'enhancement', label: 'Enhancement', blurb: 'Histogram Equalization vs CLAHE compared side by side and quantitatively.' },
  { id: 'vessels', label: 'Blood vessels', blurb: 'Classical vessel segmentation: green channel → CLAHE → Gaussian → multi-scale black-hat → Otsu → connected components.' },
  { id: 'glaucoma', label: 'Optic disc & cup', blurb: 'Disc localisation and segmentation, independent cup segmentation, and cup-to-disc geometry.' },
  { id: 'lesions', label: 'DR lesions', blurb: 'Separate pipelines for microaneurysm, hemorrhage and hard-exudate candidates, plus spatial analysis.' },
  { id: 'texture', label: 'GLCM texture', blurb: 'Gray-level co-occurrence matrix on the quantised retinal field.' },
  { id: 'morphology', label: 'Morphology', blurb: 'Reference demonstration of each morphological operator on the CLAHE green channel.' },
  { id: 'edges', label: 'Edge detection', blurb: 'Sobel, Prewitt, Canny and Laplacian edge maps with their parameters.' },
  { id: 'evaluation', label: 'Evaluation', blurb: 'Prediction-vs-reference maps for every task that has ground truth.' },
]

const GROUP_LABELS: Record<string, string> = {
  shared: 'Shared preprocessing', microaneurysm: 'G1 · Microaneurysm candidates', hemorrhage: 'G2 · Hemorrhage candidates',
  exudate: 'G3 · Hard-exudate candidates', summary: 'Combined result & spatial analysis', disc: 'Optic-disc localisation & segmentation',
  cup: 'Optic-cup segmentation', vessels: 'Vessel segmentation', evaluation: 'Ground-truth comparison',
}

const kindTone = { mask: 'accent', overlay: 'neutral', comparison: 'warn', plot: 'neutral', reference: 'good', image: 'neutral' } as const

export function StageCard({ stage, next }: { stage: Stage; next?: Stage }) {
  const { width, height, ...stats } = stage.stats as Record<string, unknown>
  return (
    <article className="flex flex-col rounded-xl border border-line bg-surface">
      <div className="flex items-start gap-3 px-4 pt-3.5">
        <span className="num mt-0.5 shrink-0 rounded-md bg-ink px-1.5 py-0.5 font-mono text-[11px] font-medium text-white">
          {String(stage.step).padStart(2, '0')}
        </span>
        <div className="min-w-0 flex-1">
          <h4 className="text-[14px] font-semibold leading-snug">{stage.title}</h4>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <Badge tone={kindTone[stage.kind] ?? 'neutral'}>{stage.kind}</Badge>
            {width !== undefined && <span className="num font-mono text-[11px] text-ink-faint">{String(width)}×{String(height)}</span>}
          </div>
        </div>
      </div>
      <div className="px-4 pt-3">
        <ZoomImage src={stage.url} title={`Step ${stage.step} · ${stage.title}`} caption={stage.explanation} />
      </div>
      <div className="flex flex-1 flex-col gap-3 px-4 pb-4 pt-3">
        <p className="text-[13px] leading-relaxed text-ink-soft">{stage.explanation}</p>
        {Object.keys(stage.params).length > 0 && (
          <div>
            <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-ink-faint">Parameters</div>
            <KV data={stage.params} />
          </div>
        )}
        {Object.keys(stats).length > 0 && (
          <div>
            <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-ink-faint">Output statistics</div>
            <KV data={stats} />
          </div>
        )}
        {next && (
          <div className="mt-auto flex items-center gap-1.5 border-t border-line pt-2.5 text-[12px] text-ink-faint">
            Next <ChevronRight size={13} /> <span className="text-ink-soft">{next.title}</span>
          </div>
        )}
      </div>
    </article>
  )
}

function FlowStrip({ stages, onJump }: { stages: Stage[]; onJump: (s: Stage) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {stages.map((s, i) => (
        <div key={s.step} className="flex items-center gap-1.5">
          <button onClick={() => onJump(s)}
            className="rounded-md border border-line bg-surface px-2 py-1 text-[12px] text-ink-soft hover:border-accent hover:text-accent">
            <span className="num mr-1 font-mono text-ink-faint">{s.step}</span>{s.title}
          </button>
          {i < stages.length - 1 && <ArrowRight size={13} className="text-ink-faint" />}
        </div>
      ))}
    </div>
  )
}

export function PipelineView({ stages, moduleStatus }: { stages: Stage[]; moduleStatus: Record<string, string> }) {
  const available = MODULES.filter((m) => stages.some((s) => s.module === m.id))
  const [active, setActive] = useState(available[0]?.id ?? 'original')
  const mod = MODULES.find((m) => m.id === active) ?? MODULES[0]
  const modStages = useMemo(() => stages.filter((s) => s.module === active), [stages, active])
  const original = stages.find((s) => s.key === 'original')
  const groups = useMemo(() => {
    const g: Record<string, Stage[]> = {}
    for (const s of modStages) (g[s.group || ''] ??= []).push(s)
    return Object.entries(g)
  }, [modStages])
  const jump = (s: Stage) => document.getElementById(`stage-${s.step}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  return (
    <div className="grid gap-5 lg:grid-cols-[210px_1fr]">
      <nav className="lg:sticky lg:top-4 lg:self-start">
        <div className="mb-2 px-2 text-[11px] font-medium uppercase tracking-wide text-ink-faint">Modules (execution order)</div>
        <ul className="flex flex-wrap gap-1 lg:flex-col">
          {available.map((m) => {
            const n = stages.filter((s) => s.module === m.id).length
            const st = moduleStatus[m.id === 'glaucoma' ? 'glaucoma' : m.id]
            return (
              <li key={m.id}>
                <button onClick={() => setActive(m.id)}
                  className={`flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] ${active === m.id ? 'bg-ink text-white' : 'text-ink-soft hover:bg-surface'}`}>
                  <span className="flex items-center gap-1.5">
                    {st && st !== 'ok' && <span className={`h-1.5 w-1.5 rounded-full ${st.startsWith('error') ? 'bg-bad' : 'bg-warn'}`} />}
                    {m.label}
                  </span>
                  <span className={`num font-mono text-[11px] ${active === m.id ? 'text-white/60' : 'text-ink-faint'}`}>{n}</span>
                </button>
              </li>
            )
          })}
        </ul>
      </nav>

      <div className="min-w-0 space-y-5">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">{mod.label}</h2>
          <p className="mt-1 max-w-3xl text-[14px] text-ink-soft">{mod.blurb}</p>
          {moduleStatus[active] && moduleStatus[active] !== 'ok' && (
            <p className="mt-2 rounded-lg bg-warn-soft px-3 py-2 text-[13px] text-warn">{moduleStatus[active]}</p>
          )}
        </div>

        {groups.map(([group, gs]) => (
          <div key={group} className="space-y-3">
            {(group || groups.length > 1) && (
              <h3 className="text-[13px] font-semibold uppercase tracking-wide text-ink-soft">{GROUP_LABELS[group] ?? group}</h3>
            )}
            <div className="rounded-xl border border-line bg-surface/60 p-3">
              <FlowStrip stages={active === 'original' || !original ? gs : [original, ...gs]} onJump={jump} />
            </div>
            <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
              {gs.map((s, i) => (
                <div key={s.step} id={`stage-${s.step}`} className="scroll-mt-4">
                  <StageCard stage={s} next={gs[i + 1]} />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
