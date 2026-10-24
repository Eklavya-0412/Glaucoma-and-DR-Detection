import { useMemo, useState } from 'react'
import type { Stage } from '../api'
import { Card, Empty, ZoomImage } from './ui'

const PRESETS: { id: string; label: string; keys: [string, string][] }[] = [
  { id: 'enh', label: 'Original vs HE vs CLAHE', keys: [['original', 'original'], ['enhancement', 'histogram_equalization'], ['enhancement', 'clahe']] },
  { id: 'roi', label: 'Original vs retinal ROI', keys: [['original', 'original'], ['roi', 'retinal_mask'], ['roi', 'roi_overlay']] },
  { id: 'ves', label: 'Original vs vessel mask', keys: [['original', 'original'], ['vessels', 'vessel_mask'], ['vessels', 'vessel_overlay']] },
  { id: 'disc', label: 'Original vs optic disc / cup', keys: [['original', 'original'], ['glaucoma', 'disc_candidate'], ['glaucoma', 'disc_cup_overlay']] },
  { id: 'les', label: 'Original vs lesion masks', keys: [['original', 'original'], ['lesions', 'microaneurysm_mask'], ['lesions', 'hemorrhage_mask'], ['lesions', 'exudate_mask'], ['lesions', 'lesion_overlay']] },
  { id: 'gt', label: 'Prediction vs ground truth', keys: [['vessels', 'vessel_mask'], ['vessels', 'ground_truth'], ['vessels', 'prediction_vs_gt'], ['glaucoma', 'disc_vs_gt'], ['glaucoma', 'cup_vs_gt'], ['glaucoma', 'reference_disc_cup'], ['lesions', 'microaneurysm_vs_gt'], ['lesions', 'hemorrhage_vs_gt'], ['lesions', 'exudate_vs_gt']] },
]

function Panel({ s }: { s: Stage }) {
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1.5 flex items-baseline gap-2">
        <span className="num font-mono text-[11px] text-ink-faint">#{s.step}</span>
        <span className="text-[13px] font-semibold">{s.title}</span>
      </figcaption>
      <ZoomImage src={s.url} title={s.title} caption={s.explanation} light={s.kind === 'plot' || s.kind === 'comparison'} />
    </figure>
  )
}

export function CompareView({ stages }: { stages: Stage[] }) {
  const find = (m: string, k: string) => stages.find((s) => s.module === m && s.key === k)
  const presets = PRESETS.map((p) => ({ ...p, stages: p.keys.map(([m, k]) => find(m, k)).filter(Boolean) as Stage[] }))
    .filter((p) => p.stages.length >= 2 || (p.id === 'gt' && p.stages.length > 0))
  const [active, setActive] = useState(presets[0]?.id)
  const cur = presets.find((p) => p.id === active) ?? presets[0]
  const options = useMemo(() => stages.map((s) => ({ v: String(s.step), label: `${s.step}. ${s.title}` })), [stages])
  const [a, setA] = useState(String(stages[0]?.step ?? ''))
  const [b, setB] = useState(String(stages.find((s) => s.key === 'vessel_mask')?.step ?? stages[1]?.step ?? ''))
  const sa = stages.find((s) => String(s.step) === a)
  const sb = stages.find((s) => String(s.step) === b)

  return (
    <div className="space-y-5">
      <Card title="Side-by-side comparisons" subtitle="Each panel is a real backend output; click any image to enlarge.">
        <div className="mb-4 flex flex-wrap gap-1.5">
          {presets.map((p) => (
            <button key={p.id} onClick={() => setActive(p.id)}
              className={`rounded-lg border px-3 py-1.5 text-[13px] ${cur?.id === p.id ? 'border-ink bg-ink text-white' : 'border-line text-ink-soft hover:bg-canvas'}`}>
              {p.label}
            </button>
          ))}
        </div>
        {cur ? (
          <div className={`grid gap-4 ${cur.stages.length >= 3 ? 'md:grid-cols-3' : 'md:grid-cols-2'}`}>
            {cur.stages.map((s) => <Panel key={s.step} s={s} />)}
          </div>
        ) : <Empty>No comparable outputs.</Empty>}
        {cur?.id === 'gt' && (
          <p className="mt-3 text-[12.5px] text-ink-soft">
            Agreement maps: <b className="text-ink">white</b> true positive, <b className="text-bad">red</b> false positive,
            <b className="text-good"> green</b> false negative, grey outside the evaluated retinal field.
          </p>
        )}
        {!presets.some((p) => p.id === 'gt') && (
          <p className="mt-3 text-[12.5px] text-ink-soft">No pixel-level ground truth exists for this image, so no prediction-vs-reference comparison is shown.</p>
        )}
      </Card>

      <Card title="Custom A / B comparison" subtitle="Pick any two processing stages.">
        <div className="mb-4 grid gap-3 md:grid-cols-2">
          {[['A', a, setA], ['B', b, setB]].map(([lab, val, set]) => (
            <label key={lab as string} className="text-[12px] text-ink-faint">
              Stage {lab as string}
              <select value={val as string} onChange={(e) => (set as (v: string) => void)(e.target.value)}
                className="mt-1 w-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-[13px] text-ink">
                {options.map((o) => <option key={o.v} value={o.v}>{o.label}</option>)}
              </select>
            </label>
          ))}
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {sa && <Panel s={sa} />}
          {sb && <Panel s={sb} />}
        </div>
      </Card>
    </div>
  )
}
