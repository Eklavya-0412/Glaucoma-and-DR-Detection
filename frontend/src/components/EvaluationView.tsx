import type { Json } from '../api'
import { fmt } from '../api'
import { Badge, Card, Empty, KV, Table } from './ui'

type R = Record<string, any>

const METRIC_COLS = ['sensitivity', 'specificity', 'precision', 'f1', 'iou', 'roc_auc']

function RocCurve({ fpr, tpr, auc }: { fpr: number[]; tpr: number[]; auc?: number }) {
  const W = 240, P = 28
  const pts = fpr.map((x, i) => `${P + x * (W - P - 8)},${W - P - tpr[i] * (W - P - 8)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${W}`} className="w-full max-w-[260px]" role="img" aria-label="ROC curve">
      <rect x={P} y={8} width={W - P - 8} height={W - P - 8} fill="none" stroke="var(--color-line)" />
      <line x1={P} y1={W - P} x2={W - 8} y2={8} stroke="var(--color-ink-faint)" strokeDasharray="3 3" />
      <polyline points={pts} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
      <text x={W / 2 + 10} y={W - 8} fontSize={10} textAnchor="middle" fill="var(--color-ink-soft)">False positive rate</text>
      <text x={10} y={W / 2} fontSize={10} textAnchor="middle" fill="var(--color-ink-soft)" transform={`rotate(-90 10 ${W / 2})`}>True positive rate</text>
      {auc !== undefined && <text x={W - 14} y={W - P - 10} fontSize={11} textAnchor="end" fill="var(--color-ink)">AUC {auc}</text>}
    </svg>
  )
}

export function EvaluationView({ results }: { results: Json }) {
  const ev = (results as R).evaluation ?? {}
  const v = ev.vessels ?? {}
  const g = ev.glaucoma ?? {}
  const l = ev.lesions ?? {}
  const vesselRows = v.metrics ? [
    { comparison: 'Our prediction vs observer AH (reference)', ...v.metrics },
    v.second_observer_vs_prediction && { comparison: 'Our prediction vs observer VK', ...v.second_observer_vs_prediction },
    v.published_baseline_hoover2000_vs_ah && { comparison: 'Hoover et al. 2000 published output vs AH', ...v.published_baseline_hoover2000_vs_ah },
    v.inter_observer_vk_vs_ah && { comparison: 'Observer VK vs AH (human agreement)', ...v.inter_observer_vk_vs_ah },
  ].filter(Boolean) : []

  return (
    <div className="space-y-5">
      <Card title="Blood-vessel segmentation" subtitle={v.reference ? `${v.reference} · ${v.protocol}` : undefined}>
        {v.metrics ? (
          <div className="space-y-5">
            <div className="grid gap-5 lg:grid-cols-[1fr_260px]">
              <div className="space-y-4">
                <Table rows={vesselRows} columns={[{ key: 'comparison', label: 'Comparison' },
                  ...METRIC_COLS.map((k) => ({ key: k, label: k === 'f1' ? 'F1 / Dice' : k.replace('_', '-'), align: 'right' as const }))]} />
                <div className="grid gap-3 sm:grid-cols-4">
                  {['TP', 'FP', 'FN', 'TN'].map((k) => (
                    <div key={k} className="rounded-lg border border-line px-3 py-2">
                      <div className="text-[11px] uppercase tracking-wide text-ink-faint">{k}</div>
                      <div className="num font-mono text-[15px]">{fmt(v.metrics[k])}</div>
                    </div>
                  ))}
                </div>
                <p className="text-[12px] text-ink-soft">Evaluated on {fmt(v.metrics.evaluated_pixels)} pixels inside the valid retinal field; background outside the retina is not counted as true negatives.</p>
              </div>
              {v.roc?.fpr?.length > 0 && <RocCurve fpr={v.roc.fpr} tpr={v.roc.tpr} auc={v.metrics.roc_auc} />}
            </div>
            {v.enhancement_effect && (
              <div>
                <h4 className="mb-1.5 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Effect of enhancement on downstream segmentation (same reference, same threshold rule)</h4>
                <Table dense rows={v.enhancement_effect} columns={[{ key: 'enhancement', label: 'Enhancement before black-hat' },
                  ...['sensitivity', 'specificity', 'precision', 'f1', 'roc_auc'].map((k) => ({ key: k, label: k, align: 'right' as const }))]} />
              </div>
            )}
          </div>
        ) : <Empty>{v.status ?? 'No vessel reference mask for this image (STARE provides masks for 20 images).'}</Empty>}
      </Card>

      <Card title="Optic disc & cup">
        {Object.keys(g).length === 0 || g.status ? <Empty>{g.status ?? 'No reference available.'}</Empty> : (
          <div className="grid gap-5 lg:grid-cols-2">
            {g.localization && (
              <div>
                <h4 className="mb-1.5 flex items-center gap-2 text-[13px] font-semibold">
                  Localisation {g.localization.success !== undefined && g.localization.success !== null &&
                    <Badge tone={g.localization.success ? 'good' : 'bad'}>{g.localization.success ? 'success' : 'failure'}</Badge>}
                </h4>
                <KV data={g.localization} />
                <p className="mt-2 text-[12px] text-ink-soft">STARE provides only nerve-centre coordinates, so localisation is evaluated separately — no pixel-level disc mask exists for STARE.</p>
              </div>
            )}
            {['disc_segmentation', 'cup_segmentation'].filter((k) => g[k]).map((k) => (
              <div key={k}><h4 className="mb-1.5 text-[13px] font-semibold">{k.replace('_', ' ')}</h4><KV data={g[k]} /></div>
            ))}
            {g.vcdr && <div><h4 className="mb-1.5 text-[13px] font-semibold">VCDR error</h4><KV data={g.vcdr} /></div>}
          </div>
        )}
      </Card>

      <Card title="DR lesion candidates">
        {l.status ? <Empty>{l.status}</Empty> : (
          <div className="space-y-5">
            {l.stare_image_level && (
              <div>
                <p className="mb-2 text-[13px] text-ink-soft">{l.stare_image_level.note}</p>
                <Table rows={['microaneurysm', 'hemorrhage', 'exudate'].map((k) => {
                  const x = l.stare_image_level[k]
                  return {
                    category: k,
                    grades: Object.values(x.reference_grades as R).map((gr: any) => `${gr.name}: ${gr.meaning}`).join(' · '),
                    present: x.reference_present, count: x.predicted_candidate_count,
                  }
                })} columns={[{ key: 'category', label: 'Category' }, { key: 'grades', label: 'STARE reference grade(s)' },
                  { key: 'present', label: 'Reference present', render: (x) => x.present === null ? <Badge>unknown</Badge> : <Badge tone={x.present ? 'warn' : 'neutral'}>{x.present ? 'present' : 'absent'}</Badge> },
                  { key: 'count', label: 'Predicted candidates', align: 'right' }]} />
              </div>
            )}
            {['microaneurysm', 'hemorrhage', 'exudate'].filter((k) => l[k]).map((k) => (
              <div key={k} className="grid gap-4 md:grid-cols-2">
                <div><h4 className="mb-1 text-[13px] font-semibold">{k} — pixel level</h4><KV data={l[k].pixel} /></div>
                <div><h4 className="mb-1 text-[13px] font-semibold">{k} — component level</h4><KV data={l[k].component} /></div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
