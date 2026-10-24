import { Download, FileText, Table2 } from 'lucide-react'
import type { Json } from '../api'
import { fmt } from '../api'
import { Badge, Card, KV, Table, statusTone } from './ui'

type R = Record<string, any>

export function ReportView({ results, analysisId }: { results: Json; analysisId: string }) {
  const r = results as R
  const s = r.screening ?? {}
  const flags = s.flags ?? {}
  const g = r.glaucoma ?? {}
  const les = r.lesions?.summary ?? {}
  const files = [
    { href: `/api/analysis/${analysisId}/report.pdf`, label: 'Screening report (PDF)', icon: FileText },
    { href: `/api/analysis/${analysisId}/features.csv`, label: 'Feature table (CSV)', icon: Table2 },
    { href: `/files/${r.dataset}/${r.image_id}/processing_manifest.json`, label: 'Processing manifest (JSON)', icon: Download },
    { href: `/files/${r.dataset}/${r.image_id}/08_evaluation/evaluation_metrics.json`, label: 'Evaluation metrics (JSON)', icon: Download },
  ]
  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
        <Card title={`Report — ${String(r.dataset).toUpperCase()} / ${r.image_id}`} subtitle={`Processed in ${r.elapsed_s} s`}>
          <div className="grid gap-5 md:grid-cols-2">
            <div>
              <h4 className="mb-1 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Input & provenance</h4>
              <KV data={{ file: r.source?.file, original_shape: r.source?.original_shape, working_size: r.source?.working_size,
                notes: (r.source?.normalisation_notes ?? []).join('; ') || 'none' }} />
              <h4 className="mb-1 mt-3 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Annotations available</h4>
              <KV data={r.annotations ?? {}} />
            </div>
            <div>
              <h4 className="mb-1 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Dataset reference labels (ground truth)</h4>
              <KV data={s.reference ?? {}} />
              <h4 className="mb-1 mt-3 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Module status</h4>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(r.module_status ?? {}).map(([k, v]) => (
                  <span key={k} title={String(v)}><Badge tone={statusTone(String(v))}>{k}: {String(v).split(':')[0]}</Badge></span>
                ))}
              </div>
            </div>
          </div>
        </Card>
        <Card title="Downloads">
          <ul className="space-y-1.5">
            {files.map((f) => (
              <li key={f.href}>
                <a href={f.href} target="_blank" rel="noreferrer"
                  className="flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-[13px] hover:border-accent hover:text-accent">
                  <f.icon size={15} /> {f.label}
                </a>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <Card title="Experimental screening flags" subtitle={s.disclaimer}>
        <div className="grid gap-4 lg:grid-cols-2">
          {Object.entries(flags as R).map(([k, f]) => (
            <div key={k} className="rounded-lg border border-line p-4">
              <div className="flex items-center justify-between gap-2">
                <h4 className="text-[14px] font-semibold">{f.label}</h4>
                <Badge tone={f.flag === null ? 'neutral' : f.flag ? 'warn' : 'good'}>{f.flag === null ? 'insufficient data' : f.flag ? 'flagged' : 'not flagged'}</Badge>
              </div>
              <p className="mt-1 text-[13px] text-ink-soft">{f.result}</p>
              <Table dense rows={f.conditions} columns={[
                { key: 'feature', label: 'Feature' }, { key: 'rule', label: 'Rule', render: (c) => `${c.op} ${c.threshold}` },
                { key: 'value', label: 'Measured', align: 'right' },
                { key: 'met', label: 'Met', render: (c) => c.met === null ? '—' : c.met ? 'yes' : 'no' }]} />
              <p className="mt-2 text-[12px] text-ink-faint">Combination: {f.combine}. {f.basis}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Summary of measurements">
        <div className="grid gap-5 md:grid-cols-3">
          <div>
            <h4 className="mb-1 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Glaucoma geometry</h4>
            <KV data={{ disc: g.status?.disc, cup: g.status?.cup, VCDR: g.ratios?.vcdr, HCDR: g.ratios?.hcdr, ACDR: g.ratios?.acdr, rim_area: g.ratios?.rim_area }} />
          </div>
          <div>
            <h4 className="mb-1 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Vessels</h4>
            <KV data={{ density: r.vessels?.metrics?.vessel_density, mean_width_px: r.vessels?.metrics?.mean_vessel_width_px,
              branch_points: r.vessels?.metrics?.branch_points, F1_vs_reference: r.evaluation?.vessels?.metrics?.f1 }} />
          </div>
          <div>
            <h4 className="mb-1 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Lesion candidates</h4>
            <KV data={Object.fromEntries(Object.entries(les as R).map(([k, v]) => [k, `${fmt(v.count)} (${fmt(v.total_area_px)} px²)`]))} />
          </div>
        </div>
        <div className="mt-4 rounded-lg bg-canvas px-4 py-3 text-[12.5px] text-ink-soft">
          <b className="text-ink">How to read this report.</b> <i>Measured features</i> are direct image measurements.
          <i> Candidate detections</i> are thresholded connected components, not confirmed lesions. <i>Screening flags</i> are
          experimental rules. <i>Ground-truth labels</i> come from the dataset. Anything marked unavailable could not be measured or
          has no reference. None of this is a clinical diagnosis.
        </div>
      </Card>
    </div>
  )
}
