import type { Feature, Json } from '../api'
import { fmt } from '../api'
import { Badge, Card, KV, Table, statusTone } from './ui'

type R = Record<string, any>

function Bars({ rows, metric, label }: { rows: R[]; metric: string; label: string }) {
  const max = Math.max(...rows.map((r) => Number(r[metric]) || 0), 1e-9)
  return (
    <div>
      <div className="mb-1.5 text-[12px] font-medium text-ink-soft">{label}</div>
      <div className="space-y-1">
        {rows.map((r) => (
          <div key={r.version} className="grid grid-cols-[150px_1fr_64px] items-center gap-2 text-[12px]">
            <span className="truncate text-ink-soft">{r.version}</span>
            <div className="h-2.5 rounded bg-canvas">
              <div className="h-2.5 rounded bg-accent" style={{ width: `${(Number(r[metric]) / max) * 100}%` }} />
            </div>
            <span className="num text-right font-mono">{fmt(r[metric])}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function FeaturesView({ results }: { results: Json }) {
  const r = results as R
  const features: Feature[] = r.features ?? []
  const groups = [...new Set(features.map((f) => f.group))]
  const q = r.quality?.roi ?? {}
  const defs = r.quality?.definitions ?? {}
  const enh: R[] = r.enhancement?.metrics ?? []
  const g = r.glaucoma ?? {}
  const les = r.lesions?.summary ?? {}
  const sp = r.lesions?.spatial ?? {}
  const tx = r.texture

  return (
    <div className="space-y-5">
      <Card title="Core feature set (32 features)"
        subtitle="Missing measurements are shown as unavailable with the reason — never replaced by zero. DR lesion values are algorithmic candidates, not confirmed lesions.">
        <div className="grid gap-6 xl:grid-cols-2">
          {groups.map((grp) => (
            <div key={grp}>
              <h4 className="mb-1.5 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">{grp}</h4>
              <Table dense rows={features.filter((f) => f.group === grp) as unknown as R[]} columns={[
                { key: 'feature', label: 'Feature' },
                { key: 'value', label: 'Value', align: 'right' },
                { key: 'unit', label: 'Unit', render: (x) => <span className="text-[12px] text-ink-faint">{x.unit}</span> },
                { key: 'status', label: 'Status', render: (x) => <span title={x.note}><Badge tone={statusTone(x.status)}>{x.status}</Badge></span> },
              ]} />
            </div>
          ))}
        </div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Image quality (retinal field)" subtitle={r.quality?.region}>
          <Table dense rows={Object.entries(q).map(([k, v]) => ({ k, v, d: defs[k] }))} columns={[
            { key: 'k', label: 'Metric', render: (x) => <span className={r.quality?.core?.includes(x.k) ? 'font-semibold' : ''}>{String(x.k).replace(/_/g, ' ')}</span> },
            { key: 'v', label: 'Value', align: 'right' },
            { key: 'd', label: 'How it is computed', render: (x) => <span className="text-[12px] text-ink-soft">{x.d}</span> },
          ]} />
          <p className="mt-2 text-[12px] text-ink-faint">Bold = core metrics. These are image-quality measurements, not clinical findings.</p>
        </Card>

        <Card title="Enhancement comparison" subtitle="Quality metrics of the original, HE and CLAHE versions (no winner is assumed).">
          {enh.length ? (
            <div className="space-y-4">
              <Table dense rows={enh} columns={Object.keys(enh[0]).map((k) => ({ key: k, label: k.replace(/_/g, ' '), align: k === 'version' ? 'left' : 'right' }))} />
              <div className="grid gap-4 md:grid-cols-2">
                <Bars rows={enh} metric="rms_contrast" label="RMS contrast" />
                <Bars rows={enh} metric="entropy_bits" label="Entropy (bits)" />
                <Bars rows={enh} metric="edge_density" label="Edge density" />
                <Bars rows={enh} metric="noise_sigma" label="Noise σ estimate" />
              </div>
            </div>
          ) : <span className="text-[13px] text-ink-soft">Not available.</span>}
        </Card>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Optic disc & cup geometry" subtitle={`Disc: ${g.status?.disc ?? '—'} · Cup: ${g.status?.cup ?? '—'}`}>
          <div className="grid gap-5 md:grid-cols-3">
            <div><div className="mb-1 text-[12px] font-semibold">Disc</div><KV data={g.disc ?? {}} exclude={['bbox']} /></div>
            <div><div className="mb-1 text-[12px] font-semibold">Cup</div><KV data={g.cup ?? {}} exclude={['bbox']} /></div>
            <div><div className="mb-1 text-[12px] font-semibold">Ratios & rim</div><KV data={g.ratios ?? {}} /></div>
          </div>
          <p className="mt-3 text-[12px] text-ink-soft">VCDR = cup height / disc height · HCDR = cup width / disc width · ACDR = cup area / disc area · Rim = disc − cup area. {g.limitations}</p>
        </Card>
        <Card title="Vessel measurements">
          <KV data={r.vessels?.metrics ?? {}} />
        </Card>
      </div>

      <Card title="DR lesion candidates" subtitle={r.lesions?.note}>
        <Table dense rows={Object.entries(les).map(([k, v]) => ({ category: k, ...(v as R) }))} columns={[
          { key: 'category', label: 'Category' },
          ...['count', 'total_area_px', 'mean_area_px', 'max_area_px', 'density', 'mean_perimeter_px', 'mean_equivalent_diameter_px',
            'mean_circularity', 'mean_aspect_ratio', 'mean_eccentricity', 'mean_solidity', 'mean_brightness_L', 'mean_distance_to_disc_px']
            .map((k) => ({ key: k, label: k.replace(/_px$/, ' (px)').replace(/_/g, ' '), align: 'right' as const })),
        ]} />
        {sp.quadrants && (
          <div className="mt-5 grid gap-5 lg:grid-cols-2">
            <div>
              <h4 className="mb-1.5 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Per-quadrant distribution</h4>
              <Table dense rows={Object.entries(sp.quadrants as R).map(([qn, v]) => ({
                quadrant: qn.replace('_', ' '), ma: v.microaneurysm.count, he: v.hemorrhage.count, ex: v.exudate.count,
                total: v.total_count, density: v.total_density,
              }))} columns={[
                { key: 'quadrant', label: 'Quadrant' }, { key: 'ma', label: 'MA', align: 'right' }, { key: 'he', label: 'HE', align: 'right' },
                { key: 'ex', label: 'EX', align: 'right' }, { key: 'total', label: 'Total', align: 'right' }, { key: 'density', label: 'Area density', align: 'right' },
              ]} />
            </div>
            <div className="space-y-3">
              <div>
                <h4 className="mb-1.5 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Lesion density</h4>
                <KV data={sp.density ?? {}} />
                <p className="mt-1 text-[12px] text-ink-faint">{sp.formula}</p>
              </div>
              <div>
                <h4 className="mb-1.5 text-[12px] font-semibold uppercase tracking-wide text-ink-soft">Distance to optic-disc centre</h4>
                <KV data={sp.distance_to_disc ?? {}} />
              </div>
            </div>
          </div>
        )}
      </Card>

      {tx && (
        <Card title="GLCM texture features" subtitle="Texture descriptors of the image; they do not by themselves indicate disease.">
          <div className="grid gap-5 lg:grid-cols-2">
            <Table dense rows={Object.entries(tx.by_distance as R).map(([d, v]) => ({ distance: `d = ${d} px`, ...(v as R) }))}
              columns={[{ key: 'distance', label: 'Distance' }, ...Object.keys(tx.features).map((k) => ({ key: k, label: k, align: 'right' as const }))]} />
            <div>
              <div className="mb-1 text-[12px] font-semibold">Aggregated (mean of all distances/angles)</div>
              <KV data={tx.features} />
              <div className="mb-1 mt-3 text-[12px] font-semibold">Protocol</div>
              <KV data={tx.protocol} />
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
