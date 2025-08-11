"use client"

import { useEffect, useMemo, useState } from 'react'
import ClusterPlot from '@/components/ClusterPlot'
import { API_BASE, alQuery, cluster, extractFeatures, getLabels, listFiles, saveLabels } from '@/lib/api'

export default function HomePage() {
    const [files, setFiles] = useState<string[]>([])
    const [selected, setSelected] = useState<string[]>([])
    const [embedding, setEmbedding] = useState<number[][]>([])
    const [labels, setLabels] = useState<number[]>([])
    const [metrics, setMetrics] = useState<Record<string, number>>({})
    const [recommended, setRecommended] = useState<number[]>([])
    const [selectedPoints, setSelectedPoints] = useState<number[]>([])
    const [busy, setBusy] = useState(false)
    const hasEmbedding = embedding.length > 0

    useEffect(() => {
        listFiles().then(setFiles).catch(console.error)
    }, [])

    const canRun = selected.length > 1 && !busy

    async function runClustering() {
        try {
            setBusy(true)
            const featsResp = await extractFeatures(selected)
            const cl = await cluster({ features: featsResp.features, embed_method: 'pca', algorithm: 'kmeans', n_clusters: 3 })
            setEmbedding(cl.embedding)
            setLabels(cl.labels)
            setMetrics(cl.metrics || {})
            setRecommended([])
            setSelectedPoints([])
        } catch (e) {
            console.error(e)
            alert('Clustering failed')
        } finally {
            setBusy(false)
        }
    }

    async function runActiveLearning() {
        try {
            if (!hasEmbedding) return
            setBusy(true)
            const inds = await alQuery(embedding, labels, 20)
            setRecommended(inds)
        } catch (e) {
            console.error(e)
            alert('Active Learning query failed')
        } finally {
            setBusy(false)
        }
    }

    async function assignLabel(label: string) {
        if (selectedPoints.length === 0) return
        const map: Record<string, string> = {}
        for (const idx of selectedPoints) {
            const path = selected[idx]
            if (path) map[path] = label
        }
        try {
            await saveLabels(map)
            alert(`Saved ${Object.keys(map).length} labels.`)
        } catch (e) {
            console.error(e)
            alert('Saving labels failed')
        }
    }

    return (
        <main className="p-6 mx-auto max-w-7xl space-y-6">
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold">Interactive Clustering</h1>
                    <p className="text-sm text-slate-400">Backend: {API_BASE}</p>
                </div>
                <div className="flex gap-2">
                    <button className="btn" onClick={runClustering} disabled={!canRun}>{busy ? 'Running…' : 'Run clustering'}</button>
                    <button className="btn bg-emerald-600 hover:bg-emerald-500" onClick={runActiveLearning} disabled={!hasEmbedding || busy}>Suggest samples</button>
                </div>
            </header>

            <section className="grid lg:grid-cols-3 gap-4">
                <div className="card p-4 lg:col-span-1 space-y-3">
                    <h2 className="font-medium">Dataset</h2>
                    <div className="text-sm text-slate-400">Select ATF files from server</div>
                    <div className="max-h-72 overflow-y-auto border border-slate-800 rounded-md">
                        {files.map((f) => (
                            <label key={f} className="flex items-center gap-2 px-3 py-2 hover:bg-slate-800/60">
                                <input type="checkbox" className="accent-brand-500"
                                    checked={selected.includes(f)}
                                    onChange={(e) => {
                                        setEmbedding([]); setLabels([]); setRecommended([])
                                        setSelected((prev) => e.target.checked ? [...prev, f] : prev.filter((x) => x !== f))
                                    }}
                                />
                                <span className="truncate" title={f}>{f}</span>
                            </label>
                        ))}
                    </div>
                </div>

                <div className="lg:col-span-2 space-y-3">
                    {hasEmbedding ? (
                        <ClusterPlot embedding={embedding} labels={labels} paths={selected} highlight={recommended} onSelect={setSelectedPoints} />
                    ) : (
                        <div className="card p-8 h-[520px] grid place-items-center text-slate-400">Run clustering to see plot</div>
                    )}

                    <div className="card p-4 flex items-center justify-between">
                        <div className="flex gap-2 items-center">
                            <button className="btn bg-sky-600 hover:bg-sky-500" onClick={() => assignLabel('positive')} disabled={selectedPoints.length === 0}>Label Positive</button>
                            <button className="btn bg-rose-600 hover:bg-rose-500" onClick={() => assignLabel('negative')} disabled={selectedPoints.length === 0}>Label Negative</button>
                            <button className="btn bg-slate-600 hover:bg-slate-500" onClick={() => assignLabel('unknown')} disabled={selectedPoints.length === 0}>Label Unknown</button>
                        </div>
                        <div className="text-sm text-slate-400">Selected: {selectedPoints.length} • Silhouette: {metrics.silhouette?.toFixed?.(3) ?? '—'}</div>
                    </div>
                </div>
            </section>
        </main>
    )
}
