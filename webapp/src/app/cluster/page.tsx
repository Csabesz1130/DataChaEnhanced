"use client"

import { useEffect, useState } from 'react'
import ClusterPlot from '@/components/ClusterPlot'
import SignalPreview from '@/components/SignalPreview'
import { API_BASE, alQuery, cluster, extractFeatures, listFiles } from '@/lib/api'

export default function ClusterPage() {
    const [files, setFiles] = useState<string[]>([])
    const [selected, setSelected] = useState<string[]>([])
    const [embedding, setEmbedding] = useState<number[][]>([])
    const [labels, setLabels] = useState<number[]>([])
    const [recommended, setRecommended] = useState<number[]>([])
    const [selectedPoints, setSelectedPoints] = useState<number[]>([])

    useEffect(() => { listFiles().then(setFiles).catch(console.error) }, [])

    async function run() {
        const feats = await extractFeatures(selected)
        const c = await cluster({ features: feats.features, embed_method: 'pca', algorithm: 'kmeans', n_clusters: 3 })
        setEmbedding(c.embedding)
        setLabels(c.labels)
        setRecommended([])
    }

    return (
        <main className="p-6 mx-auto max-w-7xl space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-xl font-semibold">Cluster</h1>
                <button className="btn" onClick={run} disabled={selected.length < 2}>Run</button>
            </div>

            <div className="grid lg:grid-cols-3 gap-4">
                <div className="card p-4 space-y-2">
                    <div className="text-sm text-slate-400">Files</div>
                    <div className="max-h-80 overflow-y-auto">
                        {files.map((f) => (
                            <label key={f} className="flex items-center gap-2 px-2 py-1 hover:bg-slate-800/40">
                                <input type="checkbox" className="accent-brand-500"
                                    checked={selected.includes(f)}
                                    onChange={(e) => setSelected((prev) => e.target.checked ? [...prev, f] : prev.filter((x) => x !== f))}
                                />
                                <span className="truncate" title={f}>{f}</span>
                            </label>
                        ))}
                    </div>
                </div>

                <div className="lg:col-span-2 space-y-3">
                    {embedding.length ? (
                        <ClusterPlot embedding={embedding} labels={labels} paths={selected} highlight={recommended} onSelect={setSelectedPoints} />
                    ) : (
                        <div className="card p-8 h-[520px] grid place-items-center text-slate-400">Run clustering to see plot</div>
                    )}
                    <SignalPreview path={selected[selectedPoints[0]]} />
                </div>
            </div>
        </main>
    )
}
