"use client"

import dynamic from 'next/dynamic'
import { useEffect, useState } from 'react'
import { fetchSignal } from '@/lib/api'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

export default function SignalPreview({ path }: { path?: string }) {
    const [time, setTime] = useState<number[]>([])
    const [current, setCurrent] = useState<number[]>([])

    useEffect(() => {
        let mounted = true
        async function run() {
            if (!path) return
            try {
                const s = await fetchSignal(path)
                if (!mounted) return
                setTime(s.time_s)
                setCurrent(s.current_pA)
            } catch (e) {
                console.error(e)
            }
        }
        run()
        return () => {
            mounted = false
        }
    }, [path])

    if (!path) return <div className="card p-6 text-slate-400">Select a point or file to preview</div>

    return (
        <div className="card p-4">
            <div className="flex items-center justify-between mb-2">
                <div className="font-medium truncate" title={path}>{path}</div>
                <div className="text-xs text-slate-400">{current.length} points</div>
            </div>
            <Plot
                data={[{
                    x: time,
                    y: current,
                    type: 'scattergl',
                    mode: 'lines',
                    line: { color: '#22d3ee', width: 1 },
                } as any]}
                layout={{
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    autosize: true,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    xaxis: { title: 'Time (s)', gridcolor: '#1f2937', color: '#cbd5e1' },
                    yaxis: { title: 'Current (pA)', gridcolor: '#1f2937', color: '#cbd5e1' },
                    showlegend: false,
                }}
                style={{ width: '100%', height: '260px' }}
                config={{ displayModeBar: false, responsive: true }}
            />
        </div>
    )
}
