"use client"

import dynamic from 'next/dynamic'
import { useMemo } from 'react'
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

export type ClusterPlotProps = {
    embedding: number[][]
    labels: number[]
    highlight?: number[]
    paths?: string[]
    onSelect?: (indices: number[]) => void
}

export default function ClusterPlot({ embedding, labels, highlight = [], paths = [], onSelect }: ClusterPlotProps) {
    const x = embedding.map((p) => p[0])
    const y = embedding.map((p) => p[1])

    const colors = useMemo(() => {
        const palette = [
            '#6366f1', '#22c55e', '#eab308', '#ef4444', '#06b6d4', '#f97316', '#84cc16', '#d946ef', '#a855f7', '#14b8a6'
        ]
        return labels.map((l) => (l >= 0 ? palette[l % palette.length] : '#94a3b8'))
    }, [labels])

    const markerSize = 10

    const selectedpoints = highlight

    return (
        <div className="card p-4">
            <Plot
                data={[
                    {
                        x,
                        y,
                        text: paths,
                        type: 'scattergl',
                        mode: 'markers',
                        marker: {
                            color: colors,
                            size: markerSize,
                            opacity: 0.9,
                            line: { width: 1, color: '#0f172a' },
                        },
                        selectedpoints,
                        selected: { marker: { color: '#ffffff', size: markerSize + 2, line: { color: '#22d3ee', width: 2 } } },
                        unselected: { marker: { opacity: 0.6 } },
                    } as any,
                ]}
                layout={{
                    dragmode: 'lasso',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    autosize: true,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    xaxis: { zeroline: false, showgrid: true, gridcolor: '#1f2937', color: '#cbd5e1' },
                    yaxis: { zeroline: false, showgrid: true, gridcolor: '#1f2937', color: '#cbd5e1' },
                    showlegend: false,
                }}
                style={{ width: '100%', height: '520px' }}
                config={{ displayModeBar: true, responsive: true }}
                onSelected={(ev) => {
                    const inds = (ev?.points || []).map((p: any) => p.pointIndex)
                    onSelect?.(inds)
                }}
            />
        </div>
    )
}
