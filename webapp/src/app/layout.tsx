import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
    title: 'DataCha Enhanced - Interactive Clustering',
    description: 'Interactive visualization and Active Learning for electrophysiology datasets',
}

export default function RootLayout({
    children,
}: Readonly<{ children: React.ReactNode }>) {
    return (
        <html lang="en">
            <body className="min-h-screen">
                {children}
            </body>
        </html>
    )
}
