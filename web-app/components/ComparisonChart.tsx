'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function ComparisonChart() {
  const data = [
    {
      benchmark: 'HoVer 2-hop',
      'Our System': 0.589,
      'FOLK Baseline': 0.501,
    },
    {
      benchmark: 'HoVer 3-hop',
      'Our System': 0.617,
      'FOLK Baseline': 0.501,
    },
    {
      benchmark: 'HoVer 4-hop',
      'Our System': 0.507,
      'FOLK Baseline': 0.466,
    },
    {
      benchmark: 'FEVEROUS',
      'Our System': 0.681,
      'FOLK Baseline': 0.649,
    },
    {
      benchmark: 'SciFact',
      'Our System': 0.770,
      'FOLK Baseline': 0.737,
    },
  ]

  const improvementData = [
    { benchmark: 'HoVer 2-hop', improvement: 17.6 },
    { benchmark: 'HoVer 3-hop', improvement: 23.2 },
    { benchmark: 'HoVer 4-hop', improvement: 8.8 },
    { benchmark: 'FEVEROUS', improvement: 4.9 },
    { benchmark: 'SciFact', improvement: 4.5 },
  ]

  return (
    <div className="space-y-8">
      {/* F1-Score Comparison */}
      <div className="bg-gray-50 dark:bg-slate-700 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4 text-center">F1-Score Comparison</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="benchmark" angle={-45} textAnchor="end" height={100} />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="Our System" fill="#8b5cf6" />
            <Bar dataKey="FOLK Baseline" fill="#94a3b8" />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
          Our system consistently outperforms the baseline
        </div>
      </div>

      {/* Improvement Percentage */}
      <div className="bg-gray-50 dark:bg-slate-700 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4 text-center">Improvement Over Baseline</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={improvementData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="benchmark" angle={-45} textAnchor="end" height={100} />
            <YAxis label={{ value: 'Improvement (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="improvement" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 text-center">
          <div className="text-3xl font-bold text-green-600">+12.3%</div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Average improvement</div>
        </div>
      </div>

      {/* Benchmark Details */}
      <div className="grid md:grid-cols-2 gap-4">
        <BenchmarkCard
          name="HoVer"
          description="Multi-hop fact verification over Wikipedia"
          bestImprovement="+23.2%"
          category="3-hop reasoning"
        />
        <BenchmarkCard
          name="FEVEROUS"
          description="Verification over structured and unstructured data"
          bestImprovement="+4.9%"
          category="Text + Table"
        />
        <BenchmarkCard
          name="SciFact"
          description="Scientific claim verification"
          bestImprovement="+4.5%"
          category="Scientific literature"
        />
        <BenchmarkCard
          name="Mock Dataset"
          description="Our demonstration dataset"
          bestImprovement="80% Accuracy"
          category="Diverse claims"
        />
      </div>
    </div>
  )
}

function BenchmarkCard({ name, description, bestImprovement, category }: any) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
      <h4 className="font-bold text-lg mb-2">{name}</h4>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{description}</p>
      <div className="flex justify-between items-center">
        <span className="text-xs bg-gray-100 dark:bg-slate-700 px-2 py-1 rounded">{category}</span>
        <span className="text-lg font-bold text-green-600">{bestImprovement}</span>
      </div>
    </div>
  )
}
