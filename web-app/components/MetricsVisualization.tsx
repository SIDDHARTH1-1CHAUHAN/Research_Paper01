'use client'

import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'

export default function MetricsVisualization() {
  const confusionData = [
    { name: 'True Positives', value: 6, color: '#10b981' },
    { name: 'True Negatives', value: 2, color: '#3b82f6' },
    { name: 'False Positives', value: 0, color: '#ef4444' },
    { name: 'False Negatives', value: 2, color: '#f59e0b' },
  ]

  const performanceData = [
    { metric: 'Accuracy', value: 0.80 },
    { metric: 'Precision', value: 1.00 },
    { metric: 'Recall', value: 0.75 },
    { metric: 'F1-Score', value: 0.857 },
  ]

  const explanationData = [
    { metric: 'Coverage', value: 0.85 },
    { metric: 'Soundness', value: 0.82 },
    { metric: 'Readability', value: 0.68 },
  ]

  return (
    <div className="grid md:grid-cols-2 gap-8">
      {/* Confusion Matrix Pie Chart */}
      <div className="bg-gray-50 dark:bg-slate-700 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4 text-center">Confusion Matrix</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={confusionData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value}`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {confusionData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
        <div className="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
          8/10 correct predictions (80% accuracy)
        </div>
      </div>

      {/* Classification Metrics Bar Chart */}
      <div className="bg-gray-50 dark:bg-slate-700 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4 text-center">Classification Metrics</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
          All metrics between 0.75-1.00
        </div>
      </div>

      {/* Explanation Quality */}
      <div className="bg-gray-50 dark:bg-slate-700 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4 text-center">Explanation Quality</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={explanationData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 1]} />
            <YAxis dataKey="metric" type="category" />
            <Tooltip />
            <Bar dataKey="value" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
          Overall quality: 0.7833/1.0
        </div>
      </div>

      {/* Processing Stats */}
      <div className="bg-gray-50 dark:bg-slate-700 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">Processing Statistics</h3>
        <div className="space-y-4">
          <StatBar label="Queries per Claim" value={5.4} max={10} color="blue" />
          <StatBar label="Evidence per Claim" value={5.4} max={10} color="purple" />
          <StatBar label="High Credibility" value={66.67} max={100} color="green" suffix="%" />
        </div>
        <div className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-lg">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Performance Summary</div>
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600">10</div>
              <div className="text-xs text-gray-500">Claims Processed</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600">54</div>
              <div className="text-xs text-gray-500">Total Queries</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatBar({ label, value, max, color, suffix = '' }: any) {
  const percentage = (value / max) * 100
  const colorClasses: any = {
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    green: 'bg-green-500',
  }

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="font-semibold">{label}</span>
        <span className="font-bold">{value}{suffix}</span>
      </div>
      <div className="w-full bg-gray-300 dark:bg-gray-600 rounded-full h-3">
        <div
          className={`${colorClasses[color]} h-3 rounded-full transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}
