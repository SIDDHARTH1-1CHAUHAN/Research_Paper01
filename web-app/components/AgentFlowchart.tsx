'use client'

import { ArrowRight, FileText, Search, Shield, BarChart3, Lightbulb, TrendingUp } from 'lucide-react'

export default function AgentFlowchart() {
  return (
    <div className="w-full overflow-x-auto">
      <div className="min-w-[800px] p-8">
        {/* Claim Input */}
        <div className="flex justify-center mb-8">
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-8 py-4 rounded-xl shadow-lg text-center">
            <div className="font-bold text-lg">User Claim</div>
            <div className="text-sm opacity-90 mt-1">Natural Language Input</div>
          </div>
        </div>

        <div className="flex justify-center mb-4">
          <ArrowRight className="w-6 h-6 text-gray-400 rotate-90" />
        </div>

        {/* Pipeline */}
        <div className="space-y-8">
          {/* Agent 1 */}
          <AgentNode
            number="1"
            icon={<FileText className="w-6 h-6" />}
            title="Input Ingestion Agent"
            description="FOL-based claim decomposition"
            output="Verifiable Subclaims"
            color="blue"
          />

          {/* Agent 2 */}
          <AgentNode
            number="2"
            icon={<Search className="w-6 h-6" />}
            title="Query Generation Agent"
            description="Generate k=3 diverse queries per subclaim"
            output="Search Queries"
            color="green"
          />

          {/* Agent 3 */}
          <AgentNode
            number="3"
            icon={<Shield className="w-6 h-6" />}
            title="Evidence Seeking Agent"
            description="3-stage: Search → Credibility → Extract"
            output="Validated Evidence"
            color="yellow"
          />

          {/* Agent 4 */}
          <AgentNode
            number="4"
            icon={<BarChart3 className="w-6 h-6" />}
            title="Verdict Prediction Agent"
            description="Weighted evidence aggregation"
            output="Verdict + Explanation"
            color="purple"
          />
        </div>

        {/* Final Verdict */}
        <div className="flex justify-center my-4">
          <ArrowRight className="w-6 h-6 text-gray-400 rotate-90" />
        </div>

        <div className="flex justify-center mb-8">
          <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-8 py-4 rounded-xl shadow-lg text-center">
            <div className="font-bold text-lg">Final Verdict</div>
            <div className="text-sm opacity-90 mt-1">SUPPORTED / NOT_SUPPORTED</div>
          </div>
        </div>

        {/* Post-Processing Agents */}
        <div className="grid md:grid-cols-2 gap-8 mt-12">
          <PostProcessingAgent
            number="5"
            icon={<Lightbulb className="w-6 h-6" />}
            title="Explainable AI Agent"
            description="Generates LIME/SHAP-inspired explanations"
            features={['Feature Importance', 'Counterfactuals', 'Conflict Resolution']}
            color="pink"
          />

          <PostProcessingAgent
            number="6"
            icon={<TrendingUp className="w-6 h-6" />}
            title="Reinforcement Learning Agent"
            description="Tracks performance and optimizes"
            features={['Pattern Analysis', 'Performance Scoring', 'Suggestions']}
            color="indigo"
          />
        </div>
      </div>
    </div>
  )
}

function AgentNode({ number, icon, title, description, output, color }: any) {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    yellow: 'from-yellow-500 to-yellow-600',
    purple: 'from-purple-500 to-purple-600',
  }

  return (
    <>
      <div className="flex items-center justify-center">
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 w-full max-w-2xl">
          <div className="flex items-center gap-4">
            <div className={`w-12 h-12 rounded-full bg-gradient-to-r ${colorClasses[color]} text-white flex items-center justify-center font-bold text-xl flex-shrink-0`}>
              {number}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <div className={`text-${color}-500`}>{icon}</div>
                <h4 className="font-bold text-lg">{title}</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
              <div className="mt-2 inline-block bg-gray-100 dark:bg-slate-700 px-3 py-1 rounded text-sm font-semibold">
                → {output}
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="flex justify-center">
        <ArrowRight className="w-6 h-6 text-gray-400 rotate-90" />
      </div>
    </>
  )
}

function PostProcessingAgent({ number, icon, title, description, features, color }: any) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-10 h-10 rounded-full bg-${color}-500 text-white flex items-center justify-center font-bold`}>
          {number}
        </div>
        <div className={`text-${color}-500`}>{icon}</div>
      </div>
      <h4 className="font-bold text-lg mb-2">{title}</h4>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">{description}</p>
      <ul className="space-y-2">
        {features.map((feature: string, index: number) => (
          <li key={index} className="flex items-center gap-2 text-sm">
            <div className="w-1.5 h-1.5 rounded-full bg-gray-400" />
            <span>{feature}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
