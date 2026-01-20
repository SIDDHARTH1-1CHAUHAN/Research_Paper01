'use client'

import { useState } from 'react'
import {
  CheckCircle2,
  XCircle,
  ArrowRight,
  BarChart3,
  FileText,
  Brain,
  Search,
  Shield,
  Lightbulb,
  TrendingUp,
  Github
} from 'lucide-react'
import AgentFlowchart from '@/components/AgentFlowchart'
import MetricsVisualization from '@/components/MetricsVisualization'
import ComparisonChart from '@/components/ComparisonChart'

export default function Home() {
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold gradient-text">
                Multi-Agent Fact-Checking System
              </h1>
              <p className="text-gray-600 dark:text-gray-300 mt-2">
                Research-Grade AI System for Automated Fact Verification
              </p>
            </div>
            <a
              href="https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-700 transition"
            >
              <Github size={20} />
              <span className="hidden sm:inline">View on GitHub</span>
            </a>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
        <div className="flex gap-4 overflow-x-auto pb-2">
          {['overview', 'architecture', 'results', 'comparison'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-lg font-semibold whitespace-nowrap transition ${
                activeTab === tab
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white dark:bg-slate-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-600'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && <OverviewSection />}
        {activeTab === 'architecture' && <ArchitectureSection />}
        {activeTab === 'results' && <ResultsSection />}
        {activeTab === 'comparison' && <ComparisonSection />}
      </div>

      {/* Footer */}
      <footer className="mt-16 bg-white dark:bg-slate-800 border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-gray-600 dark:text-gray-400">
          <p>Based on research: <span className="font-semibold">arXiv:2506.17878v1 (2025)</span></p>
          <p className="mt-2 text-sm">Multi-Agent System with Advanced Evidence Retrieval</p>
        </div>
      </footer>
    </main>
  )
}

function OverviewSection() {
  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8">
        <h2 className="text-4xl font-bold mb-4">What is This System?</h2>
        <p className="text-xl text-gray-700 dark:text-gray-300 mb-6">
          An <span className="font-semibold text-blue-600">intelligent multi-agent system</span> that automatically
          verifies factual claims using <span className="font-semibold text-purple-600">6 specialized AI agents</span> working together.
        </p>

        <div className="grid md:grid-cols-3 gap-6 mt-8">
          <div className="p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
            <div className="text-4xl font-bold text-blue-600 mb-2">80%</div>
            <div className="text-gray-700 dark:text-gray-300 font-semibold">Accuracy</div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">On diverse claims</div>
          </div>
          <div className="p-6 bg-purple-50 dark:bg-purple-900/20 rounded-xl">
            <div className="text-4xl font-bold text-purple-600 mb-2">0.857</div>
            <div className="text-gray-700 dark:text-gray-300 font-semibold">F1-Score</div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">Balanced performance</div>
          </div>
          <div className="p-6 bg-green-50 dark:bg-green-900/20 rounded-xl">
            <div className="text-4xl font-bold text-green-600 mb-2">100%</div>
            <div className="text-gray-700 dark:text-gray-300 font-semibold">Precision</div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">No false positives</div>
          </div>
        </div>
      </div>

      {/* Key Features */}
      <div className="grid md:grid-cols-2 gap-6">
        <FeatureCard
          icon={<Brain className="w-8 h-8" />}
          title="6 Specialized Agents"
          description="Each agent has a specific role: decomposition, search, retrieval, prediction, explanation, and optimization."
          color="blue"
        />
        <FeatureCard
          icon={<Shield className="w-8 h-8" />}
          title="Credibility Checking"
          description="Validates source reliability using domain analysis and factuality ratings."
          color="green"
        />
        <FeatureCard
          icon={<Lightbulb className="w-8 h-8" />}
          title="Explainable AI"
          description="LIME/SHAP-inspired explanations showing exactly why each decision was made."
          color="yellow"
        />
        <FeatureCard
          icon={<TrendingUp className="w-8 h-8" />}
          title="Continuous Learning"
          description="RL agent tracks performance and suggests improvements over time."
          color="purple"
        />
      </div>

      {/* What Makes It Better */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl shadow-xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-6">What Makes This System Better?</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="flex gap-4">
            <CheckCircle2 className="w-6 h-6 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-lg mb-2">Based on Peer-Reviewed Research</h3>
              <p className="text-blue-100">
                Implements architecture from arXiv:2506.17878v1 showing 12.3% improvement over baselines
              </p>
            </div>
          </div>
          <div className="flex gap-4">
            <CheckCircle2 className="w-6 h-6 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-lg mb-2">Transparent & Explainable</h3>
              <p className="text-blue-100">
                Not a black box - shows feature importance and reasoning behind every verdict
              </p>
            </div>
          </div>
          <div className="flex gap-4">
            <CheckCircle2 className="w-6 h-6 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-lg mb-2">Free-Tier Implementation</h3>
              <p className="text-blue-100">
                Works completely free using heuristic methods, optional upgrade to LLMs
              </p>
            </div>
          </div>
          <div className="flex gap-4">
            <CheckCircle2 className="w-6 h-6 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-lg mb-2">Modular Architecture</h3>
              <p className="text-blue-100">
                Each agent can be enhanced independently, easy to extend and maintain
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function ArchitectureSection() {
  return (
    <div className="space-y-8">
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8">
        <h2 className="text-3xl font-bold mb-6">System Architecture</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-8">
          The system uses a pipeline of 6 specialized agents, each handling a specific aspect of fact-checking:
        </p>

        <AgentFlowchart />

        <div className="mt-12 grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AgentCard
            number="1"
            icon={<FileText />}
            title="Input Ingestion"
            description="Decomposes complex claims into atomic subclaims using First-Order Logic"
            color="blue"
          />
          <AgentCard
            number="2"
            icon={<Search />}
            title="Query Generation"
            description="Creates diverse search queries (k=3) for each subclaim using SEO principles"
            color="green"
          />
          <AgentCard
            number="3"
            icon={<Shield />}
            title="Evidence Seeking"
            description="3-stage pipeline: Web Search → Credibility Check → Content Extraction"
            color="yellow"
          />
          <AgentCard
            number="4"
            icon={<BarChart3 />}
            title="Verdict Prediction"
            description="Aggregates evidence using weighted voting (HIGH: 1.0, MEDIUM: 0.6, LOW: 0.3)"
            color="purple"
          />
          <AgentCard
            number="5"
            icon={<Lightbulb />}
            title="Explainable AI"
            description="LIME/SHAP-inspired explanations with feature importance and counterfactuals"
            color="pink"
          />
          <AgentCard
            number="6"
            icon={<TrendingUp />}
            title="Reinforcement Learning"
            description="Tracks performance patterns and suggests system improvements"
            color="indigo"
          />
        </div>
      </div>
    </div>
  )
}

function ResultsSection() {
  return (
    <div className="space-y-8">
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8">
        <h2 className="text-3xl font-bold mb-6">Demo Results & Observations</h2>

        <MetricsVisualization />

        {/* Detailed Metrics */}
        <div className="mt-12 grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-2xl font-bold mb-4">Classification Metrics</h3>
            <div className="space-y-4">
              <MetricRow label="Accuracy" value="80.00%" color="blue" />
              <MetricRow label="F1-Score" value="0.8571" color="purple" />
              <MetricRow label="Precision" value="1.0000" color="green" />
              <MetricRow label="Recall" value="0.7500" color="yellow" />
            </div>

            <div className="mt-6 p-4 bg-gray-50 dark:bg-slate-700 rounded-lg">
              <h4 className="font-semibold mb-3">Confusion Matrix</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                  <span>True Positives: 6</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                  <span>True Negatives: 2</span>
                </div>
                <div className="flex items-center gap-2">
                  <XCircle className="w-5 h-5 text-red-500" />
                  <span>False Positives: 0</span>
                </div>
                <div className="flex items-center gap-2">
                  <XCircle className="w-5 h-5 text-red-500" />
                  <span>False Negatives: 2</span>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-2xl font-bold mb-4">Performance Metrics</h3>
            <div className="space-y-4">
              <MetricRow label="Mean Queries/Claim" value="5.4" color="blue" />
              <MetricRow label="Mean Evidence/Claim" value="5.4" color="purple" />
              <MetricRow label="High Credibility Ratio" value="66.67%" color="green" />
              <MetricRow label="Explanation Quality" value="0.7833" color="yellow" />
            </div>

            <div className="mt-6 p-4 bg-gray-50 dark:bg-slate-700 rounded-lg">
              <h4 className="font-semibold mb-3">Explanation Quality Breakdown</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Coverage:</span>
                  <span className="font-semibold">0.85</span>
                </div>
                <div className="flex justify-between">
                  <span>Soundness:</span>
                  <span className="font-semibold">0.82</span>
                </div>
                <div className="flex justify-between">
                  <span>Readability:</span>
                  <span className="font-semibold">0.68</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sample Claims */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8">
        <h3 className="text-2xl font-bold mb-6">Sample Verified Claims</h3>
        <div className="space-y-4">
          <ClaimResult
            claim="The Eiffel Tower was completed in 1889 and is located in Paris, France"
            truth="SUPPORTED"
            predicted="SUPPORTED"
            confidence={85}
            correct={true}
          />
          <ClaimResult
            claim="Albert Einstein was born in Germany in 1879 and won the Nobel Prize in Physics in 1921"
            truth="SUPPORTED"
            predicted="SUPPORTED"
            confidence={85}
            correct={true}
          />
          <ClaimResult
            claim="The Great Wall of China is visible from space and was built in the 5th century BC"
            truth="NOT_SUPPORTED"
            predicted="NOT_SUPPORTED"
            confidence={85}
            correct={true}
          />
        </div>
      </div>
    </div>
  )
}

function ComparisonSection() {
  return (
    <div className="space-y-8">
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8">
        <h2 className="text-3xl font-bold mb-6">Comparison with Baseline Systems</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-8">
          Performance comparison on academic benchmarks from the original research paper:
        </p>

        <ComparisonChart />

        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <div className="p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
            <h4 className="font-bold text-lg mb-2">HoVer (3-hop)</h4>
            <div className="text-3xl font-bold text-blue-600 mb-2">+23.2%</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              F1: 0.617 vs 0.501 baseline
            </p>
          </div>
          <div className="p-6 bg-purple-50 dark:bg-purple-900/20 rounded-xl">
            <h4 className="font-bold text-lg mb-2">FEVEROUS</h4>
            <div className="text-3xl font-bold text-purple-600 mb-2">+4.9%</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              F1: 0.681 vs 0.649 baseline
            </p>
          </div>
          <div className="p-6 bg-green-50 dark:bg-green-900/20 rounded-xl">
            <h4 className="font-bold text-lg mb-2">SciFact</h4>
            <div className="text-3xl font-bold text-green-600 mb-2">+4.5%</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              F1: 0.770 vs 0.737 baseline
            </p>
          </div>
        </div>

        <div className="mt-8 p-6 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl border-2 border-green-200 dark:border-green-700">
          <h4 className="font-bold text-xl mb-2">Average Improvement</h4>
          <div className="text-5xl font-bold text-green-600 mb-2">+12.3%</div>
          <p className="text-gray-700 dark:text-gray-300">
            Across all benchmarks compared to FOLK baseline system
          </p>
        </div>
      </div>
    </div>
  )
}

function FeatureCard({ icon, title, description, color }: any) {
  const colorClasses = {
    blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-600',
    green: 'bg-green-50 dark:bg-green-900/20 text-green-600',
    yellow: 'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-600',
    purple: 'bg-purple-50 dark:bg-purple-900/20 text-purple-600',
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 card-hover">
      <div className={`w-16 h-16 rounded-lg flex items-center justify-center mb-4 ${colorClasses[color]}`}>
        {icon}
      </div>
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="text-gray-600 dark:text-gray-400">{description}</p>
    </div>
  )
}

function AgentCard({ number, icon, title, description, color }: any) {
  return (
    <div className="bg-gray-50 dark:bg-slate-700 rounded-xl p-6 card-hover">
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-10 h-10 rounded-full bg-${color}-500 text-white flex items-center justify-center font-bold`}>
          {number}
        </div>
        <div className={`text-${color}-500`}>{icon}</div>
      </div>
      <h3 className="text-lg font-bold mb-2">{title}</h3>
      <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
    </div>
  )
}

function MetricRow({ label, value, color }: any) {
  return (
    <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-slate-700 rounded-lg">
      <span className="font-semibold">{label}</span>
      <span className={`text-xl font-bold text-${color}-600`}>{value}</span>
    </div>
  )
}

function ClaimResult({ claim, truth, predicted, confidence, correct }: any) {
  return (
    <div className={`p-4 rounded-lg border-2 ${correct ? 'border-green-200 bg-green-50 dark:bg-green-900/20' : 'border-red-200 bg-red-50 dark:bg-red-900/20'}`}>
      <div className="flex items-start gap-3">
        {correct ? (
          <CheckCircle2 className="w-6 h-6 text-green-500 flex-shrink-0 mt-1" />
        ) : (
          <XCircle className="w-6 h-6 text-red-500 flex-shrink-0 mt-1" />
        )}
        <div className="flex-1">
          <p className="text-gray-800 dark:text-gray-200 mb-2">{claim}</p>
          <div className="flex gap-4 text-sm">
            <span className="font-semibold">Ground Truth: {truth}</span>
            <span className="font-semibold">Predicted: {predicted}</span>
            <span>Confidence: {confidence}%</span>
          </div>
        </div>
      </div>
    </div>
  )
}
