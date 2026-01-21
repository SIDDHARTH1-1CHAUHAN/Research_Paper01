# System Improvement Guide

This document outlines various strategies to enhance the Multi-Agent Fact-Checking System using paid AI services, advanced techniques, and alternative approaches.

---

## 🚀 Current System (Free Tier)

**What We Use Now:**
- **LLM**: Ollama (Llama 3.2 3B / Qwen 2.5 3B) - Local, free
- **Search**: DuckDuckGo Search API - Free, no API key
- **Credibility**: Heuristic-based (domain analysis) - Free
- **Web Scraping**: Selenium + BeautifulSoup - Free

**Performance:**
- 80% Accuracy on mock dataset
- 0.857 F1-Score
- 100% Precision
- Works completely offline for LLM inference

---

## 💎 Premium Enhancements

### 1. Upgrade to Advanced LLMs

#### Option A: OpenAI GPT-4o
**Cost:** ~$5 per 1M input tokens, ~$15 per 1M output tokens

**Benefits:**
- Superior reasoning capabilities
- Better FOL decomposition accuracy
- More nuanced query generation
- Enhanced explanation quality

**Implementation:**
```yaml
# config/api_config.yaml
llm:
  primary_provider: 'openai'
  openai:
    api_key: 'your-api-key-here'
    model: 'gpt-4o'
    temperature: 0.1
```

**Expected Improvements:**
- Accuracy: 80% → 88-92%
- F1-Score: 0.857 → 0.90-0.95
- Explanation Quality: 0.78 → 0.85-0.92

#### Option B: Anthropic Claude 3.5 Sonnet
**Cost:** ~$3 per 1M input tokens, ~$15 per 1M output tokens

**Benefits:**
- Excellent at complex reasoning
- Strong citation capabilities
- Better handling of contradictory evidence
- Superior multi-hop reasoning

**Implementation:**
```python
# Add to src/utils/llm_interface.py
from anthropic import Anthropic

class LLMInterface:
    def __init__(self):
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def _generate_anthropic(self, prompt, system_prompt):
        response = self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
```

**Expected Improvements:**
- Multi-hop reasoning: +15-20% accuracy
- Scientific claims: +10-15% accuracy
- Explanation coherence: Significant improvement

#### Option C: Google Gemini 1.5 Pro
**Cost:** Free tier (50 requests/day), then ~$1.25 per 1M input tokens

**Benefits:**
- Large context window (1M tokens)
- Can process entire web pages in one go
- Good at structured output
- Cost-effective

**Implementation:**
```yaml
# config/api_config.yaml
llm:
  primary_provider: 'gemini'
  gemini:
    api_key: 'your-api-key-here'
    model: 'gemini-1.5-pro'
```

**Expected Improvements:**
- Accuracy: 80% → 85-90%
- Evidence extraction efficiency: +30%
- Context understanding: Significant improvement

---

### 2. Advanced Search APIs

#### Option A: SerperAPI (Google Search)
**Cost:** $50/month (2,500 searches), $200/month (15,000 searches)

**Benefits:**
- Access to Google's search index
- Rich snippets and knowledge panels
- Higher quality results than DuckDuckGo
- API rate limits suitable for production

**Implementation:**
```python
# src/agents/evidence_seeking.py
import requests

def search_web_serper(query: str) -> List[SearchResult]:
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 10})
    headers = {
        'X-API-KEY': os.getenv('SERPER_API_KEY'),
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    results = response.json()

    return [
        SearchResult(
            url=item['link'],
            title=item['title'],
            snippet=item['snippet']
        )
        for item in results.get('organic', [])
    ]
```

**Expected Improvements:**
- Evidence quality: +25%
- Relevant sources found: +40%
- Credible sources discovered: +30%

#### Option B: Bing Web Search API
**Cost:** $7/month (1,000 transactions), custom pricing for higher volumes

**Benefits:**
- Microsoft's search index
- Good for academic and news sources
- Reasonable pricing
- JSON API with structured results

**Expected Improvements:**
- Academic source discovery: +35%
- News article freshness: +40%

#### Option C: Tavily AI Search (Specialized for AI)
**Cost:** $29/month (1,000 searches), $99/month (5,000 searches)

**Benefits:**
- Designed for LLM fact-checking
- Pre-filtered credible sources
- Extracted content included
- Reduces need for separate web scraping

**Implementation:**
```python
from tavily import TavilyClient

def search_web_tavily(query: str) -> List[Evidence]:
    tavily = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    response = tavily.search(
        query=query,
        search_depth="advanced",
        include_domains=["edu", "gov", "org"]
    )

    return [
        Evidence(
            url=result['url'],
            content=result['content'],
            credibility_score=0.8,  # Pre-filtered by Tavily
            source=result['title']
        )
        for result in response['results']
    ]
```

**Expected Improvements:**
- End-to-end pipeline speed: +50%
- Evidence quality: +30%
- Credibility false positives: -60%

---

### 3. Premium Credibility Assessment

#### Option A: Media Bias/Fact Check (MBFC) API
**Cost:** Custom pricing (~$500-1000/month for API access)

**Benefits:**
- Professional fact-checking organization
- Detailed factuality ratings
- Bias assessment included
- Covers 5,000+ sources

**Implementation:**
```python
# src/utils/credibility_checker.py
def _check_mbfc_api(self, url: str) -> CredibilityScore:
    domain = urlparse(url).netloc
    response = requests.get(
        f"https://api.mediabiasfactcheck.com/v1/source/{domain}",
        headers={'Authorization': f'Bearer {os.getenv("MBFC_API_KEY")}'}
    )

    if response.status_code == 200:
        data = response.json()
        factuality = data['factuality']

        if factuality in ['Very High', 'High']:
            return CredibilityScore(level=CredibilityLevel.HIGH, score=0.95)
        elif factuality == 'Mostly Factual':
            return CredibilityScore(level=CredibilityLevel.MEDIUM, score=0.70)
        else:
            return CredibilityScore(level=CredibilityLevel.LOW, score=0.30)
```

**Expected Improvements:**
- Credibility assessment accuracy: +50%
- False positive rate: -70%
- Coverage of known sources: 5,000+ domains

#### Option B: NewsGuard API
**Cost:** Enterprise pricing (contact for quote)

**Benefits:**
- Real-time credibility ratings
- Detailed transparency reports
- Trust scores for 8,000+ news sites
- Regular updates

**Expected Improvements:**
- News source credibility: +60%
- Real-time misinformation detection: Enabled

#### Option C: Ground News API
**Cost:** Starting at $50/month

**Benefits:**
- Multiple source verification
- Political bias detection
- Fact-check integration
- Historical source reliability

**Expected Improvements:**
- Multi-source corroboration: +40%
- Bias-aware evidence selection: Enabled

---

### 4. Real-World Reinforcement Learning

**Current System:** Heuristic-based performance tracking (not ML-based)

#### Upgrade to Actual RL with Ray RLlib

**Benefits:**
- Learns optimal query generation strategies
- Adapts to domain-specific patterns
- Improves agent coordination
- Reduces redundant evidence gathering

**Implementation:**
```python
# src/agents/reinforcement_learning.py
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

class RLAgent:
    def __init__(self):
        ray.init()
        self.config = {
            "env": "FactCheckingEnv",
            "num_workers": 4,
            "framework": "torch",
        }
        self.trainer = PPO(config=self.config)

    def train_episode(self, claim: str, ground_truth: str):
        # State: claim features, current evidence
        # Action: which queries to generate, which sources to prioritize
        # Reward: based on verdict accuracy and efficiency

        result = self.trainer.train()
        return result
```

**Expected Improvements:**
- Query efficiency: +40% (fewer queries for same accuracy)
- Evidence selection: +25% accuracy
- Processing time: -30%

**Cost Considerations:**
- Requires GPU for training (~$0.50-2/hour on cloud)
- One-time training cost: ~$50-200
- Inference: negligible cost increase

---

### 5. Advanced Evidence Extraction

#### Option A: Firecrawl (AI-Powered Web Scraping)
**Cost:** $20/month (500 credits), $100/month (3,000 credits)

**Benefits:**
- Bypasses anti-scraping measures
- Extracts clean markdown from complex pages
- Handles JavaScript-heavy sites
- Built-in content cleaning

**Implementation:**
```python
from firecrawl import FirecrawlApp

def extract_evidence_firecrawl(url: str) -> str:
    app = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))
    result = app.scrape_url(url, params={'formats': ['markdown']})
    return result['markdown']
```

**Expected Improvements:**
- Successful content extraction: +50%
- JavaScript site handling: +80%
- Processing speed: +60%

#### Option B: ScrapingBee (Premium Proxy Service)
**Cost:** $49/month (50,000 API credits)

**Benefits:**
- Rotating proxies (avoid IP bans)
- JavaScript rendering
- Geographic targeting
- CAPTCHA solving

**Expected Improvements:**
- Success rate on protected sites: +70%
- Geographic content access: Enabled

---

### 6. Knowledge Base Integration

#### Option A: Wikipedia API + Wikidata
**Cost:** Free (donations appreciated)

**Benefits:**
- Structured knowledge graph
- Verified information
- Cross-referenced facts
- Multi-language support

**Implementation:**
```python
import wikipediaapi

def enrich_with_wikipedia(entity: str) -> Dict[str, Any]:
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(entity)

    return {
        'summary': page.summary,
        'categories': page.categories,
        'links': [link.title for link in page.links.values()],
        'verified': True  # Wikipedia editorial process
    }
```

**Expected Improvements:**
- Entity verification: +30%
- Structured data access: Enabled
- Multi-hop reasoning: +20%

#### Option B: Wolfram Alpha API
**Cost:** $3.50/month (2,000 calls), custom for higher volumes

**Benefits:**
- Computational knowledge engine
- Mathematical and scientific facts
- Verified data sources
- Natural language processing

**Expected Improvements:**
- Numerical claim verification: +60%
- Scientific accuracy: +40%
- Mathematical reasoning: Significantly enhanced

---

### 7. Enhanced Evaluation with Full Benchmarks

**Current:** 10 samples per benchmark (30 total)

#### Upgrade to Full Academic Evaluation

**FEVEROUS:**
- Current: 10 samples
- Full: 87,026 claims
- Expected F1: 0.68 (based on paper)

**HoVer:**
- Current: 10 samples
- Full: 21,749 claims
- Expected F1: 0.59-0.62 (based on paper)

**SciFact:**
- Current: 10 samples
- Full: 1,409 claims
- Expected F1: 0.77 (based on paper)

**Cost Considerations:**
- With paid APIs: ~$100-500 for full evaluation
- Processing time: 10-20 hours
- Results suitable for peer-reviewed publication

---

## 📊 Cost-Benefit Analysis

### Budget Tiers

#### Tier 1: Minimal Upgrade ($20-50/month)
**Recommended For:** Academic research, proof of concept

**Components:**
- Gemini 1.5 Pro (free tier + paid overflow)
- Firecrawl basic plan
- Wikipedia/Wikidata integration (free)

**Expected ROI:**
- Accuracy: 80% → 86%
- F1-Score: 0.857 → 0.89
- Total cost: ~$30/month

#### Tier 2: Professional Setup ($100-200/month)
**Recommended For:** Research publication, production demos

**Components:**
- OpenAI GPT-4o
- SerperAPI standard plan
- Firecrawl pro plan
- MBFC API access

**Expected ROI:**
- Accuracy: 80% → 90%
- F1-Score: 0.857 → 0.93
- Evidence quality: +40%
- Total cost: ~$150/month

#### Tier 3: Enterprise Grade ($500-1000/month)
**Recommended For:** Production deployment, commercial use

**Components:**
- Claude 3.5 Sonnet (high volume)
- Tavily AI Search advanced
- NewsGuard API
- Ray RLlib training infrastructure
- ScrapingBee unlimited

**Expected ROI:**
- Accuracy: 80% → 93-95%
- F1-Score: 0.857 → 0.95-0.97
- Processing speed: 3x faster
- Total cost: ~$700/month

---

## 🔬 Alternative Approaches

### 1. Ensemble Models
**Strategy:** Run multiple LLMs and vote on results

```python
def ensemble_verdict(claim: str) -> Verdict:
    verdicts = [
        verify_with_gpt4(claim),
        verify_with_claude(claim),
        verify_with_gemini(claim)
    ]
    return majority_vote(verdicts)
```

**Benefits:**
- Accuracy: +5-10%
- Confidence estimation: More reliable
- Cost: 3x LLM usage

### 2. Fine-Tuning Custom Models
**Strategy:** Fine-tune Llama 3 8B on fact-checking datasets

**Process:**
1. Gather 10,000+ labeled fact-checking examples
2. Fine-tune using LoRA/QLoRA
3. Deploy with Ollama

**Benefits:**
- Domain-specific performance: +15-25%
- Cost: One-time training (~$50-200), then free inference
- Latency: Much faster than API calls

**Datasets to Use:**
- FEVER (185,000 claims)
- LIAR (12,800 claims)
- MultiFC (34,000 claims)

### 3. Retrieval-Augmented Generation (RAG)
**Strategy:** Build local knowledge base of verified facts

**Implementation:**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class RAGFactChecker:
    def __init__(self):
        self.vectorstore = Chroma(
            persist_directory="./verified_facts_db",
            embedding_function=OpenAIEmbeddings()
        )

    def check_against_knowledge_base(self, claim: str) -> List[Evidence]:
        relevant_docs = self.vectorstore.similarity_search(claim, k=5)
        return [doc.page_content for doc in relevant_docs]
```

**Benefits:**
- Speed: Instant retrieval for known facts
- Cost: One-time embedding cost, then near-free
- Accuracy on common claims: +30%

### 4. Claim Verification Networks
**Strategy:** Cross-reference with professional fact-checkers

**APIs to Integrate:**
- ClaimBuster API (University of Texas)
- Full Fact API (UK fact-checking charity)
- Snopes API (partner program)
- PolitiFact API

**Benefits:**
- Human-verified claims: Direct access
- Pre-checked facts: Instant results
- Authority: Cite professional fact-checkers

---

## 🎯 Recommended Upgrade Path

### Phase 1: Quick Wins (Week 1)
1. Integrate Gemini 1.5 Pro (free tier)
2. Add Wikipedia API enrichment
3. Implement basic RAG with local knowledge base

**Cost:** $0-10
**Expected Improvement:** +5-8% accuracy

### Phase 2: Core Upgrades (Month 1)
1. Switch to OpenAI GPT-4o for critical agents
2. Add SerperAPI for better search results
3. Implement Firecrawl for robust scraping

**Cost:** ~$100/month
**Expected Improvement:** +10-12% accuracy

### Phase 3: Professional Features (Month 2-3)
1. Integrate MBFC API for credibility
2. Add ensemble voting across multiple LLMs
3. Fine-tune custom Llama model for specific domains

**Cost:** ~$150-200/month (one-time training cost)
**Expected Improvement:** +15-18% accuracy

### Phase 4: Production Ready (Month 4+)
1. Implement actual RL training with Ray
2. Full benchmark evaluation (300 samples → 110,000+ samples)
3. Deploy with caching, monitoring, and auto-scaling

**Cost:** ~$500-700/month
**Expected Result:** Publication-ready system with 92-95% accuracy

---

## 📈 Performance Tracking

After each upgrade, measure:

```python
# Evaluation metrics to track
metrics = {
    'accuracy': [],
    'f1_score': [],
    'precision': [],
    'recall': [],
    'evidence_quality': [],
    'explanation_quality': [],
    'processing_time': [],
    'cost_per_claim': [],
    'source_credibility_avg': []
}
```

**Compare:**
- Free tier baseline (current)
- After each paid integration
- Against published research benchmarks
- ROI: (Accuracy improvement / Cost increase)

---

## 🤝 Community & Open Source Alternatives

### Free Alternatives to Paid Services:

1. **LLMs:**
   - Mistral 7B (better than Llama 3.2 3B)
   - Mixtral 8x7B (excellent reasoning)
   - Llama 3.1 70B (if you have GPU)

2. **Search:**
   - SearxNG (meta-search, self-hosted)
   - Common Crawl (web corpus access)

3. **Credibility:**
   - Academic database crossref
   - Archive.org for historical verification
   - OpenPageRank for site authority

4. **Knowledge Bases:**
   - DBpedia (structured Wikipedia)
   - ConceptNet (common sense reasoning)
   - YAGO (knowledge graph)

---

## 📚 Further Reading

- **Papers:**
  - "Truth of Varying Shades: Analyzing Language in Fake News and Political Fact-Checking"
  - "FEVER: a Large-scale Dataset for Fact Extraction and VERification"
  - "Explainable Automated Fact-Checking: A Survey"

- **Tools:**
  - LangChain documentation for agent orchestration
  - Ray RLlib tutorials for reinforcement learning
  - Hugging Face model hub for fine-tuning

- **Benchmarks:**
  - FEVER dataset: fever.ai
  - HoVer dataset: hover-nlp.github.io
  - SciFact: scifact.apps.allenai.org

---

## 💡 Key Takeaways

1. **Start Small:** Gemini 1.5 Pro free tier is an excellent first upgrade
2. **Measure Everything:** Track accuracy, cost, and speed for each change
3. **ROI Focus:** Some free alternatives (fine-tuning, RAG) offer better ROI than paid APIs
4. **Incremental:** Don't upgrade everything at once - validate each improvement
5. **Publication Ready:** Full benchmark evaluation ($100-500) is essential for research papers

---

## 🔗 Quick Implementation Checklist

- [ ] Set up API keys in `.env` file
- [ ] Update `config/api_config.yaml` with chosen providers
- [ ] Modify `src/utils/llm_interface.py` to add new LLM providers
- [ ] Update `src/agents/evidence_seeking.py` for new search APIs
- [ ] Implement credibility API in `src/utils/credibility_checker.py`
- [ ] Run evaluation before and after each change
- [ ] Document improvements in DEMO_OBSERVATIONS.md
- [ ] Update RESEARCH_PAPER.md with new methodology
- [ ] Commit changes with clear descriptions

---

**Ready to upgrade?** Start with the free/low-cost options and measure impact before investing in premium services.

For questions or implementation help, refer to the main README.md and ARCHITECTURE.md.
