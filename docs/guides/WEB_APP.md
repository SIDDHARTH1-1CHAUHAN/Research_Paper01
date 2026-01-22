# Web App - Professor Presentation Guide

## Quick Start (3 Easy Steps)

### Step 1: Install Dependencies

```bash
cd web-app
npm install
```

**Duration:** 2-3 minutes

### Step 2: Start the App

```bash
npm run dev
```

**The app will open at:** http://localhost:3000

### Step 3: Present!

Navigate through the 4 tabs to present your system:

---

## Presentation Flow

### Tab 1: Overview (2-3 minutes)

**What to show:**
- Hero section with key metrics
  - 80% Accuracy
  - 0.857 F1-Score
  - 100% Precision
- 6 specialized agents working together
- Key features cards

**What to say:**
> "This is a multi-agent AI system that automatically verifies factual claims. It uses 6 specialized agents working together in a pipeline. As you can see, we achieved 80% accuracy with perfect precision on our test dataset."

### Tab 2: Architecture (3-4 minutes)

**What to show:**
- Interactive flowchart of the entire pipeline
- Each agent's role and output
- Post-processing agents (XAI and RL)

**What to say:**
> "Let me show you how the system works. A claim comes in at the top, goes through 6 agents, and produces a verdict with explanations. Each agent has a specific job - one breaks down the claim, one searches for evidence, one checks source credibility, and so on."

**Point out:**
- Agent 1: FOL-based decomposition
- Agent 3: 3-stage evidence pipeline
- Agent 5: Explainable AI (transparent decisions)
- Agent 6: Continuous improvement

### Tab 3: Results (4-5 minutes)

**What to show:**
- Confusion matrix (pie chart)
- Classification metrics (bar chart)
- Explanation quality metrics
- Sample verified claims

**What to say:**
> "Here are the actual results from our demo. You can see the confusion matrix showing 8 out of 10 correct predictions. Our precision is perfect at 100%, meaning we had zero false positives. The explanation quality is 0.78 out of 1.0, showing our system provides clear, understandable reasoning."

**Highlight:**
- Zero false positives (red slice is 0)
- High explanation quality
- Real examples of verified claims

### Tab 4: Comparison (2-3 minutes)

**What to show:**
- F1-Score comparison with baseline
- Improvement percentages
- Average +12.3% improvement

**What to say:**
> "This system is based on peer-reviewed research published in 2025. When compared to baseline systems on academic benchmarks, it shows consistent improvements. The average improvement is 12.3%, with the best result being 23.2% improvement on complex multi-hop reasoning tasks."

---

## Technical Setup

### Prerequisites

- Node.js 18+ installed
- NPM or Yarn package manager

### Installation

```bash
# Navigate to web app directory
cd C:\Users\Dell\Desktop\2026\Research_Paper_01\multi-agent-fact-checker\web-app

# Install dependencies
npm install

# Start development server
npm run dev
```

### Building for Production

```bash
npm run build
npm start
```

### Export Static Site

```bash
npm run build
```

Static files will be in `out/` directory.

---

## If Professor Asks Technical Questions

**Q: "Is this responsive?"**
A: "Yes, it works on desktop, tablet, and mobile. It also has automatic dark mode support."

**Q: "Can I access this later?"**
A: "Yes, it's on GitHub at https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01. You can clone it and run it anytime. I can also deploy it to a live URL if you'd like."

**Q: "What framework did you use?"**
A: "Next.js 14 with TypeScript and Tailwind CSS for styling. The charts are built with Recharts for interactive visualizations."

**Q: "How does this relate to the Python code?"**
A: "This web app visualizes the results from the Python fact-checking system. The demo results shown here come from running the Python system on our mock dataset. All the metrics are real outputs from the actual system."

---

## Troubleshooting

### Port already in use

```bash
# Use a different port
npm run dev -- -p 3001
```

### Dependencies not installing

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Build errors

```bash
# Check Node version (need 18+)
node --version

# Update if needed
nvm install 18
nvm use 18
```

---

## Deployment Options

### Option 1: Vercel (Easiest - Free)

```bash
npm i -g vercel
vercel
```

Follow prompts. Get a live URL in ~2 minutes.

### Option 2: Netlify

1. Push to GitHub
2. Go to netlify.com
3. "New site from Git"
4. Select your repo
5. Build command: `npm run build`
6. Publish directory: `out`

### Option 3: GitHub Pages

```bash
npm run build
# Upload 'out' directory to gh-pages branch
```

---

## Features Showcase

### Interactive Elements

- **Tab Navigation** - Click different tabs to see different sections
- **Responsive Charts** - Hover over charts for detailed tooltips
- **Dark Mode** - Automatically adapts to system theme
- **Mobile Friendly** - Works on all screen sizes

### Visualizations

1. **Pie Chart** - Confusion matrix breakdown
2. **Bar Charts** - Classification metrics, explanation quality
3. **Progress Bars** - Processing statistics
4. **Comparison Charts** - Baseline vs our system

---

## Customization (If Needed)

### Update Metrics

Edit `components/MetricsVisualization.tsx`:
```typescript
const performanceData = [
  { metric: 'Accuracy', value: 0.80 }, // Change value here
  // ...
]
```

### Change Colors

Edit `tailwind.config.js`:
```javascript
theme: {
  extend: {
    colors: {
      primary: '#2563eb', // Change to your color
    },
  },
},
```

---

## Best Practices for Presentation

### Before Starting

- [ ] App is running and loaded at http://localhost:3000
- [ ] Tested all 4 tabs work correctly
- [ ] Browser is in full-screen mode
- [ ] Close unnecessary tabs/applications

### During Presentation

- **Speak confidently** - You built this!
- **Use the data** - Point to specific numbers
- **Tell a story** - Walk through the pipeline step by step
- **Be ready for questions** - Know where the GitHub repo is

### After Presentation

- Share the GitHub link
- Offer to deploy a live demo
- Mention the DEMO_OBSERVATIONS.md file
- Reference the RESEARCH_PAPER.md for methodology

---

## Summary

**To run the web presentation:**

```bash
cd web-app
npm install
npm run dev
```

**Then open:** http://localhost:3000

**Present in this order:**
1. Overview → Show metrics and features
2. Architecture → Walk through the pipeline
3. Results → Show actual performance
4. Comparison → Demonstrate improvement

**Good luck with your presentation!** 🚀
