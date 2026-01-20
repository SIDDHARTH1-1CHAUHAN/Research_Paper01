# Multi-Agent Fact-Checking System - Web Presentation

An interactive Next.js web application for presenting the Multi-Agent Fact-Checking System to professors, researchers, and stakeholders.

## Features

- **Interactive Flowchart** - Visual representation of the 6-agent pipeline
- **Live Metrics** - Real-time visualization of demo results
- **Comparison Charts** - Performance comparison with baseline systems
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Dark Mode Support** - Automatic dark/light theme
- **Export Ready** - Can be deployed as static site

## Quick Start

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

### Export Static Site

```bash
npm run build
```

The static site will be in the `out/` directory, ready to deploy.

## What's Included

### Pages

1. **Overview** - System introduction and key features
2. **Architecture** - Interactive flowchart showing all 6 agents
3. **Results** - Demo observations with metrics visualizations
4. **Comparison** - Performance comparison with research baselines

### Components

- `AgentFlowchart` - Interactive pipeline visualization
- `MetricsVisualization` - Charts for classification and performance metrics
- `ComparisonChart` - Baseline comparison visualizations

## For Professor Presentation

1. **Start the app:**
   ```bash
   npm run dev
   ```

2. **Navigate through tabs:**
   - **Overview** - Explain what the system does
   - **Architecture** - Show the 6-agent pipeline
   - **Results** - Present demo metrics (80% accuracy, 0.857 F1-score)
   - **Comparison** - Show 12.3% improvement over baseline

3. **Key talking points:**
   - Based on peer-reviewed research (arXiv:2506.17878v1)
   - All 6 agents working together
   - Explainable AI (not a black box)
   - Free-tier implementation

## Deployment Options

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Static Export (GitHub Pages, Netlify)

```bash
npm run build
# Upload the 'out' directory
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Tech Stack

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Lucide React** - Icons

## Customization

### Update Demo Results

Edit the data in components:
- `components/MetricsVisualization.tsx` - Update metrics
- `components/ComparisonChart.tsx` - Update benchmark data
- `app/page.tsx` - Update text and statistics

### Change Colors

Edit `tailwind.config.js`:
```javascript
theme: {
  extend: {
    colors: {
      primary: '#your-color',
      // ...
    },
  },
},
```

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

MIT License - Same as main project

## Support

For issues or questions about the web app, see the main project README.
