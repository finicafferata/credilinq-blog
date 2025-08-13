import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Header } from './components/Header';
import { LandingPage } from './pages/LandingPage'; // Keep landing page eager loaded
import { lazyWithFallback } from './utils/lazyLoader';

// Lazy load pages with intelligent grouping
const ImprovedDashboard = lazyWithFallback(
  () => import('./pages/ImprovedDashboard'),
  { loadingMessage: 'Loading dashboard...', name: 'Dashboard' }
);

// Content Creation Features
const EnhancedBlogCreator = lazyWithFallback(
  () => import('./pages/EnhancedBlogCreator') ,
  { loadingMessage: 'Loading blog creator...', name: 'BlogCreator' }
);

const EnhancedBlogEditor = lazyWithFallback(
  () => import('./pages/EnhancedBlogEditor'),
  { loadingMessage: 'Loading blog editor...', name: 'BlogEditor' }
);

const WorkflowPage = lazyWithFallback(
  () => import('./pages/WorkflowPage'),
  { loadingMessage: 'Loading workflow...', name: 'Workflow' }
);

// Campaign Management Features
const Campaigns = lazyWithFallback(
  () => import('./pages/Campaigns'),
  { loadingMessage: 'Loading campaigns...', name: 'Campaigns' }
);

const CampaignView = lazyWithFallback(
  () => import('./pages/CampaignView').then(module => ({ default: module.CampaignView })),
  { loadingMessage: 'Loading campaign details...', name: 'CampaignView' }
);

const CampaignWizard = lazyWithFallback(
  () => import('./pages/CampaignWizard').then(module => ({ default: module.CampaignWizard })),
  { loadingMessage: 'Loading campaign wizard...', name: 'CampaignWizard' }
);

// Analytics Features
const Analytics = lazyWithFallback(
  () => import('./pages/Analytics').then(module => ({ default: module.Analytics })),
  { loadingMessage: 'Loading analytics...', name: 'Analytics' }
);

const AnalyticsDashboard = lazyWithFallback(
  () => import('./pages/AnalyticsDashboard').then(module => ({ default: module.AnalyticsDashboard })),
  { loadingMessage: 'Loading analytics dashboard...', name: 'AnalyticsDashboard' }
);

const AdvancedReporting = lazyWithFallback(
  () => import('./pages/AdvancedReporting').then(module => ({ default: module.AdvancedReporting })),
  { loadingMessage: 'Loading advanced reporting...', name: 'AdvancedReporting' }
);

// Competitor Intelligence Features
const CompetitorIntelligence = lazyWithFallback(
  () => import('./pages/CompetitorIntelligence').then(module => ({ default: module.CompetitorIntelligence })),
  { loadingMessage: 'Loading competitor intelligence...', name: 'CompetitorIntelligence' }
);

const CompetitorManagement = lazyWithFallback(
  () => import('./pages/CompetitorManagement').then(module => ({ default: module.CompetitorManagement })),
  { loadingMessage: 'Loading competitor management...', name: 'CompetitorManagement' }
);

const AddCompetitor = lazyWithFallback(
  () => import('./pages/AddCompetitor').then(module => ({ default: module.AddCompetitor })),
  { loadingMessage: 'Loading add competitor form...', name: 'AddCompetitor' }
);

const EditCompetitor = lazyWithFallback(
  () => import('./pages/EditCompetitor').then(module => ({ default: module.EditCompetitor })),
  { loadingMessage: 'Loading edit competitor form...', name: 'EditCompetitor' }
);

const CompetitorDetail = lazyWithFallback(
  () => import('./pages/CompetitorDetail').then(module => ({ default: module.CompetitorDetail })),
  { loadingMessage: 'Loading competitor details...', name: 'CompetitorDetail' }
);

const AIContentAnalysis = lazyWithFallback(
  () => import('./pages/AIContentAnalysis').then(module => ({ default: module.AIContentAnalysis })),
  { loadingMessage: 'Loading AI content analysis...', name: 'AIContentAnalysis' }
);

const AlertDetail = lazyWithFallback(
  () => import('./pages/AlertDetail').then(module => ({ default: module.AlertDetail })),
  { loadingMessage: 'Loading alert...', name: 'AlertDetail' }
);

const AlertsOverview = lazyWithFallback(
  () => import('./pages/AlertsOverview').then(module => ({ default: module.AlertsOverview })),
  { loadingMessage: 'Loading alerts overview...', name: 'AlertsOverview' }
);

// Other Features
const KnowledgeBase = lazyWithFallback(
  () => import('./pages/KnowledgeBase').then(module => ({ default: module.KnowledgeBase })),
  { loadingMessage: 'Loading knowledge base...', name: 'KnowledgeBase' }
);

const Settings = lazyWithFallback(
  () => import('./pages/Settings'),
  { loadingMessage: 'Loading settings...', name: 'Settings' }
);

const ImageAgent = lazyWithFallback(
  () => import('./pages/ImageAgent'),
  { loadingMessage: 'Loading image generator...', name: 'ImageAgent' }
);

const IntegrationManagement = lazyWithFallback(
  () => import('./pages/IntegrationManagement').then(module => ({ default: module.IntegrationManagement })),
  { loadingMessage: 'Loading integration management...', name: 'IntegrationManagement' }
);

function App() {
  return (
    <Router 
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}
    >
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={
          <>
            <Header />
            <ImprovedDashboard />
          </>
        } />
        <Route path="/new" element={
          <>
            <Header />
            <EnhancedBlogCreator />
          </>
        } />
        <Route path="/edit/:blogId" element={
          <>
            <Header />
            <EnhancedBlogEditor />
          </>
        } />
        <Route path="/campaign/:blogId" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <CampaignView />
            </main>
          </div>
        } />
        <Route path="/campaigns/:campaignId" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <CampaignView />
            </main>
          </div>
        } />
        <Route path="/campaign-wizard/:campaignId" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <CampaignWizard />
            </main>
          </div>
        } />
        <Route path="/campaigns" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <Campaigns onNavigate={() => {}} />
            </main>
          </div>
        } />
        <Route path="/analytics" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <Analytics />
            </main>
          </div>
        } />
        <Route path="/knowledge-base" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <KnowledgeBase />
            </main>
          </div>
        } />
        <Route path="/settings" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <Settings />
            </main>
          </div>
        } />
        <Route path="/workflow" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <WorkflowPage />
            </main>
          </div>
        } />
        <Route path="/image-agent" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <ImageAgent />
            </main>
          </div>
        } />
        
        {/* Competitor Intelligence Routes */}
        <Route path="/competitor-intelligence" element={
          <>
            <Header />
            <CompetitorIntelligence />
          </>
        } />
        <Route path="/competitor-intelligence/competitors" element={
          <>
            <Header />
            <CompetitorManagement />
          </>
        } />
        <Route path="/competitor-intelligence/competitors/new" element={
          <>
            <Header />
            <AddCompetitor />
          </>
        } />
        <Route path="/competitor-intelligence/competitors/:id" element={
          <>
            <Header />
            <CompetitorDetail />
          </>
        } />
        <Route path="/competitor-intelligence/competitors/:id/edit" element={
          <>
            <Header />
            <EditCompetitor />
          </>
        } />
        <Route path="/competitor-intelligence/dashboard" element={
          <>
            <Header />
            <CompetitorIntelligence />
          </>
        } />
        <Route path="/competitor-intelligence/analytics" element={
          <>
            <Header />
            <AnalyticsDashboard />
          </>
        } />
        <Route path="/competitor-intelligence/alerts" element={
          <>
            <Header />
            <AlertsOverview />
          </>
        } />
        <Route path="/competitor-intelligence/ai-analysis" element={
          <>
            <Header />
            <AIContentAnalysis />
          </>
        } />
        <Route path="/competitor-intelligence/alerts/:alertId" element={
          <>
            <Header />
            <AlertDetail />
          </>
        } />
        <Route path="/competitor-intelligence/reporting" element={
          <>
            <Header />
            <AdvancedReporting />
          </>
        } />
        <Route path="/competitor-intelligence/integrations" element={
          <>
            <Header />
            <IntegrationManagement />
          </>
        } />
      </Routes>
      <Toaster 
        position="top-right"
        reverseOrder={false}
        gutter={8}
        containerClassName=""
        containerStyle={{}}
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#10B981',
              secondary: '#fff',
            },
          },
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#EF4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </Router>
  );
}

export default App;
