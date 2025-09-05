import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
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

// const ContentBriefCreator = lazyWithFallback(
//   () => import('./pages/ContentBriefCreator'),
//   { loadingMessage: 'Loading content brief creator...', name: 'ContentBriefCreator' }
// );

// Campaign Management Features
const Campaigns = lazyWithFallback(
  () => import('./pages/Campaigns'),
  { loadingMessage: 'Loading campaigns...', name: 'Campaigns' }
);

const CampaignView = lazyWithFallback(
  () => import('./pages/CampaignView'),
  { loadingMessage: 'Loading campaign details...', name: 'CampaignView' }
);

const CampaignWizard = lazyWithFallback(
  () => import('./pages/CampaignWizard').then(module => ({ default: module.CampaignWizard })),
  { loadingMessage: 'Loading campaign wizard...', name: 'CampaignWizard' }
);

// Analytics Features - Commented out for production
// const Analytics = lazyWithFallback(
//   () => import('./pages/Analytics').then(module => ({ default: module.Analytics })),
//   { loadingMessage: 'Loading analytics...', name: 'Analytics' }
// );

// const AnalyticsDashboard = lazyWithFallback(
//   () => import('./pages/AnalyticsDashboard').then(module => ({ default: module.AnalyticsDashboard })),
//   { loadingMessage: 'Loading analytics dashboard...', name: 'AnalyticsDashboard' }
// );

// const AdvancedReporting = lazyWithFallback(
//   () => import('./pages/AdvancedReporting').then(module => ({ default: module.AdvancedReporting })),
//   { loadingMessage: 'Loading advanced reporting...', name: 'AdvancedReporting' }
// );

// Competitor Intelligence Features - Commented out for production
// const CompetitorIntelligence = lazyWithFallback(
//   () => import('./pages/CompetitorIntelligence').then(module => ({ default: module.CompetitorIntelligence })),
//   { loadingMessage: 'Loading competitor intelligence...', name: 'CompetitorIntelligence' }
// );

// const CompetitorManagement = lazyWithFallback(
//   () => import('./pages/CompetitorManagement').then(module => ({ default: module.CompetitorManagement })),
//   { loadingMessage: 'Loading competitor management...', name: 'CompetitorManagement' }
// );

// const AddCompetitor = lazyWithFallback(
//   () => import('./pages/AddCompetitor').then(module => ({ default: module.AddCompetitor })),
//   { loadingMessage: 'Loading add competitor form...', name: 'AddCompetitor' }
// );

// const EditCompetitor = lazyWithFallback(
//   () => import('./pages/EditCompetitor').then(module => ({ default: module.EditCompetitor })),
//   { loadingMessage: 'Loading edit competitor form...', name: 'EditCompetitor' }
// );

// const CompetitorDetail = lazyWithFallback(
//   () => import('./pages/CompetitorDetail').then(module => ({ default: module.CompetitorDetail })),
//   { loadingMessage: 'Loading competitor details...', name: 'CompetitorDetail' }
// );

// const AIContentAnalysis = lazyWithFallback(
//   () => import('./pages/AIContentAnalysis').then(module => ({ default: module.AIContentAnalysis })),
//   { loadingMessage: 'Loading AI content analysis...', name: 'AIContentAnalysis' }
// );

// const AlertDetail = lazyWithFallback(
//   () => import('./pages/AlertDetail').then(module => ({ default: module.AlertDetail })),
//   { loadingMessage: 'Loading alert...', name: 'AlertDetail' }
// );

// const AlertsOverview = lazyWithFallback(
//   () => import('./pages/AlertsOverview').then(module => ({ default: module.AlertsOverview })),
//   { loadingMessage: 'Loading alerts overview...', name: 'AlertsOverview' }
// );

// Other Features
const KnowledgeBase = lazyWithFallback(
  () => import('./pages/KnowledgeBase').then(module => ({ default: module.KnowledgeBase })),
  { loadingMessage: 'Loading knowledge base...', name: 'KnowledgeBase' }
);

const Settings = lazyWithFallback(
  () => import('./pages/Settings'),
  { loadingMessage: 'Loading settings...', name: 'Settings' }
);


const IntegrationManagement = lazyWithFallback(
  () => import('./pages/IntegrationManagement').then(module => ({ default: module.IntegrationManagement })),
  { loadingMessage: 'Loading integration management...', name: 'IntegrationManagement' }
);

// Campaign Orchestration Dashboard
const CampaignOrchestrationDashboard = lazyWithFallback(
  () => import('./pages/CampaignOrchestrationDashboard'),
  { loadingMessage: 'Loading orchestration dashboard...', name: 'CampaignOrchestrationDashboard' }
);

// Agent Management - Commented out for production
// const AgentManagement = lazyWithFallback(
//   () => import('./pages/AgentManagement'),
//   { loadingMessage: 'Loading agent management...', name: 'AgentManagement' }
// );

// Real-Time Monitoring - Commented out for production
// const RealTimeMonitoring = lazyWithFallback(
//   () => import('./pages/RealTimeMonitoring'),
//   { loadingMessage: 'Loading real-time monitoring...', name: 'RealTimeMonitoring' }
// );

// Performance Analytics - Commented out for production
// const PerformanceAnalytics = lazyWithFallback(
//   () => import('./pages/PerformanceAnalytics'),
//   { loadingMessage: 'Loading performance analytics...', name: 'PerformanceAnalytics' }
// );

// Predictive Analytics - Commented out for production
// const PredictiveAnalytics = lazyWithFallback(
//   () => import('./pages/PredictiveAnalytics'),
//   { loadingMessage: 'Loading predictive analytics...', name: 'PredictiveAnalytics' }
// );

// Integration Testing - Commented out for production
// const IntegrationTesting = lazyWithFallback(
//   () => import('./pages/IntegrationTesting'),
//   { loadingMessage: 'Loading integration testing...', name: 'IntegrationTesting' }
// );

// Content Review is now integrated directly into Campaign Details
// No separate page needed - unified workflow experience

// Layout wrapper for pages with sidebar
function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="main-content">
        <div className="container mx-auto px-4 py-8">
          {children}
        </div>
      </main>
    </div>
  );
}

// Simple layout wrapper for pages without container/padding
function SimpleAppLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Header />
      <main className="main-content">
        {children}
      </main>
    </>
  );
}

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
          <SimpleAppLayout>
            <CampaignOrchestrationDashboard />
          </SimpleAppLayout>
        } />
        <Route path="/dashboard/legacy" element={
          <SimpleAppLayout>
            <ImprovedDashboard />
          </SimpleAppLayout>
        } />
        <Route path="/orchestration" element={
          <SimpleAppLayout>
            <CampaignOrchestrationDashboard />
          </SimpleAppLayout>
        } />
        <Route path="/new" element={
          <SimpleAppLayout>
            <EnhancedBlogCreator />
          </SimpleAppLayout>
        } />
        <Route path="/edit/:blogId" element={
          <SimpleAppLayout>
            <EnhancedBlogEditor />
          </SimpleAppLayout>
        } />
        <Route path="/blogs/:blogId" element={
          <SimpleAppLayout>
            <EnhancedBlogEditor />
          </SimpleAppLayout>
        } />
        {/* Content Brief - Commented out for production */}
        {/* <Route path="/content-brief" element={
          <SimpleAppLayout>
            <ContentBriefCreator />
          </SimpleAppLayout>
        } /> */}
        <Route path="/campaign/:blogId" element={
          <AppLayout>
            <CampaignView />
          </AppLayout>
        } />
        <Route path="/campaigns/:campaignId" element={
          <AppLayout>
            <CampaignView />
          </AppLayout>
        } />
        <Route path="/campaign-wizard/:campaignId" element={
          <AppLayout>
            <CampaignWizard />
          </AppLayout>
        } />
        <Route path="/campaigns" element={
          <AppLayout>
            <Campaigns onNavigate={() => {}} />
          </AppLayout>
        } />
        {/* Analytics - Commented out for production */}
        {/* <Route path="/analytics" element={
          <SimpleAppLayout>
            <PerformanceAnalytics />
          </SimpleAppLayout>
        } />
        <Route path="/analytics/predictive" element={
          <SimpleAppLayout>
            <PredictiveAnalytics />
          </SimpleAppLayout>
        } />
        <Route path="/analytics/legacy" element={
          <AppLayout>
            <Analytics />
          </AppLayout>
        } /> */}
        {/* Testing - Commented out for production */}
        {/* <Route path="/testing" element={
          <SimpleAppLayout>
            <IntegrationTesting />
          </SimpleAppLayout>
        } /> */}
        <Route path="/knowledge-base" element={
          <AppLayout>
            <KnowledgeBase />
          </AppLayout>
        } />
        <Route path="/settings" element={
          <AppLayout>
            <Settings />
          </AppLayout>
        } />
        <Route path="/workflow" element={<Navigate to="/orchestration" replace />} />
        <Route path="/image-agent" element={<Navigate to="/campaigns" replace />} />
        {/* Agents - Commented out for production */}
        {/* <Route path="/agents" element={
          <SimpleAppLayout>
            <AgentManagement />
          </SimpleAppLayout>
        } /> */}
        {/* Monitoring - Commented out for production */}
        {/* <Route path="/monitoring" element={
          <RealTimeMonitoring />
        } /> */}
        
        {/* Competitor Intelligence Routes - Commented out for production */}
        {/* <Route path="/competitor-intelligence" element={
          <SimpleAppLayout>
            <CompetitorIntelligence />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/competitors" element={
          <SimpleAppLayout>
            <CompetitorManagement />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/competitors/new" element={
          <SimpleAppLayout>
            <AddCompetitor />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/competitors/:id" element={
          <SimpleAppLayout>
            <CompetitorDetail />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/competitors/:id/edit" element={
          <SimpleAppLayout>
            <EditCompetitor />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/dashboard" element={
          <SimpleAppLayout>
            <CompetitorIntelligence />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/analytics" element={
          <SimpleAppLayout>
            <AnalyticsDashboard />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/alerts" element={
          <SimpleAppLayout>
            <AlertsOverview />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/ai-analysis" element={
          <SimpleAppLayout>
            <AIContentAnalysis />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/alerts/:alertId" element={
          <SimpleAppLayout>
            <AlertDetail />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/reporting" element={
          <SimpleAppLayout>
            <AdvancedReporting />
          </SimpleAppLayout>
        } />
        <Route path="/competitor-intelligence/integrations" element={
          <SimpleAppLayout>
            <IntegrationManagement />
          </SimpleAppLayout>
        } /> */}
        
        {/* Content Review is now integrated into Campaign Details - no separate route needed */}
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