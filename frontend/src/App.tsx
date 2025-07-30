import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import { LandingPage } from './pages/LandingPage';
import { ImprovedDashboard } from './pages/ImprovedDashboard';
import { EnhancedBlogCreator } from './pages/EnhancedBlogCreator';
import { EnhancedBlogEditor } from './pages/EnhancedBlogEditor';
import { KnowledgeBase } from './pages/KnowledgeBase';
import { CampaignView } from './pages/CampaignView';
import Campaigns from './pages/Campaigns';
import { CampaignWizard } from './pages/CampaignWizard';
import { Analytics } from './pages/Analytics';
import WorkflowPage from './pages/WorkflowPage';
import ImageAgent from './pages/ImageAgent';
import SimpleImageTest from './components/SimpleImageTest';
import DebugImageTest from './components/DebugImageTest';
import StaticImageTest from './components/StaticImageTest';

function App() {
  return (
    <Router>
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
        <Route path="/image-test" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <SimpleImageTest />
            </main>
          </div>
        } />
        <Route path="/debug-image" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <DebugImageTest />
            </main>
          </div>
        } />
        <Route path="/static-image" element={
          <div className="min-h-screen bg-gray-50">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <StaticImageTest />
            </main>
          </div>
        } />
      </Routes>
    </Router>
  );
}

export default App;
