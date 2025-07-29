import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import { LandingPage } from './pages/LandingPage';
import { ImprovedDashboard } from './pages/ImprovedDashboard';
import { EnhancedBlogCreator } from './pages/EnhancedBlogCreator';
import { EnhancedBlogEditor } from './pages/EnhancedBlogEditor';
import { KnowledgeBase } from './pages/KnowledgeBase';
import { CampaignView } from './pages/CampaignView';
import { Analytics } from './pages/Analytics';

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
      </Routes>
    </Router>
  );
}

export default App;
