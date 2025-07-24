import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import { Dashboard } from './pages/Dashboard';
import { BlogEditor } from './pages/BlogEditor';
import { KnowledgeBase } from './pages/KnowledgeBase';
import { NewBlog } from './pages/NewBlog';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/new" element={<NewBlog />} />
            <Route path="/edit/:blogId" element={<BlogEditor />} />
            <Route path="/knowledge-base" element={<KnowledgeBase />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
