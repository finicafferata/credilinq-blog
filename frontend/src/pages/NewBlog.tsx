import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { blogApi } from '../lib/api';

export function NewBlog() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    title: '',
    company_context: 'Credilinq.ai is a fintech leader in embedded lending and B2B credit solutions across Southeast Asia. We help businesses access funding through embedded financial products and innovative credit infrastructure.',
    content_type: 'blog' as 'blog' | 'linkedin',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.title.trim()) return;

    try {
      setLoading(true);
      const newBlog = await blogApi.create(formData);
      navigate(`/edit/${newBlog.id}`);
    } catch (error) {
      console.error('Failed to create blog:', error);
      alert('Failed to create blog. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Create New Content</h1>
        <p className="text-gray-600 mt-2">
          Let our AI agents create professional content for LinkedIn or comprehensive blog posts
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="card">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Content Type
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setFormData({ ...formData, content_type: 'blog' })}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    formData.content_type === 'blog'
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium">📝 Blog Post</div>
                  <div className="text-sm mt-1 opacity-75">
                    Comprehensive, detailed articles (1500-2500 words)
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => setFormData({ ...formData, content_type: 'linkedin' })}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    formData.content_type === 'linkedin'
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium">💼 LinkedIn Post</div>
                  <div className="text-sm mt-1 opacity-75">
                    Professional, engaging posts (800-1200 words)
                  </div>
                </button>
              </div>
            </div>

            <div>
              <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-2">
                {formData.content_type === 'linkedin' ? 'Post Title' : 'Blog Title'}
              </label>
              <input
                type="text"
                id="title"
                className="input"
                placeholder="e.g., How Embedded Lending is Transforming B2B Payments"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                required
              />
            </div>

            <div>
              <label htmlFor="context" className="block text-sm font-medium text-gray-700 mb-2">
                Company Context
              </label>
              <textarea
                id="context"
                rows={6}
                className="textarea"
                placeholder="Describe your company, products, and target audience..."
                value={formData.company_context}
                onChange={(e) => setFormData({ ...formData, company_context: e.target.value })}
                required
              />
              <p className="mt-2 text-sm text-gray-500">
                This context helps our AI understand your company's voice and focus areas.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="btn-secondary"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading || !formData.title.trim()}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Creating...</span>
              </div>
            ) : (
              formData.content_type === 'linkedin' ? 'Create LinkedIn Post' : 'Create Blog'
            )}
          </button>
        </div>
      </form>

      <div className="mt-8 p-4 bg-blue-50 rounded-lg">
        <h3 className="text-sm font-medium text-blue-900 mb-2">How it works:</h3>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Our Planner Agent creates a structured outline optimized for {formData.content_type === 'linkedin' ? 'LinkedIn engagement' : 'comprehensive blog content'}</li>
          <li>• The Researcher Agent finds relevant information from your knowledge base</li>
          <li>• The Writer Agent crafts content tailored for {formData.content_type === 'linkedin' ? 'professional social media' : 'in-depth blog reading'}</li>
          <li>• The Editor Agent reviews and refines the content for quality and platform optimization</li>
        </ul>
      </div>
    </div>
  );
} 