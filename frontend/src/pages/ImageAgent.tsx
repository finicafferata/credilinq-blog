import React, { useState, useEffect } from 'react';
import { ArrowLeftIcon } from '@heroicons/react/24/outline';
import ImageAgentPanel from '../components/ImageAgentPanel';

interface ImageData {
  id: string;
  prompt: string;
  url: string;
  alt_text: string;
  style: string;
  size: string;
}

const ImageAgent: React.FC = () => {
  const [blogTitle, setBlogTitle] = useState('');
  const [content, setContent] = useState('');
  const [outline, setOutline] = useState<string[]>([]);
  const [generatedImages, setGeneratedImages] = useState<ImageData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedBlogInfo, setSelectedBlogInfo] = useState<{title: string, content: string} | null>(null);

  // Load data from localStorage if it exists
  useEffect(() => {
    const savedTitle = localStorage.getItem('blog_title');
    const savedContent = localStorage.getItem('blog_content');
    const savedOutline = localStorage.getItem('blog_outline');

    if (savedTitle) setBlogTitle(savedTitle);
    if (savedContent) setContent(savedContent);
    if (savedOutline) {
      try {
        setOutline(JSON.parse(savedOutline));
      } catch (e) {
        console.error('Error parsing outline:', e);
      }
    }
  }, []);

  const handleImagesGenerated = (images: ImageData[]) => {
    setGeneratedImages(images);
    // Save to localStorage
    localStorage.setItem('generated_images', JSON.stringify(images));
  };

  const handleBack = () => {
    window.history.back();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBack}
                className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
              >
                <ArrowLeftIcon className="h-6 w-6" />
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Image Agent</h1>
                <p className="text-sm text-gray-600">Generate professional images for your content</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Configuration Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Blog Configuration
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Blog Title
                  </label>
                  <input
                    type="text"
                    value={blogTitle}
                    onChange={(e) => {
                      setBlogTitle(e.target.value);
                      localStorage.setItem('blog_title', e.target.value);
                    }}
                    placeholder="Enter your blog title"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Blog Content
                  </label>
                  <textarea
                    value={content}
                    onChange={(e) => {
                      setContent(e.target.value);
                      localStorage.setItem('blog_content', e.target.value);
                    }}
                    placeholder="Paste your blog content here..."
                    rows={8}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Outline (Opcional)
                  </label>
                  <textarea
                    value={outline.join('\n')}
                    onChange={(e) => {
                      const lines = e.target.value.split('\n').filter(line => line.trim());
                      setOutline(lines);
                      localStorage.setItem('blog_outline', JSON.stringify(lines));
                    }}
                    placeholder="One section per line..."
                    rows={4}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Selected Blog Information */}
            {selectedBlogInfo && (
              <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Selected Blog
                </h3>
                <div className="space-y-3">
                  <div>
                    <span className="text-sm font-medium text-gray-600">Title:</span>
                    <p className="text-gray-900 font-medium">{selectedBlogInfo.title}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-600">Content:</span>
                    <p className="text-gray-700 text-sm line-clamp-3">
                      {selectedBlogInfo.content.substring(0, 200)}...
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Statistics */}
            {generatedImages.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Statistics
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Generated images:</span>
                    <span className="font-semibold">{generatedImages.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Styles used:</span>
                    <span className="font-semibold">
                      {[...new Set(generatedImages.map(img => img.style))].length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Sizes:</span>
                    <span className="font-semibold">
                      {[...new Set(generatedImages.map(img => img.size))].length}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Image Agent Panel */}
          <div className="lg:col-span-2">
            <ImageAgentPanel
              blogTitle={blogTitle}
              content={content}
              outline={outline}
              onImagesGenerated={handleImagesGenerated}
            />
          </div>
        </div>

        {/* Generated Images Gallery */}
        {generatedImages.length > 0 && (
          <div className="mt-8">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Image Gallery
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {generatedImages.map((image) => (
                  <div key={image.id} className="bg-gray-50 rounded-lg p-3">
                    <img
                      src={image.url}
                      alt={image.alt_text}
                      className="w-full h-32 object-cover rounded-lg mb-2"
                    />
                    <p className="text-xs text-gray-600 line-clamp-2">
                      {image.prompt}
                    </p>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs text-gray-500 capitalize">
                        {image.style}
                      </span>
                      <span className="text-xs text-gray-500">
                        {image.size}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageAgent; 