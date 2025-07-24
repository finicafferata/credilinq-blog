import { useState, useRef } from 'react';
import { documentsApi } from '../lib/api';

interface UploadedFile {
  name: string;
  status: 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
  documentId?: string;
}

export function KnowledgeBase() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (fileList: FileList) => {
    const newFiles = Array.from(fileList).map(file => ({
      name: file.name,
      status: 'uploading' as const,
      progress: 0,
    }));

    setFiles(prev => [...prev, ...newFiles]);

    Array.from(fileList).forEach((file, index) => {
      uploadFile(file, files.length + index);
    });
  };

  const uploadFile = async (file: File, fileIndex: number) => {
    try {
      setFiles(prev => prev.map((f, i) => 
        i === fileIndex ? { ...f, progress: 50 } : f
      ));

      const result = await documentsApi.upload(file, file.name);

      setFiles(prev => prev.map((f, i) => 
        i === fileIndex 
          ? { ...f, status: 'success', progress: 100, documentId: result.document_id }
          : f
      ));
    } catch (error) {
      setFiles(prev => prev.map((f, i) => 
        i === fileIndex 
          ? { ...f, status: 'error', progress: 0, error: 'Upload failed' }
          : f
      ));
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const clearFiles = () => {
    setFiles([]);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Knowledge Base</h1>
        <p className="text-gray-600 mt-2">
          Upload documents to enhance your AI agent's knowledge for better blog generation
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Documents</h3>
            
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive 
                  ? 'border-primary-500 bg-primary-50' 
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              
              <h4 className="text-lg font-medium text-gray-900 mb-2">
                Drop files here or click to browse
              </h4>
              <p className="text-gray-600 mb-4">
                Supports .txt and .md files up to 10MB
              </p>
              
              <button
                onClick={() => fileInputRef.current?.click()}
                className="btn-primary"
              >
                Choose Files
              </button>
              
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.md"
                onChange={handleFileInputChange}
                className="hidden"
              />
            </div>
          </div>

          {files.length > 0 && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Upload Progress</h3>
                <button
                  onClick={clearFiles}
                  className="text-sm text-gray-500 hover:text-red-600"
                >
                  Clear All
                </button>
              </div>
              
              <div className="space-y-3">
                {files.map((file, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-gray-900 truncate">
                          {file.name}
                        </span>
                        <span className={`text-xs ${
                          file.status === 'success' ? 'text-green-600' :
                          file.status === 'error' ? 'text-red-600' :
                          'text-blue-600'
                        }`}>
                          {file.status === 'success' ? 'Processed' :
                           file.status === 'error' ? 'Failed' :
                           'Processing...'}
                        </span>
                      </div>
                      
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-300 ${
                            file.status === 'success' ? 'bg-green-500' :
                            file.status === 'error' ? 'bg-red-500' :
                            'bg-blue-500'
                          }`}
                          style={{ width: `${file.progress}%` }}
                        />
                      </div>
                      
                      {file.error && (
                        <p className="text-xs text-red-600 mt-1">{file.error}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">How It Works</h3>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                  1
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Upload Documents</h4>
                  <p className="text-sm text-gray-600">
                    Add your company documents, product specs, case studies, and other relevant materials.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                  2
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">AI Processing</h4>
                  <p className="text-sm text-gray-600">
                    Our system breaks down documents into chunks and creates embeddings for semantic search.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                  3
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Enhanced Blogs</h4>
                  <p className="text-sm text-gray-600">
                    When creating blogs, the AI will use this knowledge to create more accurate and relevant content.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Best Practices</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start space-x-2">
                <span className="text-primary-500">•</span>
                <span>Use clear, descriptive filenames for your documents</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-primary-500">•</span>
                <span>Include product documentation, whitepapers, and case studies</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-primary-500">•</span>
                <span>Keep documents focused and well-structured</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-primary-500">•</span>
                <span>Regularly update your knowledge base with new content</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 