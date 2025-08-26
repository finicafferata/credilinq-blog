import { useState, useRef, useEffect } from 'react';

interface UploadedFile {
  name: string;
  status: 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
  documentId?: string;
}

interface Document {
  id: string;
  filename: string;
  file_size: number;
  mime_type: string;
  description?: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export function KnowledgeBase() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch('/api/documents');
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents);
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    } finally {
      setLoading(false);
    }
  };

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
        i === fileIndex ? { ...f, progress: 20 } : f
      ));

      // Create FormData for file upload
      const formData = new FormData();
      formData.append('files', file);
      formData.append('description', `Uploaded document: ${file.name}`);

      setFiles(prev => prev.map((f, i) => 
        i === fileIndex ? { ...f, progress: 50 } : f
      ));

      // Upload to backend
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();
      const uploadResult = result[0]; // First file result

      setFiles(prev => prev.map((f, i) => 
        i === fileIndex 
          ? { 
              ...f, 
              status: 'success', 
              progress: 100, 
              documentId: uploadResult.id 
            }
          : f
      ));

      // Refresh documents list
      setTimeout(() => {
        fetchDocuments();
      }, 1000); // Give backend time to process

    } catch (error) {
      console.error('Upload error:', error);
      setFiles(prev => prev.map((f, i) => 
        i === fileIndex 
          ? { 
              ...f, 
              status: 'error', 
              progress: 0, 
              error: error instanceof Error ? error.message : 'Upload failed' 
            }
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
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Knowledge Base</h1>
              <p className="mt-2 text-gray-600">Empower your AI agent with domain expertise by uploading your documents. Transform your content into intelligent, contextually-aware blog posts.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">

        <div className="grid gap-8 lg:grid-cols-12">
        {/* Upload Section */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8 hover:shadow-xl transition-all duration-300">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl flex items-center justify-center mr-4">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-900">Upload Documents</h3>
            </div>
            
            <div
              className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${
                dragActive 
                  ? 'border-blue-400 bg-blue-50 scale-105' 
                  : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className={`w-20 h-20 mx-auto mb-6 rounded-2xl flex items-center justify-center transition-all duration-300 ${
                dragActive ? 'bg-blue-100' : 'bg-gradient-to-br from-gray-50 to-gray-100'
              }`}>
                <svg className={`w-10 h-10 transition-all duration-300 ${
                  dragActive ? 'text-blue-500 scale-110' : 'text-gray-400'
                }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              
              <h4 className="text-xl font-semibold text-gray-900 mb-2">
                {dragActive ? 'Drop your files here!' : 'Drop files here or click to browse'}
              </h4>
              <p className="text-gray-600 mb-6 leading-relaxed">
                Supports <span className="font-medium text-blue-600">.txt</span>, <span className="font-medium text-blue-600">.md</span>, <span className="font-medium text-blue-600">.pdf</span>, and <span className="font-medium text-blue-600">.csv</span> files up to <span className="font-medium">10MB</span>
              </p>
              
              <button
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-medium rounded-xl hover:from-blue-700 hover:to-indigo-700 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
                Choose Files
              </button>
              
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.md,.pdf,.csv"
                onChange={handleFileInputChange}
                className="hidden"
              />
            </div>
          </div>

          {files.length > 0 && (
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 hover:shadow-xl transition-all duration-300">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center mr-3">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-bold text-gray-900">Upload Progress</h3>
                </div>
                <button
                  onClick={clearFiles}
                  className="text-sm text-gray-500 hover:text-red-600 font-medium px-3 py-1 rounded-lg hover:bg-red-50 transition-colors"
                >
                  Clear All
                </button>
              </div>
              
              <div className="space-y-4">
                {files.map((file, index) => (
                  <div key={index} className="bg-gray-50 rounded-xl p-4 border border-gray-100">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                          file.status === 'success' ? 'bg-green-100' :
                          file.status === 'error' ? 'bg-red-100' :
                          'bg-blue-100'
                        }`}>
                          {file.status === 'success' ? (
                            <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          ) : file.status === 'error' ? (
                            <svg className="w-4 h-4 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          ) : (
                            <svg className="w-4 h-4 text-blue-600 animate-spin" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                          )}
                        </div>
                        <div>
                          <span className="font-medium text-gray-900 truncate block max-w-[200px]">
                            {file.name}
                          </span>
                          <span className={`text-xs font-medium ${
                            file.status === 'success' ? 'text-green-600' :
                            file.status === 'error' ? 'text-red-600' :
                            'text-blue-600'
                          }`}>
                            {file.status === 'success' ? '✨ Processed & Ready' :
                             file.status === 'error' ? '❌ Processing Failed' :
                             '🔄 Processing...'}
                          </span>
                        </div>
                      </div>
                      <span className="text-sm font-bold text-gray-600">
                        {file.progress}%
                      </span>
                    </div>
                    
                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-2 rounded-full transition-all duration-500 ${
                          file.status === 'success' ? 'bg-gradient-to-r from-green-400 to-green-500' :
                          file.status === 'error' ? 'bg-gradient-to-r from-red-400 to-red-500' :
                          'bg-gradient-to-r from-blue-400 to-blue-500'
                        }`}
                        style={{ width: `${file.progress}%` }}
                      />
                    </div>
                    
                    {file.error && (
                      <p className="text-sm text-red-600 mt-2 bg-red-50 rounded-lg p-2 border border-red-200">
                        {file.error}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Documents List Section */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8 hover:shadow-xl transition-all duration-300">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mr-4">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900">Uploaded Documents</h3>
                <p className="text-sm text-gray-500">Your knowledge base library</p>
              </div>
            </div>
            
            {loading ? (
              <div className="text-center py-12">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-2xl mb-4">
                  <svg className="animate-spin w-8 h-8 text-blue-600" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                </div>
                <p className="text-gray-600 font-medium">Loading your documents...</p>
              </div>
            ) : documents.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-gray-100 to-gray-200 rounded-2xl flex items-center justify-center">
                  <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h4 className="text-lg font-semibold text-gray-900 mb-2">No documents yet</h4>
                <p className="text-gray-600 mb-4">Upload your first document to build your knowledge base</p>
                <div className="text-sm text-gray-500 bg-blue-50 rounded-xl p-4 border border-blue-100">
                  💡 <span className="font-medium">Tip:</span> Start with your most important company documents, product guides, or industry resources
                </div>
              </div>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                {documents.map((doc) => (
                  <div key={doc.id} className="group bg-gradient-to-r from-gray-50 to-gray-100 hover:from-blue-50 hover:to-indigo-50 rounded-xl p-4 border border-gray-200 hover:border-blue-200 transition-all duration-200 hover:shadow-md">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 flex-1 min-w-0">
                        <div className="flex-shrink-0">
                          <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                            doc.mime_type === 'application/pdf' ? 'bg-red-100 text-red-600' :
                            doc.mime_type?.includes('text') ? 'bg-blue-100 text-blue-600' :
                            'bg-green-100 text-green-600'
                          }`}>
                            {doc.mime_type === 'application/pdf' ? (
                              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
                              </svg>
                            ) : (
                              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                              </svg>
                            )}
                          </div>
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-gray-900 truncate group-hover:text-blue-900 transition-colors">
                            {doc.filename}
                          </p>
                          <div className="flex items-center space-x-3 mt-1">
                            <span className="text-sm text-gray-500">
                              {(doc.file_size / 1024).toFixed(1)} KB
                            </span>
                            <span className="text-gray-300">•</span>
                            <span className="text-sm text-gray-500">
                              {new Date(doc.created_at).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex-shrink-0 ml-4">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold ${
                          doc.status === 'completed' ? 'bg-green-100 text-green-700 border border-green-200' :
                          doc.status === 'processing' ? 'bg-yellow-100 text-yellow-700 border border-yellow-200' :
                          doc.status === 'failed' ? 'bg-red-100 text-red-700 border border-red-200' :
                          'bg-gray-100 text-gray-700 border border-gray-200'
                        }`}>
                          {doc.status === 'completed' ? '✅ Ready' :
                           doc.status === 'processing' ? '⏳ Processing' :
                           doc.status === 'failed' ? '❌ Failed' :
                           doc.status}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* How It Works & Best Practices Section */}
        <div className="lg:col-span-3 space-y-6">
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8 hover:shadow-xl transition-all duration-300">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mr-4">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900">How It Works</h3>
                <p className="text-sm text-gray-500">Simple 3-step process</p>
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="relative">
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg">
                    1
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900 mb-2">📄 Upload Documents</h4>
                    <p className="text-sm text-gray-600 leading-relaxed">
                      Add your company documents, product specs, case studies, and other relevant materials to build your knowledge foundation.
                    </p>
                  </div>
                </div>
                <div className="absolute left-5 top-10 w-0.5 h-6 bg-gradient-to-b from-blue-300 to-purple-300"></div>
              </div>
              
              <div className="relative">
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-indigo-500 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg">
                    2
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900 mb-2">🤖 AI Processing</h4>
                    <p className="text-sm text-gray-600 leading-relaxed">
                      Our intelligent system breaks down documents into semantic chunks and creates vector embeddings for advanced search capabilities.
                    </p>
                  </div>
                </div>
                <div className="absolute left-5 top-10 w-0.5 h-6 bg-gradient-to-b from-purple-300 to-emerald-300"></div>
              </div>
              
              <div className="flex items-start space-x-4">
                <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg">
                  3
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-gray-900 mb-2">✨ Enhanced Blogs</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    When creating blogs, the AI leverages this knowledge to generate more accurate, contextual, and relevant content.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8 hover:shadow-xl transition-all duration-300">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-r from-amber-500 to-orange-500 rounded-xl flex items-center justify-center mr-4">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900">Best Practices</h3>
                <p className="text-sm text-gray-500">Optimize your knowledge base</p>
              </div>
            </div>
            
            <ul className="space-y-4">
              <li className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
                  <svg className="w-3 h-3 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
                <span className="text-sm text-gray-700 leading-relaxed">Use clear, descriptive filenames that reflect document content</span>
              </li>
              <li className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
                  <svg className="w-3 h-3 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
                <span className="text-sm text-gray-700 leading-relaxed">Include product documentation, whitepapers, and detailed case studies</span>
              </li>
              <li className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
                  <svg className="w-3 h-3 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
                <span className="text-sm text-gray-700 leading-relaxed">Keep documents focused, well-structured, and regularly updated</span>
              </li>
              <li className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-amber-100 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
                  <svg className="w-3 h-3 text-amber-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
                <span className="text-sm text-gray-700 leading-relaxed">Continuously refresh your knowledge base with new industry insights</span>
              </li>
            </ul>

            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100">
              <div className="flex items-center space-x-2 mb-2">
                <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-sm font-semibold text-blue-900">Pro Tip</span>
              </div>
              <p className="text-xs text-blue-800 leading-relaxed">
                Start with 3-5 of your most important documents. Quality over quantity leads to better AI-generated content.
              </p>
            </div>
          </div>
        </div>
        </div>

        {/* Optional: Add floating elements for visual enhancement */}
        <div className="fixed top-20 left-10 w-32 h-32 bg-gradient-to-r from-blue-400/10 to-purple-400/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="fixed bottom-20 right-10 w-40 h-40 bg-gradient-to-r from-emerald-400/10 to-cyan-400/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>
    </div>
  );
}

export default KnowledgeBase; 