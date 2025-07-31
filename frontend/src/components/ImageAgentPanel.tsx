import React, { useState, useEffect } from 'react';
import { 
  PhotoIcon, 
  ArrowPathIcon, 
  EyeIcon, 
  ArrowDownTrayIcon,
  SparklesIcon,
  CheckIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';

// API configuration - same logic as api.ts
const isDev = import.meta.env.DEV;
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 
  (isDev ? 'http://localhost:8000' : 'https://credilinq-blog-production.up.railway.app');

interface ImageData {
  id: string;
  prompt: string;
  url: string;
  alt_text: string;
  style: string;
  size: string;
}

interface ImageGenerationRequest {
  content?: string;
  blog_title?: string;
  blog_id?: string;
  outline?: string[];
  style: string;
  count: number;
}

interface BlogOption {
  id: string;
  title: string;
  status: string;
  created_at: string;
}

interface ImageAgentPanelProps {
  workflowId?: string;
  blogTitle?: string;
  content?: string;
  outline?: string[];
  onImagesGenerated?: (images: ImageData[]) => void;
}

const ImageAgentPanel: React.FC<ImageAgentPanelProps> = ({
  workflowId,
  blogTitle,
  content,
  outline,
  onImagesGenerated
}) => {
  const [images, setImages] = useState<ImageData[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [imagesLoading, setImagesLoading] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState('professional');
  const [imageCount, setImageCount] = useState(3);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableBlogs, setAvailableBlogs] = useState<BlogOption[]>([]);
  const [selectedBlogId, setSelectedBlogId] = useState<string>('');
  const [useExistingBlog, setUseExistingBlog] = useState(false);

  const styles = [
    { id: 'professional', name: 'Profesional', description: 'Diseño limpio y corporativo' },
    { id: 'creative', name: 'Creativo', description: 'Estilo artístico y llamativo' },
    { id: 'minimalist', name: 'Minimalista', description: 'Diseño simple y elegante' },
    { id: 'modern', name: 'Moderno', description: 'Tendencias actuales de diseño' },
    { id: 'vintage', name: 'Vintage', description: 'Estilo retro y clásico' }
  ];

  // Cargar blogs disponibles al montar el componente
  useEffect(() => {
    const fetchAvailableBlogs = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/images/blogs`);
        if (response.ok) {
          const data = await response.json();
          const blogs = data.blogs || [];
          setAvailableBlogs(blogs);
          console.log(`Loaded ${blogs.length} blogs for image generation`);
        } else {
          console.error('Error fetching blogs:', response.status, response.statusText);
          setError(`Error cargando blogs: ${response.status}`);
        }
      } catch (err) {
        console.error('Error fetching blogs:', err);
        setError('Error de conexión al cargar blogs');
      }
    };

    fetchAvailableBlogs();
  }, []);



  const generateImages = async () => {
    // Validar que tengamos contenido o un blog seleccionado
    if (useExistingBlog) {
      if (!selectedBlogId) {
        setError('Selecciona un blog existente');
        return;
      }
    } else {
      if (!blogTitle || !content) {
        setError('Title and content are required to generate images');
        return;
      }
    }

    setIsGenerating(true);
    setError(null);

    try {
      const request: ImageGenerationRequest = {
        style: selectedStyle,
        count: imageCount,
        outline
      };

      // Agregar contenido según el modo seleccionado
      if (useExistingBlog) {
        request.blog_id = selectedBlogId;
      } else {
        request.content = content;
        request.blog_title = blogTitle;
      }

      let response;
      
      if (workflowId) {
        // Usar el workflow existente
        response = await fetch(`${apiBaseUrl}/api/workflow/image`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow_id: workflowId })
        });
      } else {
        // Llamada directa al agente de imágenes
        response = await fetch(`${apiBaseUrl}/api/images/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request)
        });
      }

      if (!response.ok) {
        throw new Error('Error al generar imágenes');
      }

      const data = await response.json();
      console.log('Response from backend:', data);
      console.log('Response structure:', Object.keys(data));
      console.log('Images key exists:', 'images' in data);
      console.log('Data.images type:', typeof data.images);
      console.log('Data.images is array:', Array.isArray(data.images));
      
      const generatedImages = data.images || data.data?.images || [];
      console.log('Generated images:', generatedImages);
      console.log('Generated images type:', typeof generatedImages);
      console.log('Generated images is array:', Array.isArray(generatedImages));
      console.log('Generated images length:', generatedImages.length);
      
      // Log each image URL to debug
      generatedImages.forEach((image: ImageData, index: number) => {
        console.log(`Image ${index + 1} full object:`, image);
        console.log(`Image ${index + 1} keys:`, Object.keys(image));
        console.log(`Image ${index + 1} URL:`, image.url);
        console.log(`Image ${index + 1} URL type:`, typeof image.url);
        console.log(`Image ${index + 1} URL starts with data:`, image.url?.startsWith('data:'));
        console.log(`Image ${index + 1} URL length:`, image.url?.length);
        console.log(`Image ${index + 1} URL first 100 chars:`, image.url?.substring(0, 100));
        
        // Log URL length for debugging
        if (image.url) {
          console.log(`Image ${index + 1} URL length: ${image.url.length} characters`);
        }
      });
      
      setImagesLoading(true);
      setImages(generatedImages);
      onImagesGenerated?.(generatedImages);
      
      // Set loading to false after a short delay
      setTimeout(() => {
        setImagesLoading(false);
      }, 500);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setIsGenerating(false);
    }
  };

  const regenerateImage = async (imageId: string) => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/api/images/regenerate/${imageId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          style: selectedStyle,
          blog_title: blogTitle,
          content
        })
      });

      if (!response.ok) {
        throw new Error('Error al regenerar imagen');
      }

      const data = await response.json();
      const newImage = data.image;

      setImages(prev => prev.map(img => 
        img.id === imageId ? newImage : img
      ));

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadImage = async (image: ImageData) => {
    try {
      const response = await fetch(image.url);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `image-${image.id}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('Error al descargar la imagen');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <PhotoIcon className="h-8 w-8 text-blue-600" />
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Generador de Imágenes</h3>
            <p className="text-sm text-gray-600">Create professional images for your content</p>
          </div>
        </div>
        <SparklesIcon className="h-6 w-6 text-purple-500" />
      </div>

      {/* Modo de Contenido */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Content Source
        </label>
        <div className="flex space-x-4">
          <label className="flex items-center">
            <input
              type="radio"
              checked={!useExistingBlog}
              onChange={() => setUseExistingBlog(false)}
              className="mr-2"
            />
            <span className="text-sm">New content</span>
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              checked={useExistingBlog}
              onChange={() => setUseExistingBlog(true)}
              className="mr-2"
            />
            <span className="text-sm">Blog existente</span>
          </label>
        </div>
      </div>

      {/* Selección de Blog Existente */}
      {useExistingBlog && (
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Seleccionar Blog
          </label>
          {availableBlogs.length > 0 ? (
            <select
              value={selectedBlogId}
              onChange={(e) => setSelectedBlogId(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Selecciona un blog...</option>
              {availableBlogs.map(blog => (
                <option key={blog.id} value={blog.id}>
                  {blog.title} ({blog.status})
                </option>
              ))}
            </select>
          ) : (
            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-md">
              <div className="flex items-center space-x-2">
                <svg className="h-5 w-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <span className="text-yellow-700">
                  No blogs available. Create some blogs first or use "New content".
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Configuración */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Estilo de Imagen
          </label>
          <select
            value={selectedStyle}
            onChange={(e) => setSelectedStyle(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {styles.map(style => (
              <option key={style.id} value={style.id}>
                {style.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Cantidad de Imágenes
          </label>
          <select
            value={imageCount}
            onChange={(e) => setImageCount(Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={1}>1 imagen</option>
            <option value={2}>2 imágenes</option>
            <option value={3}>3 imágenes</option>
            <option value={5}>5 imágenes</option>
          </select>
        </div>

        <div className="flex items-end">
          <button
            onClick={generateImages}
            disabled={isGenerating || (useExistingBlog ? !selectedBlogId : (!blogTitle || !content))}
            className="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {isGenerating ? (
              <>
                <ArrowPathIcon className="h-5 w-5 animate-spin" />
                <span>Generando...</span>
              </>
            ) : (
              <>
                <PhotoIcon className="h-5 w-5" />
                <span>
                  {useExistingBlog 
                    ? `Generar Imágenes para Blog Seleccionado`
                    : 'Generar Imágenes'
                  }
                </span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-center space-x-2">
            <XMarkIcon className="h-5 w-5 text-red-500" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Imágenes Generadas */}
      {images.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-lg font-medium text-gray-900">
            Imágenes Generadas ({images.length})
            {imagesLoading && <span className="text-blue-600 ml-2">(Cargando...)</span>}
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {images.map((image) => (
              <div key={image.id} className="bg-gray-50 rounded-lg p-4">
                <div className="relative group">
                  <img
                    src={image.url}
                    alt={image.alt_text}
                    className="w-full h-48 object-cover rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                    onClick={() => {
                      setSelectedImage(image);
                      setShowModal(true);
                    }}
                    onLoad={() => console.log(`Image loaded successfully: ${image.id}`)}
                    onError={(e) => {
                      console.error(`Image failed to load: ${image.id}`);
                      console.error(`Failed URL: ${image.url}`);
                    }}
                  />
                  
                  {/* Overlay con acciones */}
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-200 rounded-lg flex items-center justify-center">
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex space-x-2">
                      <button
                        onClick={() => setSelectedImage(image)}
                        className="p-2 bg-white rounded-full shadow-lg hover:bg-gray-100"
                        title="Ver imagen"
                      >
                        <EyeIcon className="h-4 w-4 text-gray-700" />
                      </button>
                                             <button
                         onClick={() => downloadImage(image)}
                         className="p-2 bg-white rounded-full shadow-lg hover:bg-gray-100"
                         title="Descargar"
                       >
                         <ArrowDownTrayIcon className="h-4 w-4 text-gray-700" />
                       </button>
                      <button
                        onClick={() => regenerateImage(image.id)}
                        disabled={isGenerating}
                        className="p-2 bg-white rounded-full shadow-lg hover:bg-gray-100 disabled:opacity-50"
                        title="Regenerar"
                      >
                        <ArrowPathIcon className="h-4 w-4 text-gray-700" />
                      </button>
                    </div>
                  </div>
                </div>
                
                <div className="mt-3">
                  <p className="text-sm text-gray-600 line-clamp-2">
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
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Modal para vista previa */}
      {showModal && selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-semibold">Vista Previa de Imagen</h3>
              <button
                onClick={() => setShowModal(false)}
                className="p-2 hover:bg-gray-100 rounded-full"
              >
                <XMarkIcon className="h-6 w-6" />
              </button>
            </div>
            
            <div className="p-6">
              <img
                src={selectedImage.url}
                alt={selectedImage.alt_text}
                className="w-full max-h-[60vh] object-contain rounded-lg"
              />
              
              <div className="mt-4 space-y-2">
                <p className="text-sm text-gray-600">
                  <strong>Prompt:</strong> {selectedImage.prompt}
                </p>
                <p className="text-sm text-gray-600">
                  <strong>Estilo:</strong> {selectedImage.style}
                </p>
                <p className="text-sm text-gray-600">
                  <strong>Tamaño:</strong> {selectedImage.size}
                </p>
              </div>
              
              <div className="flex space-x-3 mt-4">
                                 <button
                   onClick={() => downloadImage(selectedImage)}
                   className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                 >
                   <ArrowDownTrayIcon className="h-4 w-4" />
                   <span>Descargar</span>
                 </button>
                <button
                  onClick={() => {
                    setShowModal(false);
                    regenerateImage(selectedImage.id);
                  }}
                  disabled={isGenerating}
                  className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50"
                >
                  <ArrowPathIcon className="h-4 w-4" />
                  <span>Regenerar</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageAgentPanel; 